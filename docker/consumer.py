# -*- coding: utf-8 -*-
"""
consumer.py — Consumidor de inferencia en tiempo real
=====================================================
Se suscribe al broker MQTT, mantiene el estado por motor con StreamingFeatureBuilder,
y cada vez que un vuelo (ciclo) se cierra calcula sus características, predice el RUL con
el modelo entrenado (por defecto el cuantil 0.25, conservador), agrega por mediana y
guarda el resultado en SQLite, además de imprimirlo por consola.

Es la pieza central de la arquitectura: convierte un flujo de lecturas sueltas en una
predicción de RUL por vuelo completado, exactamente con las mismas features que el batch.

Uso:
    python consumer.py
    python consumer.py --model modelos/xgb_flota_q0.10.joblib   # máxima seguridad

Requisitos: pip install paho-mqtt numpy joblib scikit-learn xgboost
Broker: MQTT en localhost:1883 (p.ej. mosquitto). Arráncalo ANTES que el productor.
"""
import argparse, json, sqlite3, time
import numpy as np, joblib
import paho.mqtt.client as mqtt
from streaming_features import StreamingFeatureBuilder


def init_db(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")          # lectores y escritor a la vez, sin bloqueos
    con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
        engine_id TEXT, cycle INTEGER, rul_pred REAL, rul_real REAL,
        n_obs INTEGER, ts REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pred ON predicciones(engine_id, cycle)")
    con.commit()
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--broker', default='localhost'); ap.add_argument('--port', type=int, default=1883)
    ap.add_argument('--model', default='modelos/xgb_flota_q0.25.joblib')
    ap.add_argument('--features', default='modelos/features.joblib')
    ap.add_argument('--config', default='modelos/config.joblib')
    ap.add_argument('--db', default='predicciones.db')
    args = ap.parse_args()

    modelo = joblib.load(args.model)
    FEATURES = joblib.load(args.features)
    cfg = joblib.load(args.config)
    cap = cfg['RUL_CAP']
    builder = StreamingFeatureBuilder(FEATURES, cfg['SENSORS'], cfg['WVARS'],
                                      windows=cfg['ROLL_WINDOWS'], slope_w=cfg['SLOPE_W'])
    con = init_db(args.db)
    print(f"Modelo: {args.model.split('/')[-1]} | features: {len(FEATURES)} | cap: {cap}")
    print("Esperando telemetría...\n")

    real_acc = {}   # engine -> [cycle, [rul_real, ...]]  (buffer paralelo solo para mostrar real)
    seen, ended = set(), set()

    def procesar(X, meta, real_cap):
        pred = modelo.predict(X)
        rul_pred = float(np.clip(np.median(pred), 0, cap))   # mediana por vuelo, acotada
        con.execute("INSERT INTO predicciones VALUES (?,?,?,?,?,?)",
                    (meta['engine_id'], meta['cycle'], rul_pred, real_cap, meta['n_obs'], time.time()))
        con.commit()
        rr = f"{real_cap:.1f}" if real_cap is not None else "n/d"
        print(f"  [{meta['engine_id']}] vuelo {meta['cycle']:>3}  ->  RUL pred = {rul_pred:5.1f}  | real(cap) = {rr}")

    def on_connect(cli, *a):
        cli.subscribe("telemetria/#")

    def on_message(cli, userdata, m):
        d = json.loads(m.payload.decode())
        eng = d['engine_id']; seen.add(eng)

        if d.get('end'):
            res = builder.flush(eng)
            rb = real_acc.get(eng)
            real_cap = (min(float(np.median(rb[1])), cap) if (rb and rb[1]) else None)
            if res is not None:
                procesar(res[0], res[1], real_cap)
            ended.add(eng)
            print(f"  [fin] {eng}")
            if seen and ended >= seen:
                print("\nStream completo. Predicciones en", args.db)
                cli.disconnect()
            return

        cycle = d['cycle']; rr = d.get('rul_real')
        # buffer paralelo de RUL real (para comparar predicho vs real al cerrar el vuelo)
        real_cap_closed = None
        rb = real_acc.get(eng)
        if rb is not None and rb[0] != cycle:
            real_cap_closed = min(float(np.median(rb[1])), cap) if rb[1] else None
            real_acc[eng] = [cycle, []]
        elif rb is None:
            real_acc[eng] = [cycle, []]
        if rr is not None:
            real_acc[eng][1].append(rr)

        res = builder.add_reading(eng, cycle, d)   # devuelve el vuelo ANTERIOR si se cerró
        if res is not None:
            procesar(res[0], res[1], real_cap_closed)

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    cli.on_connect = on_connect; cli.on_message = on_message
    cli.connect(args.broker, args.port, 60)
    try:
        cli.loop_forever()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        con.close()


if __name__ == '__main__':
    main()
