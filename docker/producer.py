# -*- coding: utf-8 -*-
"""
producer.py — Productor de telemetría (replay)
==============================================
Lee un subconjunto de TEST de N-CMAPSS y reproduce sus lecturas como un flujo en
tiempo real, publicándolas en un broker MQTT. Simula varios motores emitiendo a la vez
(round-robin) a 1 Hz, acelerado por un factor SPEED para que la demo no dure horas.

Uso:
    python producer.py --h5 N-CMAPSS/N-CMAPSS_DS02-006.h5 --subset DS02 --max-engines 3 --speed 200

Requisitos: pip install paho-mqtt h5py numpy joblib
Broker: un MQTT corriendo en localhost:1883 (p.ej. mosquitto).
"""
import argparse, json, time, itertools, os
import numpy as np, h5py, joblib
import paho.mqtt.client as mqtt

# Carpeta donde están los .h5 (los datos viven lejos del código).
# Cámbiala aquí o pásala con --data-dir.
DATA_DIR = r"C:\Users\11jav\Documents\Master\TFM\Dataset_Nasa\Turbofan_Engine_degradation\Datasets\data_set\data_set"

# Nombre de fichero por subconjunto.
FILES = {
    'DS01': 'N-CMAPSS_DS01-005.h5', 'DS02': 'N-CMAPSS_DS02-006.h5',
    'DS03': 'N-CMAPSS_DS03-012.h5', 'DS04': 'N-CMAPSS_DS04.h5',
    'DS05': 'N-CMAPSS_DS05.h5',     'DS06': 'N-CMAPSS_DS06.h5',
    'DS07': 'N-CMAPSS_DS07.h5',     'DS08a': 'N-CMAPSS_DS08a-009.h5',
    'DS08c': 'N-CMAPSS_DS08c-008.h5',
}

def cargar_test(path, subset, ds_code_map, max_engines=None):
    with h5py.File(path, 'r') as f:
        W   = np.array(f['W_test'],   dtype=np.float32)
        X_s = np.array(f['X_s_test'], dtype=np.float32)
        A   = np.array(f['A_test'],   dtype=np.float32)
        Y   = np.array(f['Y_test'],   dtype=np.float32).reshape(-1)
        W_var  = [v.decode() for v in f['W_var'][:]]
        Xs_var = [v.decode() for v in f['X_s_var'][:]]
        A_var  = [v.decode() for v in f['A_var'][:]]
    iu, ic, ifc = A_var.index('unit'), A_var.index('cycle'), A_var.index('Fc')
    ds_code = ds_code_map[subset]

    # Filtrar ANTES del bucle costoso: nos quedamos solo con los motores que
    # de verdad se van a reproducir (max_engines), en vez de procesar el test
    # completo (que puede ser millones de filas) para usar solo una fracción.
    unidades_col = A[:, iu]
    if max_engines:
        vistas, orden = set(), []
        for u in unidades_col:
            if u not in vistas:
                vistas.add(u); orden.append(u)
            if len(orden) >= max_engines:
                break
        seleccion = set(orden)
        mask = np.isin(unidades_col, list(seleccion))
        W, X_s, A, Y = W[mask], X_s[mask], A[mask], Y[mask]
        print(f"  Filtrado a {max_engines} motor(es) antes de procesar: "
              f"{len(unidades_col):,} -> {mask.sum():,} filas")

    # Convertir a listas de Python UNA VEZ: iterar sobre arrays de NumPy elemento a
    # elemento dentro de un bucle es mucho más lento que iterar listas ya extraídas.
    units_l  = A[:, iu].astype(int).tolist()
    cycles_l = A[:, ic].astype(int).tolist()
    fc_l     = A[:, ifc].astype(int).tolist()
    Y_l      = Y.tolist()
    Xs_l     = X_s.tolist()   # lista de listas: una fila = una lista de floats
    W_l      = W.tolist()

    lecturas_por_motor = {}
    for k in range(len(A)):
        eng = f"{subset}_u{units_l[k]}"
        msg = {'engine_id': eng, 'cycle': cycles_l[k], 'Fc': fc_l[k],
               'ds_code': ds_code, 'rul_real': Y_l[k]}
        row_s, row_w = Xs_l[k], W_l[k]
        for j, s in enumerate(Xs_var): msg[s] = row_s[j]
        for j, w in enumerate(W_var):  msg[w] = row_w[j]
        lecturas_por_motor.setdefault(eng, []).append(msg)
    # asegurar orden temporal por ciclo dentro de cada motor
    for eng in lecturas_por_motor:
        lecturas_por_motor[eng].sort(key=lambda m: m['cycle'])
    return lecturas_por_motor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True, help='Nombre del subconjunto, p.ej. DS02')
    ap.add_argument('--h5', default=None, help='Ruta directa al .h5 (opcional; si no, se construye con --data-dir y --subset)')
    ap.add_argument('--data-dir', default=DATA_DIR, help='Carpeta donde están los .h5')
    ap.add_argument('--broker', default='localhost'); ap.add_argument('--port', type=int, default=1883)
    ap.add_argument('--speed', type=float, default=200.0, help='Factor de aceleración sobre 1 Hz')
    ap.add_argument('--max-engines', type=int, default=3)
    ap.add_argument('--config', default='modelos/config.joblib')
    args = ap.parse_args()

    # Resolver la ruta del dataset
    if args.h5:
        h5_path = args.h5
    elif args.subset in FILES:
        h5_path = os.path.join(args.data_dir, FILES[args.subset])
    else:
        raise SystemExit(f"Subconjunto desconocido: {args.subset}. Opciones: {list(FILES)}. "
                         f"O pasa la ruta directa con --h5.")
    if not os.path.exists(h5_path):
        raise SystemExit(f"No encuentro el dataset en:\n  {h5_path}\n"
                         f"Ajusta DATA_DIR en el script, o usa --data-dir / --h5.")
    print(f"Dataset: {h5_path}")

    cfg = joblib.load(args.config)
    data = cargar_test(h5_path, args.subset, cfg['DS_CODE'], max_engines=args.max_engines)
    motores = list(data)[:args.max_engines]
    print(f"Motores a reproducir: {motores}")

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    cli.connect(args.broker, args.port, 60); cli.loop_start()

    # round-robin: un mensaje de cada motor por vuelta -> simula emisión concurrente a 1 Hz
    iters = {e: iter(data[e]) for e in motores}
    dt = 1.0 / args.speed
    activos = set(motores)
    enviados = 0
    while activos:
        for e in list(activos):
            try:
                msg = next(iters[e])
                cli.publish(f"telemetria/{e}", json.dumps(msg))
                enviados += 1
                time.sleep(dt)
            except StopIteration:
                cli.publish(f"telemetria/{e}", json.dumps({'engine_id': e, 'end': True}))
                activos.discard(e)
                print(f"  [fin] {e}")
    print(f"Reproducidas {enviados} lecturas. Stream terminado.")
    cli.loop_stop(); cli.disconnect()

if __name__ == '__main__':
    main()