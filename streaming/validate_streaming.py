# -*- coding: utf-8 -*-
"""
validate_streaming.py — Validación batch vs streaming
=====================================================
Demuestra numéricamente que la inferencia en streaming produce las MISMAS predicciones
que el pipeline batch. Para los motores que hay en la base de datos de predicciones:

  1. Reconstruye las predicciones en BATCH (carga el test, calcula features, predice por
     lectura y agrega por ciclo con la mediana — sin submuestrear, igual que el streaming).
  2. Lee las predicciones que el consumidor guardó en SQLite (STREAMING).
  3. Compara vuelo a vuelo (diferencia máxima/media; debe ser ~0).
  4. Calcula RMSE y sesgo de ambos frente al RUL real, en la zona degradada (real < cap),
     que es donde la métrica tiene sentido.

Si las diferencias son ~0 y las métricas coinciden, queda probado que servir el modelo en
tiempo real no degrada su precisión.

Uso:
    python validate_streaming.py
Requisitos: pip install h5py numpy pandas joblib scikit-learn xgboost
"""
import argparse, os, sqlite3
import numpy as np, pandas as pd, h5py, joblib
from sklearn.metrics import mean_squared_error

DATA_DIR = r"C:\Users\11jav\Documents\Master\TFM\Dataset_Nasa\Turbofan_Engine_degradation\Datasets\data_set\data_set"
FILES = {
    'DS01': 'N-CMAPSS_DS01-005.h5', 'DS02': 'N-CMAPSS_DS02-006.h5',
    'DS03': 'N-CMAPSS_DS03-012.h5', 'DS04': 'N-CMAPSS_DS04.h5',
    'DS05': 'N-CMAPSS_DS05.h5',     'DS06': 'N-CMAPSS_DS06.h5',
    'DS07': 'N-CMAPSS_DS07.h5',     'DS08a': 'N-CMAPSS_DS08a-009.h5',
    'DS08c': 'N-CMAPSS_DS08c-008.h5',
}

# ---- pipeline batch (idéntico al de los notebooks) ----
def load_subset(path, subset, ds_code):
    with h5py.File(path, 'r') as f:
        W   = np.array(f['W_test'],   dtype=np.float32)
        X_s = np.array(f['X_s_test'], dtype=np.float32)
        A   = np.array(f['A_test'],   dtype=np.float32)
        Y   = np.array(f['Y_test'],   dtype=np.float32).reshape(-1)
        W_var  = [v.decode() for v in f['W_var'][:]]
        Xs_var = [v.decode() for v in f['X_s_var'][:]]
        A_var  = [v.decode() for v in f['A_var'][:]]
    if 'hs' in A_var:
        i = A_var.index('hs'); A = np.delete(A, i, axis=1); A_var = [v for v in A_var if v != 'hs']
    df = pd.DataFrame(np.hstack([A, W, X_s, Y.reshape(-1, 1)]), columns=A_var + W_var + Xs_var + ['RUL'])
    for c in ['unit', 'cycle', 'Fc']: df[c] = df[c].astype(int)
    df['ds_code'] = ds_code
    df['engine_id'] = subset + '_u' + df['unit'].astype(str)
    return df, W_var, Xs_var

def cycle_level_features(df, sensors, windows, slope_w):
    df = df.sort_values(['engine_id', 'cycle'])
    cyc = df.groupby(['engine_id', 'cycle'])[sensors].mean().reset_index().sort_values(['engine_id', 'cycle'])
    rc = []
    for w in windows:
        for col in sensors:
            cyc[f'{col}_roll{w}_mean'] = cyc.groupby('engine_id')[col].transform(lambda x: x.rolling(w, min_periods=1).mean())
            cyc[f'{col}_roll{w}_std']  = cyc.groupby('engine_id')[col].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            rc += [f'{col}_roll{w}_mean', f'{col}_roll{w}_std']
    def slope(x, w):
        a = x.values; out = np.zeros(len(a))
        for i in range(len(a)):
            seg = a[max(0, i - w + 1): i + 1]
            if len(seg) >= 2: out[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        return pd.Series(out, index=x.index)
    for col in sensors:
        cyc[f'{col}_slope{slope_w}'] = cyc.groupby('engine_id')[col].transform(lambda x: slope(x, slope_w)); rc.append(f'{col}_slope{slope_w}')
    stats = df.groupby(['engine_id', 'cycle'])[sensors].agg(['max', 'std'])
    stats.columns = [f'{s}_cyc_{a}' for s, a in stats.columns]; stats = stats.fillna(0.0).reset_index()
    return cyc[['engine_id', 'cycle'] + rc].merge(stats, on=['engine_id', 'cycle'])

def assemble(df, cyc_feat, sensors, w_var):
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    pos = df.groupby(['engine_id', 'cycle']).cumcount()
    size = df.groupby(['engine_id', 'cycle'])['cycle'].transform('size') - 1
    df['pos_relativa'] = (pos / size.replace(0, 1)).astype(np.float32)
    df['age'] = df['cycle'].astype(np.float32)
    keep = ['engine_id', 'unit', 'cycle', 'Fc', 'ds_code', 'RUL', 'pos_relativa', 'age'] + sensors + w_var
    return df[keep].merge(cyc_feat, on=['engine_id', 'cycle'], how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='predicciones.db')
    ap.add_argument('--model', default='modelos/xgb_flota_q0.25.joblib')
    ap.add_argument('--features', default='modelos/features.joblib')
    ap.add_argument('--config', default='modelos/config.joblib')
    ap.add_argument('--data-dir', default=DATA_DIR)
    args = ap.parse_args()

    modelo = joblib.load(args.model)
    FEATURES = joblib.load(args.features)
    cfg = joblib.load(args.config)
    cap = cfg['RUL_CAP']

    # Predicciones del streaming
    con = sqlite3.connect(args.db)
    stream = pd.read_sql_query("SELECT engine_id, cycle, rul_pred AS pred_stream, rul_real FROM predicciones", con)
    con.close()
    if stream.empty:
        raise SystemExit("La base de datos no tiene predicciones.")
    subsets = sorted({e.split('_u')[0] for e in stream.engine_id.unique()})
    print(f"Motores en la BD: {sorted(stream.engine_id.unique())}")
    print(f"Subconjuntos: {subsets}\n")

    # Predicciones en batch para los mismos motores
    batch_rows = []
    for subset in subsets:
        path = os.path.join(args.data_dir, FILES[subset])
        df, Wv, Xv = load_subset(path, subset, cfg['DS_CODE'][subset])
        df = df[df.engine_id.isin(stream.engine_id.unique())]
        cyc = cycle_level_features(df, cfg['SENSORS'], cfg['ROLL_WINDOWS'], cfg['SLOPE_W'])
        mat = assemble(df, cyc, cfg['SENSORS'], cfg['WVARS'])
        p = np.clip(modelo.predict(mat[FEATURES].values), 0, cap)
        t = mat[['engine_id', 'cycle', 'RUL']].copy(); t['p'] = p
        g = t.groupby(['engine_id', 'cycle']).agg(pred_batch=('p', 'median'),
                                                  real=('RUL', 'first')).reset_index()
        batch_rows.append(g)
    batch = pd.concat(batch_rows, ignore_index=True)
    batch['real_cap'] = np.minimum(batch.real, cap)

    # Comparación vuelo a vuelo
    m = stream.merge(batch, on=['engine_id', 'cycle'], how='inner')
    m['dif'] = (m.pred_stream - m.pred_batch).abs()
    print("===== Equivalencia batch vs streaming (por vuelo) =====")
    print(f"  Vuelos comparados : {len(m)}")
    print(f"  Diferencia media  : {m.dif.mean():.6f} ciclos")
    print(f"  Diferencia máxima : {m.dif.max():.6f} ciclos")
    print("  (cercano a 0 = el streaming predice lo mismo que el batch)\n")

    # Calidad en la zona degradada (real < cap), que es donde la métrica importa
    deg = m[m.real_cap < cap]
    if len(deg):
        rmse_s = mean_squared_error(deg.real_cap, deg.pred_stream) ** .5
        rmse_b = mean_squared_error(deg.real_cap, deg.pred_batch) ** .5
        print("===== Calidad en zona degradada (RUL real < cap) =====")
        print(f"  Vuelos degradados : {len(deg)}")
        print(f"  RMSE  streaming={rmse_s:.3f} | batch={rmse_b:.3f}")
        print(f"  Sesgo streaming={(deg.pred_stream-deg.real_cap).mean():+.3f} | "
              f"batch={(deg.pred_batch-deg.real_cap).mean():+.3f}")
        print(f"  % sobreestima streaming={100*((deg.pred_stream-deg.real_cap)>0).mean():.1f}%\n")

    print("===== Por motor (RMSE y sesgo en streaming, zona degradada) =====")
    for eng in sorted(deg.engine_id.unique()):
        s = deg[deg.engine_id == eng]
        print(f"  {eng}: vida={int(m[m.engine_id==eng].cycle.max())}  "
              f"RMSE={mean_squared_error(s.real_cap, s.pred_stream)**.5:5.2f}  "
              f"sesgo={(s.pred_stream-s.real_cap).mean():+5.2f}")


if __name__ == '__main__':
    main()
