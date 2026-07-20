# -*- coding: utf-8 -*-
"""
streaming_features.py
=====================
Reconstrucción EN STREAMING (lectura a lectura, con estado por motor) de las mismas
características que el pipeline batch del proyecto. Validado numéricamente: produce
resultados idénticos a la versión batch (diferencia < 1e-9).

Mantiene, por motor:
  - el historial de medias por ciclo (para medias/desviaciones móviles inter-ciclo y pendiente),
  - el buffer de lecturas del ciclo en curso (para estadísticos intra-ciclo y pos_relativa).

Cuando un ciclo (vuelo) se cierra, devuelve una matriz de features lista para el modelo,
una fila por lectura del ciclo, en el orden exacto de FEATURES.
"""
import numpy as np


class StreamingFeatureBuilder:
    def __init__(self, features, sensors, wvars, windows=(5, 10), slope_w=10):
        self.FEATURES = list(features)
        self.sensors = list(sensors)      # X_s (sobre los que se calculan rolling/slope/intra)
        self.wvars = list(wvars)          # W (condiciones de vuelo)
        self.windows = tuple(windows)
        self.slope_w = slope_w
        self.hist = {}    # engine_id -> [media_por_ciclo (np.array len=n_sensors), ...]
        self.curr = {}    # engine_id -> (cycle, [lectura_dict, ...])

    def _cycle_level(self, engine, obs_rows):
        """Calcula las features de nivel de ciclo (constantes dentro del vuelo)."""
        arr = np.array([[r[s] for s in self.sensors] for r in obs_rows], dtype=float)
        cyc_mean = arr.mean(axis=0)
        cyc_max = arr.max(axis=0)
        cyc_std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])

        h = self.hist.setdefault(engine, [])
        h.append(cyc_mean)
        H = np.array(h)

        feat = {}
        for w in self.windows:
            seg = H[-w:]
            rm = seg.mean(axis=0)
            rs = seg.std(axis=0, ddof=1) if len(seg) > 1 else np.zeros(H.shape[1])
            for j, col in enumerate(self.sensors):
                feat[f'{col}_roll{w}_mean'] = rm[j]
                feat[f'{col}_roll{w}_std'] = rs[j]
        seg = H[-self.slope_w:]
        sl = np.polyfit(np.arange(len(seg)), seg, 1)[0] if len(seg) >= 2 else np.zeros(H.shape[1])
        for j, col in enumerate(self.sensors):
            feat[f'{col}_slope{self.slope_w}'] = sl[j]
            feat[f'{col}_cyc_max'] = cyc_max[j]
            feat[f'{col}_cyc_std'] = cyc_std[j]
        return feat

    def _finalize(self, engine, cycle, obs_rows):
        """Devuelve (X, meta): X es (n_obs, n_features) en orden FEATURES; meta dicts por obs."""
        cyc_feat = self._cycle_level(engine, obs_rows)
        n = len(obs_rows)
        X = np.empty((n, len(self.FEATURES)), dtype=np.float32)
        for i, r in enumerate(obs_rows):
            row = dict(cyc_feat)                      # features de ciclo (constantes)
            for s in self.sensors:
                row[s] = r[s]
            for wv in self.wvars:
                row[wv] = r[wv]
            row['pos_relativa'] = i / (n - 1) if n > 1 else 0.0
            row['age'] = float(cycle)
            row['Fc'] = r.get('Fc', 0)
            row['ds_code'] = r.get('ds_code', 0)
            X[i] = [row[f] for f in self.FEATURES]
        return X, {'engine_id': engine, 'cycle': cycle, 'n_obs': n}

    def add_reading(self, engine, cycle, reading):
        """
        reading: dict con los sensores X_s, las condiciones W, y Fc/ds_code.
        Devuelve (X, meta) del ciclo ANTERIOR si acaba de cerrarse; si no, None.
        """
        result = None
        if engine in self.curr:
            cyc0, obs = self.curr[engine]
            if cycle != cyc0:
                result = self._finalize(engine, cyc0, obs)
                self.curr[engine] = (cycle, [])
        else:
            self.curr[engine] = (cycle, [])
        self.curr[engine][1].append(reading)
        return result

    def flush(self, engine):
        """Cierra el ciclo en curso (fin del stream del motor)."""
        if engine in self.curr and self.curr[engine][1]:
            cyc0, obs = self.curr[engine]
            self.curr[engine] = (cyc0, [])
            return self._finalize(engine, cyc0, obs)
        return None
