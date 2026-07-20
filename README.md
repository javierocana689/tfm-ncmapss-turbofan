# Mantenimiento predictivo de motores turbofán — N-CMAPSS

Sistema PHM (Prognostics and Health Management) completo sobre el dataset [NASA N-CMAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/): **prognosis** de vida útil remanente (RUL) con XGBoost calibrado por cuantiles, **diagnóstico** del modo de fallo, y una **arquitectura de streaming** (MQTT → inferencia con estado → SQLite → panel) validada con equivalencia numérica exacta respecto al procesamiento batch.

> TFM — Máster en Análisis de Grandes Volúmenes de Datos. El hilo metodológico del proyecto: *medir con cuidado, elegir por criterio, y desconfiar de las métricas que mejoran por la razón equivocada.*

## Resultados clave

| Componente | Métrica | Valor |
|---|---|---|
| Prognosis (flota, 9 subconjuntos) | RMSE medio | **6,24 ciclos** |
| Calibración conservadora (q0,25) | Sobreestimación | 52% → **36%** |
| Diagnóstico (XGBoost, por familias) | Exactitud zona degradada | **56%** |
| Streaming vs batch | Diferencia (202 vuelos) | **0,000000 ciclos** |

## Orden de lectura de los notebooks

| # | Notebook | Qué demuestra |
|---|---|---|
| 01 | `EDA_NCMAPSS_DS02` | Estructura del dataset, sensores informativos, y el **cap físico** del RUL (=57, onset del fallo) |
| 02 | `Baseline_XGBoost_DS02` | Primer modelo honesto: detección de una **fuga de información** (`cycle_norm`), diagnóstico por hipótesis (2 refutadas, 1 confirmada: **sesgo de vidas cortas**) |
| 03 | `Flota_completa` | Escalado a 60 motores / 9 subconjuntos; el sesgo de la unit 11 baja al ver más vidas cortas |
| 04 | `Evaluacion_flota_todos_los_test` | El modelo único generaliza a todos los modos de fallo (RMSE 4,86–8,95, media 6,24) |
| 05 | `Familia_vs_modelo_unico` | El modelo único **gana siempre** a los especialistas (6,94 vs 8,27) → decisión de arquitectura |
| 06 | `Calibracion_cuantiles_coste` | El error no es simétrico: calibración por cuantiles y elección por **coste esperado**, no por RMSE |
| 07 | `Modelos_secuenciales_CNN_LSTM` | CNN/LSTM pierden contra el tabular (10,53/11,76 vs 7,28) y el sesgo de vida corta **persiste** → es un problema de datos |
| 08 | `Diagnostico_modo_fallo_CNN` | Primer clasificador del componente que falla; el Fan se distingue (F1 0,82), las turbinas se confunden entre sí |
| 09 | `Diagnostico_ablacion` | Las mejoras de la literatura **empeoran** con estos datos → tercera confirmación del límite de volumen |
| 10 | `Diagnostico_XGBoost_final` | El tabular también gana en diagnóstico (0,54 vs 0,41 por motor) → **componente final del PHM** |

## Arquitectura de streaming (`/streaming`)

```
producer.py → MQTT (mosquitto) → consumer.py (StreamingFeatureBuilder + modelo q0,25) → SQLite → dashboard.py (Streamlit)
```

- `validate_streaming.py` compara batch vs streaming vuelo a vuelo: **diferencia 0,000000** sobre 202 vuelos.
- El panel muestra la trayectoria de RUL en tiempo real con umbral de aviso configurable (15 ciclos ≈ 2× el error del modelo).

### Despliegue con Docker (`/docker`)

Versión **contenerizada** de la arquitectura de streaming: levanta las cinco capas (ingesta → MQTT → inferencia → SQLite → panel) con un solo comando.

```bash
cd docker
# copia los .h5 de N-CMAPSS en docker/data/  (los modelos .joblib ya van incluidos)
docker compose up --build          # panel en http://localhost:8501
```

`streaming/` es para ejecutar en local; `docker/` es el **artefacto de despliegue autocontenido** (broker Mosquitto + producer + consumer + panel, orquestados con Docker Compose). Los `.py` se repiten a propósito: es un paquete de despliegue independiente. Detalles y opciones (p. ej. servir el cuantil 0,10) en [`docker/README_docker.md`](docker/README_docker.md).

## Requisitos

```bash
pip install pandas numpy h5py scikit-learn xgboost matplotlib joblib
pip install tensorflow          # solo notebooks 07-09
pip install paho-mqtt streamlit # solo /streaming (requiere un broker mosquitto)
```

Los datos (`.h5` de N-CMAPSS) no se incluyen; descárgalos del repositorio de la NASA y ajusta la ruta en `config.py`.

## Decisiones de diseño (resumen)

- **Solo señales observables**: se excluyen los sensores virtuales (X_v), los parámetros de salud (T) y `hs` — no existirían en un motor real.
- **cap=57 por criterio físico** (RUL medio en el onset del fallo), no por optimización de métrica.
- **Modelo único de flota** frente a especialistas por familia: decisión empírica (notebook 05).
- **q0,25 desplegado**: único modelo que mejora a la vez la sobreestimación y el score NASA.
- **Dos modelos especializados** (prognosis + diagnóstico) en lugar de multitarea: el cuerpo secuencial pierde en prognosis (notebook 07), compartirlo la sacrificaría.
