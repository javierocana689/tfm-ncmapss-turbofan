# Despliegue con Docker — Sistema PHM en streaming

Contenerización completa de la arquitectura de streaming de mantenimiento predictivo.
Levanta las cinco capas (ingesta → MQTT → inferencia → SQLite → panel) con un solo comando.

## Estructura

```
docker_streaming/
├── docker-compose.yml        # orquesta los 4 servicios
├── Dockerfile.python         # imagen de producer y consumer
├── Dockerfile.dashboard      # imagen del panel Streamlit
├── requirements.txt          # dependencias de producer/consumer
├── mosquitto/config/         # configuración del broker MQTT
├── producer.py, consumer.py, streaming_features.py, dashboard.py
├── modelos/   ← DEBES crear esta carpeta con los .joblib de la Fase 3
└── data/      ← DEBES crear esta carpeta con los .h5 de N-CMAPSS
```

## Preparación (una sola vez)

1. Instala Docker Desktop (incluye `docker compose`).
2. En esta carpeta, crea dos subcarpetas y copia dentro:
   - `modelos/` → `xgb_flota_q0.25.joblib`, `features.joblib`, `config.joblib` (de la Fase 3).
   - `data/` → los ficheros `.h5` de N-CMAPSS (al menos `N-CMAPSS_DS02-006.h5`).

## Arranque

```bash
docker compose up --build
```

Esto construye las imágenes y levanta los cuatro servicios en orden: primero el broker,
luego el consumer (se suscribe), y tras 8 segundos el producer empieza a emitir. El panel
queda accesible en **http://localhost:8501**.

Para parar todo: `Ctrl+C`, y luego `docker compose down`.

## Qué hace cada servicio

| Servicio | Imagen | Función |
|---|---|---|
| `mosquitto` | eclipse-mosquitto:2 | Broker MQTT (capa de transporte) |
| `consumer` | Dockerfile.python | Inferencia con estado por motor; escribe `salida/predicciones.db` |
| `producer` | Dockerfile.python | Reproduce la telemetría de DS02 por MQTT |
| `dashboard` | Dockerfile.dashboard | Panel Streamlit en tiempo real |

## Detalle de diseño

- **El consumer se conecta al broker por nombre de servicio** (`--broker mosquitto`), no por
  `localhost`: dentro de la red de Docker, cada servicio es alcanzable por su nombre. Este es
  el único cambio respecto a la ejecución local, y no requiere tocar el código (los scripts ya
  aceptan `--broker` como argumento).
- **Datos y modelos van como volúmenes** (`./data`, `./modelos`), no dentro de la imagen: así
  las imágenes son ligeras y los datos no se duplican. Se montan en modo solo lectura (`:ro`).
- **La base de datos SQLite se comparte** entre consumer (escribe) y dashboard (lee) mediante
  el volumen `./salida`, montado en ambos contenedores.
- El `producer` espera 8 segundos antes de emitir, para que el consumer esté suscrito y no se
  pierdan los primeros vuelos.

## Cambiar el modelo servido

Para servir el modelo de máxima seguridad (cuantil 0,10) en vez del q0,25, edita el `command`
del servicio `consumer` en `docker-compose.yml`:

```yaml
    command: >
      python consumer.py --broker mosquitto
      --model modelos/xgb_flota_q0.10.joblib
      ...
```
