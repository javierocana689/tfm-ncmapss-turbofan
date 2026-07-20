# -*- coding: utf-8 -*-
"""
dashboard.py — Panel de monitorización del RUL
==============================================
Lee la base de datos de predicciones que escribe el consumidor y muestra, por motor,
la trayectoria del RUL predicho frente al real a lo largo de los vuelos.

Diseño (rendimiento y robustez, sin tocar ningún número):
  - UNA sola consulta a la BD por refresco: trae un "snapshot" ya deduplicado por
    (motor, ciclo) quedándose con la predicción más reciente (MAX(ts)). Sobre una tabla
    con muchas ejecuciones acumuladas devuelve solo ~200 filas en pocos ms.
  - Cambiar de motor o mover el slider NO vuelve a tocar la BD: se filtra en pandas
    sobre ese snapshot en memoria. (Antes se colgaba justo aquí, al reabrir la BD
    mientras el consumer escribía.)
  - Conexión con espera acotada (busy_timeout) y query_only=ON: no puede escribir en
    la BD de predicciones y no se queda colgada indefinidamente por un lock.
  - Caché corta (st.cache_data) para no releer en cada interacción; "Refrescar" la limpia.

Uso:
    streamlit run dashboard.py
Requisitos: pip install streamlit pandas
"""
import os
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mantenimiento predictivo — RUL", layout="wide")
st.title("Monitorización de RUL en tiempo real")

DB = st.sidebar.text_input("Base de datos", os.environ.get("DB_PATH", "predicciones.db"))
st.sidebar.caption("Modelo conservador (cuantil). El RUL predicho es una estimación prudente.")
if st.sidebar.button("Refrescar"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=2, show_spinner=False)
def cargar_snapshot(path):
    """Una fila por (motor, ciclo) con la predicción más reciente. Devuelve un DataFrame pequeño."""
    con = sqlite3.connect(path, timeout=5)
    try:
        con.execute("PRAGMA busy_timeout=5000")   # espera acotada si hay un lock momentáneo
        con.execute("PRAGMA query_only=ON")        # el panel nunca escribe en la BD
        df = pd.read_sql_query(
            """SELECT engine_id, cycle, rul_pred, rul_real, n_obs, MAX(ts) AS ts
               FROM predicciones
               GROUP BY engine_id, cycle
               ORDER BY engine_id, cycle""", con)
    finally:
        con.close()
    return df


try:
    df = cargar_snapshot(DB)
except Exception as e:
    st.warning(f"No se pudo leer la base de datos todavía (reintenta con «Refrescar»): {e}")
    st.stop()

if df.empty:
    st.info("Aún no hay predicciones. Arranca el broker, el consumidor y el productor.")
    st.stop()

# Todo lo de abajo se calcula en pandas sobre el snapshot: cambiar de motor no toca la BD.
motores = sorted(df.engine_id.unique())
ultimo = df.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)

col1, col2, col3 = st.columns(3)
col1.metric("Motores monitorizados", len(motores))
col2.metric("Vuelos procesados", len(df))
col3.metric("RUL mínimo actual", f"{ultimo.rul_pred.min():.0f} ciclos")

eng = st.selectbox("Motor", motores)
d = df[df.engine_id == eng].sort_values("cycle")

st.subheader(f"Trayectoria de RUL — {eng}")
plot_df = d.set_index("cycle")[["rul_pred"]].rename(columns={"rul_pred": "RUL predicho"})
if d.rul_real.notna().any():
    plot_df["RUL real (cap)"] = d.set_index("cycle")["rul_real"]
st.line_chart(plot_df)

# Aviso simple de mantenimiento: cuando el RUL predicho cae por debajo de un umbral
umbral = st.slider("Umbral de aviso (ciclos)", 1, 40, 15)
criticos = ultimo[ultimo.rul_pred <= umbral]
if not criticos.empty:
    st.error("Motores que requieren intervención (RUL predicho bajo el umbral):")
    st.dataframe(criticos[["engine_id", "cycle", "rul_pred", "rul_real"]].reset_index(drop=True))
else:
    st.success("Ningún motor por debajo del umbral de aviso.")

st.caption("Tabla de predicciones recientes")
st.dataframe(d[["cycle", "rul_pred", "rul_real", "n_obs"]].reset_index(drop=True))