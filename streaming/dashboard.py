# -*- coding: utf-8 -*-
"""
dashboard.py — Panel de monitorización del RUL
==============================================
Lee la base de datos de predicciones que escribe el consumidor y muestra, por motor,
la trayectoria del RUL predicho frente al real a lo largo de los vuelos. Se refresca
para ver llegar las predicciones en vivo.

Uso:
    streamlit run dashboard.py
Requisitos: pip install streamlit pandas
"""
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mantenimiento predictivo — RUL", layout="wide")
st.title("Monitorización de RUL en tiempo real")

DB = st.sidebar.text_input("Base de datos", "predicciones.db")
st.sidebar.caption("Modelo conservador (cuantil). El RUL predicho es una estimación prudente.")
if st.sidebar.button("Refrescar"):
    st.rerun()

try:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM predicciones ORDER BY cycle", con)
    con.close()
except Exception as e:
    st.warning(f"No se pudo leer la base de datos todavía: {e}")
    st.stop()

if df.empty:
    st.info("Aún no hay predicciones. Arranca el broker, el consumidor y el productor.")
    st.stop()

motores = sorted(df.engine_id.unique())
col1, col2, col3 = st.columns(3)
col1.metric("Motores monitorizados", len(motores))
col2.metric("Vuelos procesados", len(df))
ultimo = df.sort_values("ts").groupby("engine_id").tail(1)
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
