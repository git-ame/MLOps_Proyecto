"""UI (punto extra): consume la API desplegada en Kubernetes. Uso:
   API_URL=http://localhost:8080 streamlit run ui/app.py"""
import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Riesgo de deserción · caso X", page_icon="🎓", layout="centered")
st.title("Riesgo de deserción — caso X")
st.caption("Datos sintéticos. La predicción la calcula la API desplegada en Kubernetes.")

try:
    info = requests.get(f"{API}/info", timeout=5).json()
    st.sidebar.success(f"API conectada\n\nModelo {info['modelo']} v{info['version']}\n\nRun {info['run_id'][:8]}…")
except Exception:
    st.sidebar.error(f"Sin conexión con {API}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    tipo_programa = st.selectbox("Tipo de programa", ["Diplomado", "Especialidad", "Maestría"], index=2)
    sede = st.selectbox("Sede", ["Sede A", "Sede B", "Sede C", "Sede D", "Sede E"])
    plan = st.radio("Plan de pago", ["Crédito", "Contado"], horizontal=True)
    promo = st.radio("Tiene promoción", ["No", "Sí"], horizontal=True)
    regul = st.radio("Regularizado", ["No", "Sí"], horizontal=True)
    meses = st.slider("Meses desde inscripción", 0, 24, 6)
with c2:
    pct = st.slider("Porcentaje pagado", 0.0, 1.0, 0.4, 0.01)
    venc = st.number_input("Cuotas vencidas", 0, 20, 2)
    dias = st.number_input("Días desde último pago", 0, 365, 45)
    reprob = st.number_input("Módulos reprobados", 0, 12, 0)
    desc = st.slider("Descuento (%)", 0, 100, 0)

colegiatura = {"Diplomado": 3000.0, "Especialidad": 5500.0, "Maestría": 9000.0}[tipo_programa]
modulos = {"Diplomado": 4, "Especialidad": 8, "Maestría": 12}[tipo_programa]
cuotas = 1 if plan == "Contado" else {"Diplomado": 6, "Especialidad": 10, "Maestría": 18}[tipo_programa]
matricula = round(colegiatura * 0.1, 2)
total = (colegiatura + matricula) * (1 - desc / 100)
body = dict(precio_matricula=matricula, precio_colegiatura=colegiatura, nro_cuotas=cuotas,
            monto_pagado_formativa=round(total * pct, 2), saldo_pendiente_formativa=round(total * (1 - pct), 2),
            porcentaje_pagado=pct, descuento_pct=desc, cuotas_vencidas=int(venc), dias_desde_ultimo_pago=float(dias),
            meses_desde_inscripcion=int(meses), modulos_programados=modulos, modulos_reprobados=int(reprob),
            tipo_programa=tipo_programa, sede=sede, tipo_plan_pago=plan, tiene_promocion=promo, es_regularizado=regul)

if st.button("Calcular riesgo", type="primary"):
    r = requests.post(f"{API}/predict", json=body, timeout=10)
    if r.ok:
        j = r.json()
        p = j["probabilidad_desercion"]
        st.metric("Probabilidad de deserción", f"{p:.1%}")
        st.progress(min(p, 1.0))
        (st.error if j["desercion"] else st.success)("Riesgo ALTO: priorizar contacto" if j["desercion"] else "Riesgo bajo")
        st.caption(f"Respondió el pod **{j['pod']}** · modelo v{j['version_modelo']}")
    else:
        st.error(r.text)
