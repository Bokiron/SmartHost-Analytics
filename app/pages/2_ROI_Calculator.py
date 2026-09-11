# frontend/pages/2_ROI_Calculator.py
from PIL import Image
from app.utils.state import init_state
from core.loader import load_all_artifacts
from services.predictor import predecir_multimodal
from services.roi_calculos import calcular_dias_ocupados, calcular_ingresos_anuales
import streamlit as st
from components.roi import foto_reforma

init_state()

st.title(" ROI Calculator")
st.caption("Simula cuánto rinde económicamente reformar el apartamento")

if st.button(" Calcular ROI de la reforma", type="primary"):
            with st.spinner("Calculando impacto de la reforma..."):
                try:
                    artifacts = load_all_artifacts()
                    pil_reforma = Image.open(foto_reforma)
                    datos = st.session_state.datos_payload
                    
                    # Predicción de la nueva foto
                    nuevo_precio_visual = predecir_multimodal(datos, pil_reforma, artifacts)
                    dias_ocupados = calcular_dias_ocupados(datos["reviews_per_month"])
                    nuevos_ingresos = calcular_ingresos_anuales(nuevo_precio_visual, dias_ocupados)
                    
                    st.session_state.resultado_reforma = {
                        "precio_visual": nuevo_precio_visual,
                        "ingresos_anuales_visual": nuevos_ingresos
                    }
                except Exception as e:
                    st.error(f"Error al calcular la reforma: {e}")
                    st.stop()