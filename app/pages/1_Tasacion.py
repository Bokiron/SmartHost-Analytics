# frontend/pages/1_Tasacion.py
from turtle import st

from PIL import Image
from app.utils.state import init_state
from core.loader import load_all_artifacts
from services.predictor import predecir_tabular, predecir_multimodal
from services.roi_calculos import calcular_dias_ocupados, calcular_ingresos_anuales
import streamlit as st
from state                     import init_state
from components.sidebar        import render_sidebar
from components.resultados     import render_resultados
from components.roi            import render_roi
from utils.calculos            import construir_payload

init_state()

st.title(" Tasación del apartamento")
st.caption("Predictor de precios para apartamentos turísticos en Málaga")

form, foto, calcular = render_sidebar()

if not calcular and st.session_state.resultado_base is None:
    st.info(" Rellena el formulario lateral y pulsa **Calcular precio y ROI**")
    st.markdown("""
    ### Cómo funciona
    1. **Rellena** los datos de tu apartamento en el panel izquierdo
    2. **Sube** la foto de portada actual del piso
    3. Obtén el **Precio Base** (solo características) y el **Precio Visual** (foto incluida)
    4. Descubre cuánto dinero te está costando una mala foto al año
    5. Ve a **ROI Calculator** para simular cuánto rinde una reforma
    """)

if calcular:
    if foto is None:
        st.error(" Debes subir una foto de portada para obtener el precio visual.")
        st.stop()

    datos = construir_payload(form)
    
    # 1. Cargar modelos en memoria (solo tarda la primera vez por la caché)
    artifacts = load_all_artifacts()

    with st.spinner("Analizando el apartamento con IA..."):
        try:
            # 2. Inferencia directa (sin requests)
            pil_img = Image.open(foto)
            precio_base = predecir_tabular(datos, artifacts)
            precio_visual = predecir_multimodal(datos, pil_img, artifacts)
            
            # 3. Cálculos de negocio (antiguo endpoint multimodal)
            dias_ocupados   = calcular_dias_ocupados(datos["reviews_per_month"])
            ingresos_base   = calcular_ingresos_anuales(precio_base, dias_ocupados)
            ingresos_visual = calcular_ingresos_anuales(precio_visual, dias_ocupados)
            impacto_eur     = round(precio_visual - precio_base, 2)
            impacto_pct     = round((impacto_eur / precio_base) * 100, 1) if precio_base else 0.0

            # 4. Construimos el diccionario que espera tu componente render_resultados()
            resultado_simulado = {
                "precio_base": precio_base,
                "precio_visual": precio_visual,
                "dias_ocupados_anio": dias_ocupados,
                "ingresos_anuales_base": ingresos_base,
                "ingresos_anuales_visual": ingresos_visual,
                "impacto_visual_eur": impacto_eur,
                "impacto_visual_pct": impacto_pct,
            }

            st.session_state.resultado_base    = resultado_simulado
            st.session_state.datos_payload     = datos
            st.session_state.resultado_reforma = None
            
        except Exception as e:
            st.error(f" Error inesperado durante la inferencia: {e}")
            st.stop()

if st.session_state.resultado_base is not None:
    render_resultados(st.session_state.resultado_base)
    render_roi(st.session_state.resultado_base)