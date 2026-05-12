# frontend/pages/1_Tasacion.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import streamlit as st

from config                    import API_URL
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

    with st.spinner("Analizando el apartamento..."):
        try:
            response = requests.post(
                f"{API_URL}/predict/multimodal",
                data={"request": json.dumps(datos)},
                files={"image": (foto.name, foto.getvalue(), foto.type)},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.resultado_base    = response.json()
            st.session_state.datos_payload     = datos
            st.session_state.resultado_reforma = None
        except requests.exceptions.ConnectionError:
            st.error(" No se puede conectar con la API. ¿Está corriendo el servidor?")
            st.code("uvicorn app.backend.main:app --reload --port 8000")
            st.stop()
        except Exception as e:
            st.error(f" Error inesperado: {e}")
            st.stop()

if st.session_state.resultado_base is not None:
    render_resultados(st.session_state.resultado_base)
    render_roi(st.session_state.resultado_base)