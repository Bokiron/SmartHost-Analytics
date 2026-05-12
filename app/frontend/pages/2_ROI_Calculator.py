# frontend/pages/2_ROI_Calculator.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from state          import init_state
from components.roi import render_roi


init_state()

st.title(" ROI Calculator")
st.caption("Simula cuánto rinde económicamente reformar el apartamento")

if st.session_state.resultado_base is None:
    st.warning(" Primero realiza un análisis en la página **Tasación**.")
    st.page_link("pages/1_Tasacion.py", label="→ Ir a Tasación", icon="🏠")
else:
    r = st.session_state.resultado_base
    with st.expander(" Resumen del análisis actual", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Precio base",   f"{r['precio_base']}€/noche")
        col2.metric("Precio visual", f"{r['precio_visual']}€/noche")
        col3.metric("Ingresos anuales (foto actual)", f"{r['ingresos_anuales_visual']:,.0f}€")
    render_roi(r)