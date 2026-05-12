# frontend/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import PAGE_CONFIG
from state  import init_state


st.set_page_config(**PAGE_CONFIG)
init_state()

st.title("🏠 SmartHost Analytics")
st.caption("Predictor de precios y ROI para apartamentos turísticos en Málaga")

st.markdown("""
## Bienvenido

**SmartHost Analytics** utiliza redes neuronales multimodales para ayudarte
a maximizar los ingresos de tu apartamento turístico en Málaga.

### Navega por las secciones 👈

| Página | Descripción |
|---|---|
|  **Tasación** | Introduce los datos del piso y obtén el precio base y visual |
|  **ROI Calculator** | Simula cuánto rinde económicamente una reforma |
|  **Sobre el Modelo** | Arquitectura, métricas y decisiones técnicas |
""")