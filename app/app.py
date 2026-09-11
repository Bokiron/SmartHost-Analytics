# app.py (Raíz del proyecto)
import sys
import os
# Aseguramos que la raíz del proyecto está en el PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils.state import init_state  # state.py ahora está en la carpeta utils

# Definimos la configuración directamente aquí ya que borramos config.py
PAGE_CONFIG = {
    "page_title": "SmartHost Analytics",
    "page_icon":  "🏠",
    "layout":     "wide",
}

st.set_page_config(**PAGE_CONFIG)
init_state()

st.title("🏠 SmartHost Analytics")
st.caption("Predictor de precios y ROI para apartamentos turísticos en Málaga")

st.markdown("""
## Bienvenido

**SmartHost Analytics** utiliza redes neuronales multimodales para ayudarte
a maximizar los ingresos de tu apartamento turístico en Málaga.

### Navega por las secciones en el menú lateral 👈

| Página | Descripción |
|---|---|
| 📈 **Tasación** | Introduce los datos del piso y obtén el precio base y visual |
| 💰 **ROI Calculator** | Simula cuánto rinde económicamente una reforma |
| 🧠 **Sobre el Modelo** | Arquitectura, métricas y decisiones técnicas |
""")