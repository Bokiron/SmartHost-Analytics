# frontend/state.py
import streamlit as st


def init_state() -> None:
    """Crea las claves del session_state si aún no existen."""
    defaults = {
        "resultado_base":     None,
        "datos_payload":      None,
        "resultado_reforma":  None,
        "inversion_guardada": 2000,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value