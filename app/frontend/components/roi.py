# frontend/components/roi.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import streamlit as st

from config import API_URL


def render_roi(r: dict) -> None:
    st.divider()
    st.subheader(" Simulador de ROI — ¿Cuánto vale reformar?")
    st.caption("Sube una foto de inspiración (post-reforma) para ver si la inversión se amortiza")

    foto_reforma = st.file_uploader(
        "Foto post-reforma o de inspiración",
        type=["jpg", "jpeg", "png"],
        key="foto_reforma",
    )

    if foto_reforma:
        st.image(foto_reforma, caption="✨ Foto post-reforma", use_container_width=True)

        inversion = st.number_input(
            " Inversión en reforma (€)",
            min_value=0, max_value=50000,
            value=st.session_state.inversion_guardada, step=500,
        )
        st.session_state.inversion_guardada = inversion

        if st.button(" Calcular ROI de la reforma", type="primary"):
            with st.spinner("Calculando impacto de la reforma..."):
                try:
                    resp2 = requests.post(
                        f"{API_URL}/predict/multimodal",
                        data={"request": json.dumps(st.session_state.datos_payload)},
                        files={"image": (foto_reforma.name, foto_reforma.getvalue(), foto_reforma.type)},
                        timeout=30,
                    )
                    resp2.raise_for_status()
                    st.session_state.resultado_reforma = resp2.json()
                except Exception as e:
                    st.error(f"Error al calcular la reforma: {e}")
                    st.stop()

    if st.session_state.resultado_reforma is not None:
        r2 = st.session_state.resultado_reforma

        ingresos_antes   = r["ingresos_anuales_visual"]
        ingresos_despues = r2["ingresos_anuales_visual"]
        beneficio_anual  = ingresos_despues - ingresos_antes
        beneficio_mes    = beneficio_anual / 12
        inversion_actual = st.session_state.inversion_guardada

        st.divider()
        st.subheader(" Resultado del ROI")

        comp1, comp2, comp3, comp4 = st.columns(4)
        comp1.metric("Precio foto actual",  f"{r['precio_visual']}€/noche")
        comp2.metric("Precio foto reforma", f"{r2['precio_visual']}€/noche",
                     delta=f"{r2['precio_visual'] - r['precio_visual']:+.2f}€")
        comp3.metric("Ingresos actuales",    f"{ingresos_antes:,.0f}€/año")
        comp4.metric("Ingresos con reforma", f"{ingresos_despues:,.0f}€/año",
                     delta=f"{beneficio_anual:+,.0f}€/año")

        st.divider()

        if inversion_actual > 0 and beneficio_anual > 0:
            payback = inversion_actual / beneficio_mes
            roi_pct = (beneficio_anual / inversion_actual) * 100
            st.success(f" **ROI: {roi_pct:.1f}% anual** — Recuperas la inversión en **{payback:.1f} meses**")
        elif beneficio_anual <= 0:
            st.warning(" La foto de reforma no mejora los ingresos respecto a la actual.")
        else:
            st.info("Introduce el coste de la reforma para calcular el payback.")