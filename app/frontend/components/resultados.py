# frontend/components/resultados.py
import streamlit as st


def render_resultados(r: dict) -> None:
    """
    Muestra las métricas de precio e ingresos a partir de la respuesta
    del backend.

    Parámetros
    ----------
    r : dict
        Respuesta JSON del endpoint /predict/multimodal con las claves:
        precio_base, precio_visual, impacto_visual_eur, impacto_visual_pct,
        dias_ocupados_anio, ingresos_anuales_base, ingresos_anuales_visual.
    """
    st.success("✅ Análisis completado")
    st.divider()

    # ── Precios por noche ─────────────────────────────────────────────────────
    st.subheader(" Precio por noche predicho")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Precio Base (mercado)",
        f"{r['precio_base']}€",
        help="Precio según características físicas del piso",
    )
    col2.metric(
        "Precio Visual (con foto actual)",
        f"{r['precio_visual']}€",
        delta=f"{r['impacto_visual_eur']}€ por la foto",
        help="Precio teniendo en cuenta el atractivo visual de la foto",
    )
    col3.metric(
        "Impacto de la foto",
        f"{r['impacto_visual_pct']}%",
        help="Cuánto suma o resta la foto respecto al precio de mercado",
    )

    st.divider()

    # ── Ingresos anuales ──────────────────────────────────────────────────────
    st.subheader(" Estimación de ingresos anuales")
    col4, col5, col6 = st.columns(3)
    col4.metric("Días ocupados/año", f"{r['dias_ocupados_anio']} días")
    col5.metric("Ingresos base/año", f"{r['ingresos_anuales_base']:,.0f}€")
    col6.metric(
        "Ingresos con foto actual/año",
        f"{r['ingresos_anuales_visual']:,.0f}€",
        delta=f"{r['ingresos_anuales_visual'] - r['ingresos_anuales_base']:,.0f}€ vs base",
    )