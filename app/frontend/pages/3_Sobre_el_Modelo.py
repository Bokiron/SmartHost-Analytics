# frontend/pages/3_Sobre_el_Modelo.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from state import init_state

init_state()

st.title(" Sobre el Modelo")
st.caption("Arquitectura, métricas de evaluación y decisiones técnicas")


# ── Arquitectura dual + métricas ──────────────────────────────────────────────
st.subheader(" Arquitectura Dual")
tab1, tab2, tab3 = st.tabs(["Modo 1 · Tabular (MLP)", "Modo 2 · Multimodal", "Fusión"])

with tab1:
    st.subheader(" Métricas en el conjunto de Test")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error Absoluto Medio)", "~26 €/noche",
                help="En media, el modelo se equivoca ±26€ por noche")
    col2.metric("R² (Coef. de determinación)", "~0.59",
                help="El modelo explica el 59% de la variación de precios")
    col3.metric("Dataset de entrenamiento", "5.739 anuncios",
                help="Anuncios reales de Airbnb en Málaga tras limpieza")

    st.divider()
    st.markdown("""
    **Tasador Base** — Solo datos tabulares

    | Parámetro | Valor |
    |---|---|
    | Arquitectura | Red Neuronal Densa (MLP) |
    | Capas ocultas | 3 capas: 128 → 64 → 32 neuronas |
    | Activación | ReLU |
    | Regularización | Dropout p=0.2 |
    | Función de pérdida | MSELoss (entrenamiento) / MAE (evaluación) |
    | Optimizador | Adam · lr=0.001 |
    | Early Stopping | Paciencia = 20 épocas |
    | Features de entrada | 35 columnas (tras One-Hot Encoding) |
    """)

with tab2:
    st.subheader(" Métricas en el conjunto de Test")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error Absoluto Medio)", "21.47 €/noche",
                delta="-4.53€ vs tabular", delta_color="inverse",
                help="En media, el modelo se equivoca ±21€ por noche")
    col2.metric("R² (Coef. de determinación)", "0.692",
                delta="+0.102 vs tabular",
                help="El modelo explica el 69.2% de la variación de precios")
    col3.metric("Dataset de entrenamiento", "5.739 anuncios",
                help="Anuncios reales de Airbnb en Málaga tras limpieza")

    st.divider()
    st.markdown("""
    **Tasador Estético** — Imagen + datos tabulares

    | Parámetro | Valor |
    |---|---|
    | Backbone visual | ResNet34 preentrenado (Transfer Learning) |
    | Preprocesado imagen | Resize 256px → CenterCrop 224×224 → Normalize ImageNet |
    | Vector visual | 64 dimensiones |
    | Vector tabular | 32 dimensiones |
    | Vector fusionado | 96 dimensiones (concatenación) |
    | Predicción final | 1 neurona con activación lineal |
    """)

with tab3:
    st.markdown("""
    **Fusión Multimodal** en PyTorch

    ```python
    class MultimodalModel(nn.Module):
        def forward(self, imagen, tabular):
            vec_visual  = self.rama_cnn(imagen)     # → 64 dims
            vec_tabular = self.rama_mlp(tabular)    # → 32 dims
            fusionado   = torch.cat([vec_visual, vec_tabular], dim=1)  # → 96 dims
            precio      = self.cabeza(fusionado)    # → 1 valor
            return precio
    ```
    La red aprende **de forma end-to-end** qué combinación de estética visual
    y características físicas maximiza el precio predicho.
    """)


st.divider()


# ── Features más importantes ──────────────────────────────────────────────────
st.subheader(" Features con mayor poder predictivo")
st.markdown("""
Según el análisis de correlaciones y el EDA realizado sobre los 5.739 anuncios:

| Rank | Feature | Correlación con precio |
|---|---|---|
| 1 | `accommodates` (capacidad) | 0.607 |
| 2 | `beds` (camas) | 0.563 |
| 3 | `bedrooms` (habitaciones) | 0.521 |
| 4 | `bathrooms` (baños) | 0.458 |
| 5 | `amenities_score` | 0.316 |
| 6 | `has_balcony_or_terrace` | 0.177 |
| 7 | `distancia_centro_km` | −0.160 (inversa) |

> El aire acondicionado genera una diferencia de precio de **+55€/noche** de media
> en Málaga, siendo el servicio con mayor impacto individual del dataset.
""")