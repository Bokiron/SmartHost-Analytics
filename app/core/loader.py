# app/core/loader.py
import json
import joblib
import torch
import pandas as pd
import streamlit as st
from pathlib import Path

from app.backend.nn_models.networks import AirbnbMLP, MultimodalMLP

MODELS_DIR = Path("models")

PREPROCESSOR_PATH  = MODELS_DIR / "preprocessor_V3.pkl"
TABULAR_MODEL_PATH = MODELS_DIR / "airbnb_mlp_PriceCAPV3_NoLog.pt"
SCALER_TAB_PATH    = MODELS_DIR / "scaler_tabular.pkl"
MULTIMODAL_PATH    = MODELS_DIR / "multimodal_mlp.pt"
METADATA_PATH      = MODELS_DIR / "multimodal_mlp_metadata.json"

@st.cache_resource(show_spinner="Cargando redes neuronales en memoria...")
def load_all_artifacts():
    """Carga los modelos en caché la primera vez que se abre la app."""
    # Forzamos CPU para evitar errores de memoria en la nube
    device = torch.device("cpu")
    
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    tabular_cols = metadata["tabular_cols"]
    fusion_dim   = metadata["fusion_dim"]

    # Modo 1
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    dummy_df = pd.DataFrame([dict.fromkeys(preprocessor.feature_names_in_, 0)])
    tabular_input_size = preprocessor.transform(dummy_df).shape[1]

    tabular_model = AirbnbMLP(input_size=tabular_input_size).to(device)
    tabular_model.load_state_dict(torch.load(TABULAR_MODEL_PATH, map_location=device, weights_only=True))
    tabular_model.eval()

    # Modo 2
    scaler_tab = joblib.load(SCALER_TAB_PATH)
    multimodal_model = MultimodalMLP(fusion_dim=fusion_dim).to(device)
    multimodal_model.load_state_dict(torch.load(MULTIMODAL_PATH, map_location=device, weights_only=True))
    multimodal_model.eval()

    return {
        "device": device,
        "preprocessor": preprocessor,
        "tabular_model": tabular_model,
        "scaler_tab": scaler_tab,
        "tabular_cols": tabular_cols,
        "multimodal_model": multimodal_model,
        "metadata": metadata,
    }