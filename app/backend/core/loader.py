# app/core/loader.py
#
# RESPONSABILIDAD: Cargar TODOS los artefactos de ML en memoria UNA SOLA VEZ
# cuando el servidor arranca. Esto es crítico para el rendimiento: si cargáramos
# los modelos en cada petición, cada request tardaría ~10 segundos en lugar de
# milisegundos.
#
# PATRÓN: "lifespan" de FastAPI — código que se ejecuta al inicio (antes del yield)
# y al apagado (después del yield). Es el equivalente a @PostConstruct en Spring.
#
# ARTEFACTOS que carga:
#   - preprocessor_V3.pkl     → ColumnTransformer del Modo 1 (OHE + Scaler)
#   - airbnb_mlp_*.pt         → Pesos del modelo tabular (Modo 1)
#   - scaler_tabular.pkl      → StandardScaler del Modo 2
#   - multimodal_mlp.pt       → Pesos del modelo de fusión (Modo 2)
#   - multimodal_mlp_metadata.json → Orden de columnas, dimensiones, métricas
#
# Todos los artefactos se guardan en el dict global `artifacts` que
# el resto de módulos importan para acceder a los modelos.

import json
import joblib
import torch
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

from app.backend.nn_models.networks import AirbnbMLP, MultimodalMLP

MODELS_DIR = Path("models")

PREPROCESSOR_PATH  = MODELS_DIR / "preprocessor_V3.pkl"
TABULAR_MODEL_PATH = MODELS_DIR / "airbnb_mlp_PriceCAPV3_NoLog.pt"
SCALER_TAB_PATH    = MODELS_DIR / "scaler_tabular.pkl"
MULTIMODAL_PATH    = MODELS_DIR / "multimodal_mlp.pt"
METADATA_PATH      = MODELS_DIR / "multimodal_mlp_metadata.json"

# Contenedor global compartido entre todos los módulos.
# Actúa como un "registro de servicios" ligero.
artifacts: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

    # 1. Metadatos del modelo multimodal
    # Contiene: tabular_cols (orden de las 30 columnas), fusion_dim (542),
    # métricas de entrenamiento, etc.
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    tabular_cols = metadata["tabular_cols"]  # lista de 30 nombres en orden exacto
    fusion_dim   = metadata["fusion_dim"]    # 542 = 30 tabulares + 512 visuales

    # 2. Modo 1 — ColumnTransformer completo (OHE + StandardScaler)
    # Este preprocesador fue entrenado en el notebook 03_ModeloTabular
    # y sabe cómo transformar el DataFrame completo de una vez.
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Inferimos el tamaño de entrada real del modelo tabular pasando una fila
    # dummy por el preprocesador (el OHE puede expandir las columnas)
    dummy_df = pd.DataFrame([dict.fromkeys(preprocessor.feature_names_in_, 0)])
    tabular_input_size = preprocessor.transform(dummy_df).shape[1]

    tabular_model = AirbnbMLP(input_size=tabular_input_size).to(device)
    tabular_model.load_state_dict(
        torch.load(TABULAR_MODEL_PATH, map_location=device, weights_only=True)
    )
    tabular_model.eval()  # Desactiva Dropout/BatchNorm en modo inferencia

    # 3. Modo 2 — StandardScaler separado + modelo de fusión
    # El Modo 2 usa su propio scaler (entrenado solo con las 30 columnas numéricas
    # ya codificadas manualmente) en lugar del ColumnTransformer del Modo 1.
    scaler_tab = joblib.load(SCALER_TAB_PATH)

    multimodal_model = MultimodalMLP(fusion_dim=fusion_dim).to(device)
    multimodal_model.load_state_dict(
        torch.load(MULTIMODAL_PATH, map_location=device, weights_only=True)
    )
    multimodal_model.eval()

    # 4. Publicar todos los artefactos en el dict global
    artifacts.update({
        "device":           device,
        "preprocessor":     preprocessor,
        "tabular_model":    tabular_model,
        "scaler_tab":       scaler_tab,
        "tabular_cols":     tabular_cols,
        "multimodal_model": multimodal_model,
        "metadata":         metadata,
    })

    print(" Todos los artefactos cargados correctamente.")
    yield  # El servidor corre aquí — todo lo de abajo es el apagado

    # Liberar memoria GPU/RAM al apagar el servidor
    artifacts.clear()
    print(" Servidor apagado. Artefactos liberados.")