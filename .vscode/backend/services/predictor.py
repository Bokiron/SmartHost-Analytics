# app/services/predictor.py
#
# RESPONSABILIDAD: Ejecutar la inferencia de los modelos de ML.
# Contiene toda la lógica de transformación de datos justo antes de pasarlos
# a las redes neuronales. Es el corazón técnico del backend.
#
# FUNCIONES:
#   predecir_tabular()    → Modo 1
#     1. Construye un DataFrame con los datos del request
#     2. Lo transforma con el ColumnTransformer (OHE + Scaler)
#     3. Pasa el tensor por AirbnbMLP → precio float
#
#   predecir_multimodal() → Modo 2
#     RAMA TABULAR:
#       1. build_tabular_vector() codifica host_response_time y room_type
#          a enteros y construye un array NumPy puro (1, 30)
#       2. scaler_tab.transform() escala el array (sin nombres de columna)
#     RAMA VISUAL:
#       3. Aplica las transformaciones de imagen de ImageNet
#          (Resize 256 → CenterCrop 224 → ToTensor → Normalize)
#       4. ResNet34 extrae el embedding visual (1, 512)
#     FUSIÓN:
#       5. MultimodalMLP concatena ambos tensores → precio float
#
# NOTA IMPORTANTE sobre ResNet34:
#   Se carga lazy (solo la primera vez que se llama) y se reutiliza en todas
#   las peticiones. La capa fc se reemplaza por nn.Identity() para obtener
#   el embedding de 512 dimensiones en lugar de las 1000 clases de ImageNet.

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.models as tv_models
import torch.nn as nn
from torchvision import transforms

from app.backend.schemas.prediction import PredictionRequest
from app.backend.core.loader import artifacts
from app.backend.src.preprocessor import build_tabular_vector

# Pipeline de transformación de imagen — DEBE ser idéntico al del notebook
# de entrenamiento. Cualquier diferencia en Normalize() invalidará los embeddings.
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),         # Encoge el lado corto a 256px (sin deformar)
    transforms.CenterCrop(224),     # Recorta el centro 224x224 (estándar ImageNet)
    transforms.ToTensor(),          # PIL Image → tensor [0,1]
    transforms.Normalize(           # Normalización ImageNet (media y std de ImageNet)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Instancia singleton de ResNet34. Se inicializa en la primera petición
# y se reutiliza en todas las siguientes (carga lazy).
_resnet: nn.Module = None

def _get_resnet(device: torch.device) -> nn.Module:
    """Carga ResNet34 preentrenado la primera vez y lo cachea para reutilizarlo."""
    global _resnet
    if _resnet is None:
        resnet = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()  # Elimina la cabeza clasificadora → salida (512,)
        resnet = resnet.to(device)
        resnet.eval()
        _resnet = resnet
    return _resnet


def predecir_tabular(request: PredictionRequest) -> float:
    """Modo 1: predicción con ColumnTransformer + AirbnbMLP."""
    preprocessor = artifacts["preprocessor"]
    model        = artifacts["tabular_model"]
    device       = artifacts["device"]

    # El ColumnTransformer espera un DataFrame con nombres de columna
    df = pd.DataFrame([request.model_dump()])
    X_scaled = preprocessor.transform(df).astype(np.float32)

    tensor = torch.tensor(X_scaled).to(device)
    with torch.no_grad():
        model.eval()
        precio = model(tensor).item()

    return round(float(precio), 2)


def predecir_multimodal(request: PredictionRequest, image: Image.Image) -> float:
    """Modo 2: predicción con datos tabulares + embedding visual de ResNet34."""
    scaler_tab       = artifacts["scaler_tab"]
    multimodal_model = artifacts["multimodal_model"]
    device           = artifacts["device"]
    resnet           = _get_resnet(device)

    # RAMA TABULAR
    # build_tabular_vector codifica las columnas categóricas a enteros
    # y devuelve un array NumPy puro (sin nombres de columna) para que
    # el StandardScaler no lance el UserWarning de feature names
    x_raw     = build_tabular_vector(request.model_dump())  # (1, 30) float64
    X_tab     = scaler_tab.transform(x_raw).astype(np.float32)  # (1, 30) escalado
    tab_tensor = torch.tensor(X_tab).to(device)

    # RAMA VISUAL
    img_tensor = IMAGE_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        vis_embedding = resnet(img_tensor)  # (1, 512)

    # FUSIÓN → precio
    with torch.no_grad():
        multimodal_model.eval()
        precio = multimodal_model(tab_tensor, vis_embedding).item()

    return round(float(precio), 2)