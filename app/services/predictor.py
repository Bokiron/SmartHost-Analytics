# app/services/predictor.py
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.models as tv_models
import torch.nn as nn
from torchvision import transforms

from src.preprocessor import build_tabular_vector
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_resnet = None

def _get_resnet(device: torch.device) -> nn.Module:
    global _resnet
    if _resnet is None:
        resnet = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        resnet = resnet.to(device)
        resnet.eval()
        _resnet = resnet
    return _resnet

def predecir_tabular(form_data: dict, artifacts: dict) -> float:
    preprocessor = artifacts["preprocessor"]
    model        = artifacts["tabular_model"]
    device       = artifacts["device"]

    df = pd.DataFrame([form_data])
    X_scaled = preprocessor.transform(df).astype(np.float32)

    tensor = torch.tensor(X_scaled).to(device)
    with torch.no_grad():
        precio = model(tensor).item()

    return round(float(precio), 2)

def predecir_multimodal(form_data: dict, image: Image.Image, artifacts: dict) -> float:
    scaler_tab       = artifacts["scaler_tab"]
    multimodal_model = artifacts["multimodal_model"]
    device           = artifacts["device"]
    resnet           = _get_resnet(device)

    x_raw = build_tabular_vector(form_data)
    X_tab = scaler_tab.transform(x_raw).astype(np.float32)
    tab_tensor = torch.tensor(X_tab).to(device)

    img_tensor = IMAGE_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vis_embedding = resnet(img_tensor)

    with torch.no_grad():
        precio = multimodal_model(tab_tensor, vis_embedding).item()

    return round(float(precio), 2)