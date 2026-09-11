# app/routers/predict.py
#
# RESPONSABILIDAD: Definir los endpoints HTTP de predicción y orquestar
# el flujo de cada petición. Este archivo NO contiene lógica de negocio —
# solo recibe, valida y delega a los servicios.
#
# ENDPOINTS:
#   POST /predict/tabular    → Modo 1: solo datos tabulares
#                              Body: JSON con PredictionRequest
#                              Útil para comparar el benchmark
#
#   POST /predict/multimodal → Modo 2: datos + imagen (el endpoint principal)
#                              Body: multipart/form-data con:
#                                - image:   archivo .jpg/.png
#                                - request: string JSON con los campos tabulares
#
# FLUJO de /predict/multimodal:
#   1. FastAPI valida que llegue image y request
#   2. Se parsea el JSON del campo request → PredictionRequest (Pydantic valida)
#   3. Se lee la imagen con PIL
#   4. predecir_tabular()  → precio_base  (Modo 1)
#   5. predecir_multimodal() → precio_visual (Modo 2)
#   6. calcular_roi() → días ocupados, ingresos anuales, impacto visual
#   7. Se devuelve PredictionResponse completo

import io
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image

from app.backend.schemas.prediction import PredictionRequest, PredictionResponse
from app.backend.services.predictor import predecir_tabular, predecir_multimodal
from app.backend.services.roi import calcular_dias_ocupados, calcular_ingresos_anuales
from app.backend.exceptions.handlers import ImageProcessingError

router = APIRouter(prefix="/predict", tags=["Predicción"])

@router.post("/tabular", response_model=PredictionResponse)
def predict_tabular(request: PredictionRequest):
    """Modo 1 — Solo datos tabulares. Sirve como benchmark comparativo."""
    precio_base   = predecir_tabular(request)
    dias_ocupados = calcular_dias_ocupados(request.reviews_per_month)
    ingresos_base = calcular_ingresos_anuales(precio_base, dias_ocupados)

    return PredictionResponse(
        precio_base=precio_base,
        precio_visual=None,
        dias_ocupados_anio=dias_ocupados,
        ingresos_anuales_base=ingresos_base,
        ingresos_anuales_visual=None,
        impacto_visual_eur=None,
        impacto_visual_pct=None,
    )

@router.post("/multimodal", response_model=PredictionResponse)
async def predict_multimodal(
    image:   UploadFile = File(..., description="Foto de portada del apartamento"),
    request: str        = Form(..., description="JSON con los datos tabulares"),
):
    """
    Modo 2 — Tabular + imagen. Devuelve precio base, precio visual y ROI.
    Enviar como multipart/form-data con dos campos:
      - image:   archivo binario .jpg/.png
      - request: string JSON con todos los campos de PredictionRequest
    """
    # Paso 1: parsear y validar el JSON tabulares con Pydantic
    try:
        data = PredictionRequest(**json.loads(request))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"JSON inválido en 'request': {e}")

    # Paso 2: leer la imagen y convertirla a objeto PIL
    try:
        contents  = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise ImageProcessingError(f"No se pudo abrir la imagen: {e}")

    # Paso 3: predicciones de ambos modelos
    precio_base   = predecir_tabular(data)
    precio_visual = predecir_multimodal(data, pil_image)

    # Paso 4: lógica de negocio (ROI)
    dias_ocupados   = calcular_dias_ocupados(data.reviews_per_month)
    ingresos_base   = calcular_ingresos_anuales(precio_base,   dias_ocupados)
    ingresos_visual = calcular_ingresos_anuales(precio_visual, dias_ocupados)
    impacto_eur     = round(precio_visual - precio_base, 2)
    impacto_pct     = round((impacto_eur / precio_base) * 100, 1) if precio_base else 0.0

    return PredictionResponse(
        precio_base=precio_base,
        precio_visual=precio_visual,
        dias_ocupados_anio=dias_ocupados,
        ingresos_anuales_base=ingresos_base,
        ingresos_anuales_visual=ingresos_visual,
        impacto_visual_eur=impacto_eur,
        impacto_visual_pct=impacto_pct,
    )