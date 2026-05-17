# app/routers/health.py
#
# RESPONSABILIDAD: Exponer endpoints de monitorización del servidor.
#
# ENDPOINTS:
#   GET /        → Bienvenida + link a /docs
#   GET /health  → Estado del servidor y los modelos
#                  Útil para verificar que el lifespan cargó todo correctamente
#                  antes de lanzar el frontend de Streamlit

from fastapi import APIRouter
from app.backend.core.loader import artifacts

router = APIRouter(tags=["General"])

@router.get("/")
def root():
    return {"message": "Bienvenido a SmartHost Analytics API", "docs": "/docs"}

@router.get("/health")
def health_check():
    """Comprueba que el servidor y los modelos están activos."""
    modelos_cargados = "tabular_model" in artifacts and "multimodal_model" in artifacts
    return {
        "status":           "ok" if modelos_cargados else "degraded",
        "modelos_cargados": modelos_cargados,
        "dispositivo":      str(artifacts.get("device", "desconocido")),
        "version":          "1.0.0",
    }