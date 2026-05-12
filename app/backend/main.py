# app/main.py
#
# RESPONSABILIDAD: Punto de entrada de la aplicación FastAPI.
# Este archivo solo debe hacer tres cosas:
#   1. Crear la instancia de la app con su configuración
#   2. Registrar el middleware (CORS)
#   3. Registrar los routers y manejadores de errores
#
# Todo lo demás (carga de modelos, lógica de predicción, endpoints)
# vive en sus módulos correspondientes dentro de app/.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.core.loader import lifespan
from app.backend.routers import health, predict
from app.backend.exceptions.handlers import register_exception_handlers


app = FastAPI(
    title="SmartHost Analytics API",
    description="Predicción de precios para apartamentos turísticos en Málaga.",
    version="1.0.0",
    lifespan=lifespan,   # Carga los modelos al arrancar (ver core/loader.py)
)

# CORS abierto para desarrollo local — en producción restringir allow_origins
# a la URL del frontend de Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)   # Errores personalizados (ver exceptions/handlers.py)
app.include_router(health.router)  # GET /  y  GET /health
app.include_router(predict.router) # POST /predict/tabular  y  POST /predict/multimodal