# app/exceptions/handlers.py
#
# RESPONSABILIDAD: Centralizar el manejo de errores personalizados.
# Sin esto, cualquier excepción no controlada devuelve un 500 genérico
# sin información útil. Con esto, cada tipo de error tiene su código HTTP
# y mensaje claro.
#
# ERRORES definidos:
#   ModelNotLoadedError  → 503 Service Unavailable
#                          Si alguien llama a /predict antes de que los modelos
#                          estén cargados (raro, pero posible en arranque lento)
#
#   ImageProcessingError → 422 Unprocessable Entity
#                          Si la imagen subida está corrupta o tiene un formato que PIL no puede leer

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class ModelNotLoadedError(Exception):
    pass

class ImageProcessingError(Exception):
    pass

def register_exception_handlers(app: FastAPI) -> None:
    """Registra todos los manejadores en la app de FastAPI."""

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code=503,
            content={"error": "Modelo no disponible", "detail": str(exc)},
        )

    @app.exception_handler(ImageProcessingError)
    async def image_processing_handler(request: Request, exc: ImageProcessingError):
        return JSONResponse(
            status_code=422,
            content={"error": "Error al procesar la imagen", "detail": str(exc)},
        )