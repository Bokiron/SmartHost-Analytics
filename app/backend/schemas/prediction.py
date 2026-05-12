# app/schemas/prediction.py
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Datos tabulares del apartamento. Equivalente al DTO de entrada en Spring."""

    # Datos del host
    host_response_time:  str   = Field(..., example="within an hour")
    host_response_rate:  float = Field(..., ge=0.0, le=1.0, example=0.95)
    host_is_superhost:   float = Field(..., example=1.0)

    # Ubicación
    longitude:           float = Field(..., example=-4.42)
    distancia_centro_km: float = Field(..., ge=0.0, example=1.2)
    distancia_playa_km:  float = Field(..., ge=0.0, example=0.8)

    # Tipo de alojamiento
    room_type:           str   = Field(..., example="Entire home/apt")

    # Características físicas
    accommodates:        int   = Field(..., ge=1, example=4)
    bathrooms:           float = Field(..., ge=0.0, example=1.0)
    bedrooms:            int   = Field(..., ge=0, example=2)
    beds:                int   = Field(..., ge=0, example=2)

    # Reviews
    number_of_reviews_ltm:    int   = Field(..., ge=0, example=12)
    review_scores_rating:     float = Field(..., ge=0.0, le=5.0, example=4.8)
    review_scores_accuracy:   float = Field(..., ge=0.0, le=5.0, example=4.7)
    review_scores_cleanliness:float = Field(..., ge=0.0, le=5.0, example=4.9)
    review_scores_location:   float = Field(..., ge=0.0, le=5.0, example=4.6)
    review_scores_value:      float = Field(..., ge=0.0, le=5.0, example=4.5)
    reviews_per_month:        float = Field(..., ge=0.0, example=1.5)

    # Amenities (binarios 0/1)
    private_bathroom:      float = Field(..., example=1.0)
    has_cooking_basics:    float = Field(..., example=1.0)
    has_tv:                float = Field(..., example=1.0)
    has_air_conditioning:  float = Field(..., example=1.0)
    has_washer:            float = Field(..., example=1.0)
    has_heating:           float = Field(..., example=1.0)
    has_freezer:           float = Field(..., example=0.0)
    has_coffee_maker:      float = Field(..., example=1.0)
    has_balcony_or_terrace:float = Field(..., example=0.0)

    # Ratios calculados
    personas_por_habitacion: float = Field(..., ge=0.0, example=2.0)
    banos_por_huesped:       float = Field(..., ge=0.0, example=0.5)
    amenities_score:         float = Field(..., ge=0.0, example=7.0)


class PredictionResponse(BaseModel):
    """Respuesta completa con precios e ingresos estimados."""

    # Precios por noche
    precio_base:   float = Field(..., description="Precio predicho solo con datos tabulares (€/noche)")
    precio_visual: Optional[float] = Field(None, description="Precio predicho con datos + foto (€/noche)")

    # Lógica de negocio
    dias_ocupados_anio:     float = Field(..., description="Días ocupados estimados al año")
    ingresos_anuales_base:  float = Field(..., description="Ingresos anuales estimados con precio base (€)")
    ingresos_anuales_visual:Optional[float] = Field(None, description="Ingresos anuales con precio visual (€)")

    # Diferencial visual
    impacto_visual_eur:     Optional[float] = Field(None, description="Diferencia de precio por la foto (€/noche)")
    impacto_visual_pct:     Optional[float] = Field(None, description="Diferencia porcentual por la foto (%)")