# frontend/utils/calculos.py
import pandas as pd
import os

_AMENITIES_SCORE_KEYS = [
    "has_cooking_basics", "has_tv", "has_air_conditioning", "has_washer",
    "has_heating", "has_freezer", "has_coffee_maker", "has_balcony_or_terrace",
    "has_kitchen", "has_hair_dryer", "has_iron", "has_bed_linens",
    "has_microwave", "has_refrigerator", "has_toaster",
]


def calcular_ratios(accommodates: int, bedrooms: int, bathrooms: float) -> dict:
    personas_por_habitacion = accommodates / bedrooms if bedrooms > 0 else float(accommodates)
    banos_por_huesped       = bathrooms / accommodates if accommodates > 0 else 0.0
    return {
        "personas_por_habitacion": round(personas_por_habitacion, 2),
        "banos_por_huesped":       round(banos_por_huesped, 3),
    }


def calcular_amenities_score(form: dict) -> float:
    return float(sum(form.get(key, False) for key in _AMENITIES_SCORE_KEYS))


def construir_payload(form: dict) -> dict:
    ratios          = calcular_ratios(form["accommodates"], form["bedrooms"], form["bathrooms"])
    amenities_score = calcular_amenities_score(form)

    return {
        "host_response_time":        form["host_response_time"],
        "host_response_rate":        form["host_response_rate"],
        "host_is_superhost":         float(form["host_is_superhost"]),
        "longitude":                 form["longitude"],
        "distancia_centro_km":       form["distancia_centro"],
        "distancia_playa_km":        form["distancia_playa"],
        "room_type":                 form["room_type"],
        "accommodates":              form["accommodates"],
        "bathrooms":                 form["bathrooms"],
        "bedrooms":                  form["bedrooms"],
        "beds":                      form["beds"],
        "number_of_reviews_ltm":     form["number_of_reviews_ltm"],
        "review_scores_rating":      form["rating"],
        "review_scores_accuracy":    form["accuracy"],
        "review_scores_cleanliness": form["cleanliness"],
        "review_scores_location":    form["location"],
        "review_scores_value":       form["value"],
        "reviews_per_month":         form["reviews_per_month"],
        "private_bathroom":          float(form["private_bathroom"]),
        "has_cooking_basics":        float(form["has_cooking_basics"]),
        "has_tv":                    float(form["has_tv"]),
        "has_air_conditioning":      float(form["has_air_conditioning"]),
        "has_washer":                float(form["has_washer"]),
        "has_heating":               float(form["has_heating"]),
        "has_freezer":               float(form["has_freezer"]),
        "has_coffee_maker":          float(form["has_coffee_maker"]),
        "has_balcony_or_terrace":    float(form["has_balcony_or_terrace"]),
        **ratios,
        "amenities_score":           amenities_score,
    }

## Calcular medias de valoraciones para nuevos anfitriones (sin historial)
_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'listingV5_PriceCapID.csv')


def _cargar_medias_valoraciones() -> dict:
    """Calcula las medias de valoraciones del mercado de Málaga."""
    try:
        df = pd.read_csv(_DATA_PATH)
        return {
            "reviews_per_month":     round(df['reviews_per_month'].mean(), 1),
            "number_of_reviews_ltm": int(round(df['number_of_reviews_ltm'].mean())),
            "rating":                round(df['review_scores_rating'].mean(), 1),
            "accuracy":              round(df['review_scores_accuracy'].mean(), 1),
            "cleanliness":           round(df['review_scores_cleanliness'].mean(), 1),
            "location":              round(df['review_scores_location'].mean(), 1),
            "value":                 round(df['review_scores_value'].mean(), 1),
        }
    except FileNotFoundError:
        return {
            "reviews_per_month":     1.5,
            "number_of_reviews_ltm": 10,
            "rating":                4.7,
            "accuracy":              4.7,
            "cleanliness":           4.7,
            "location":              4.8,
            "value":                 4.6,
        }


# Valores de valoraciones = medias reales del mercado (se calculan una vez al arrancar)
DEFAULTS_VALORACIONES = _cargar_medias_valoraciones()
