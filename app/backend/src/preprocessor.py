# app/src/preprocessor.py
#
# RESPONSABILIDAD: Replicar exactamente el preprocesamiento manual que hizo
# el notebook 05_FusionRRNNs.ipynb en la Celda 5 para el Modo 2.
#
# POR QUÉ EXISTE ESTE ARCHIVO si ya hay un ColumnTransformer (preprocessor_V3.pkl)?
#   El Modo 1 usa el ColumnTransformer completo (OneHotEncoder + StandardScaler).
#   El Modo 2 usó un preprocesamiento manual diferente en el notebook:
#     - host_response_time → pd.Categorical().codes → entero 0-3
#     - room_type          → pd.Categorical().codes → entero 0-3
#     - resto de columnas  → float sin transformar
#     - LUEGO StandardScaler solo sobre ese array numérico
#   Por eso el Modo 2 necesita este archivo y NO puede usar el ColumnTransformer.
#
# RIESGO: Los mapas HOST_RESPONSE_MAP y ROOM_TYPE_MAP deben coincidir
# con el orden en que pd.Categorical() los asignó en el notebook.
# Si no coinciden, las predicciones serán silenciosamente incorrectas.

import numpy as np

# Mapeo categórico — debe coincidir con pd.Categorical().codes del notebook
HOST_RESPONSE_MAP = {
    "within an hour":      0,
    "within a few hours":  1,
    "within a day":        2,
    "a few days or more":  3,
}

ROOM_TYPE_MAP = {
    "Entire home/apt": 0,
    "Private room":    1,
    "Shared room":     2,
    "Hotel room":      3,
}

# Orden EXACTO de las 30 columnas — debe coincidir con tabular_cols del notebook
TABULAR_COLS = [
    "host_response_time", "host_response_rate", "host_is_superhost",
    "longitude", "room_type", "accommodates", "bathrooms", "bedrooms",
    "beds", "number_of_reviews_ltm", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_location", "review_scores_value", "reviews_per_month",
    "private_bathroom", "has_cooking_basics", "has_tv", "has_air_conditioning",
    "has_washer", "has_heating", "has_freezer", "has_coffee_maker",
    "has_balcony_or_terrace", "distancia_centro_km", "personas_por_habitacion",
    "banos_por_huesped", "amenities_score", "distancia_playa_km",
]

def build_tabular_vector(form_data: dict) -> np.ndarray:
    """
    Convierte el dict del formulario al vector (1, 30) que espera el scaler del Modo 2.
    Aplica encoding categórico manual y devuelve un array NumPy puro SIN nombres
    de columna para que StandardScaler no lance el UserWarning.
    """
    row = []
    for col in TABULAR_COLS:
        val = form_data.get(col, 0)

        if col == "host_response_time":
            val = HOST_RESPONSE_MAP.get(str(val), 1)  # default: "within a few hours"
        elif col == "room_type":
            val = ROOM_TYPE_MAP.get(str(val), 0)       # default: "Entire home/apt"
        else:
            val = float(val) if val is not None else 0.0

        row.append(float(val))

    return np.array(row, dtype=np.float64).reshape(1, -1)  # shape (1, 30)