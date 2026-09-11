## Formulas geográficas y geocodificación para direcciones
# frontend/utils/geo.py
import math
from geopy.geocoders import Nominatim


# ── Coordenadas de referencia ─────────────────────────────────────────────────
CENTRO_MALAGA = (36.7209914, -4.4216968)   # Calle Larios

PLAYAS_MALAGA = {
    "El Cañuelo":       (36.6450, -4.4920),
    "Campo de Golf":    (36.6544, -4.4740),
    "Arraijanal":       (36.6600, -4.4680),
    "Guadalmar":        (36.6625, -4.4819),
    "Desembocadura":    (36.6690, -4.4590),
    "Sacaba":           (36.6855, -4.4578),
    "Misericordia":     (36.6961, -4.4447),
    "San Andres":       (36.7061, -4.4320),
    "Malagueta":        (36.7186, -4.4079),
    "La Caleta":        (36.7190, -4.3980),
    "Baños del Carmen": (36.7215, -4.3840),
    "Pedregalejo":      (36.7196, -4.3725),
    "Las Acacias":      (36.7190, -4.3650),
    "El Palo":          (36.7175, -4.3592),
    "El Candado":       (36.7145, -4.3467),
    "Peñon del Cuervo": (36.7134, -4.3364),
    "El Cementerio":    (36.7130, -4.3320),
    "La Araña":         (36.7123, -4.3211),
    "El Hornillo":      (36.7100, -4.3150),
}


# ── Fórmula Haversine (idéntica a la del notebook) ───────────────────────────
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos puntos geográficos usando la fórmula Haversine."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ── Funciones públicas ────────────────────────────────────────────────────────
def distancia_al_centro(lat: float, lon: float) -> float:
    """Distancia en km desde las coordenadas dadas hasta el centro de Málaga."""
    return round(haversine_distance(lat, lon, *CENTRO_MALAGA), 2)


def distancia_playa_mas_cercana(lat: float, lon: float) -> float:
    """
    Distancia en km a la playa más cercana del litoral malagueño.
    Replica exactamente el cálculo del notebook de ingeniería de características.
    """
    return round(min(
        haversine_distance(lat, lon, playa_lat, playa_lon)
        for playa_lat, playa_lon in PLAYAS_MALAGA.values()
    ), 2)


def geocodificar_direccion(direccion: str) -> tuple[float, float] | None:
    """
    Convierte una dirección textual en (latitud, longitud).
    Devuelve None si no se encuentra la dirección.
    """
    try:
        geolocator = Nominatim(user_agent="smarthost_analytics")
        location   = geolocator.geocode(f"{direccion}, Málaga, España", timeout=5)
        if location:
            return (location.latitude, location.longitude)
        return None
    except Exception:
        return None