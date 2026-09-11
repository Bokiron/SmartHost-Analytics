# app/services/roi.py
#
# RESPONSABILIDAD: Toda la lógica financiera del proyecto.
# Este archivo traduce las predicciones de la IA en cifras comprensibles
# para un inversor: ingresos anuales, ROI de una reforma, payback period.
#
# FÓRMULAS:
#
#   días_ocupados_año = reviews_per_month × 2 × estancia_media × 12
#     - ×2 porque se asume que solo 1 de cada 2 huéspedes deja reseña
#     - estancia_media_Málaga = 3.5 noches (ajustable)
#
#   ingresos_anuales = precio_noche × días_ocupados
#
#   ROI = (ingresos_después - ingresos_antes) / inversión × 100
#   Payback = inversión / beneficio_mensual_extra
#
# calcular_roi() es la función estrella del proyecto: permite al inversor
# simular "¿cuánto tardo en recuperar 2.000€ de reforma?"

ESTANCIA_MEDIA_MALAGA = 3.5   # noches promedio por reserva en Málaga
FACTOR_RESENAS        = 2.0   # corrección: no todos los huéspedes dejan reseña

def calcular_dias_ocupados(reviews_per_month: float) -> float:
    """Estima los días ocupados al año a partir de las reseñas mensuales."""
    return min(round(reviews_per_month * FACTOR_RESENAS * ESTANCIA_MEDIA_MALAGA * 12, 1), 365)

def calcular_ingresos_anuales(precio_noche: float, dias_ocupados: float) -> float:
    """Ingresos brutos anuales estimados."""
    return round(precio_noche * dias_ocupados, 2)

def calcular_roi(ingresos_antes, ingresos_despues, inversion):
    beneficio_anual  = ingresos_despues - ingresos_antes
    beneficio_mes    = beneficio_anual / 12

    # ROI financiero: rendimiento del dinero invertido
    roi_inversion_pct = round((beneficio_anual / inversion) * 100, 1) if inversion > 0 else 0.0

    # Crecimiento de ingresos: cuánto crecen los ingresos en %
    crecimiento_ingresos_pct = round((beneficio_anual / ingresos_antes) * 100, 1) if ingresos_antes > 0 else 0.0

    # Payback: en cuántos meses recuperas la inversión
    payback_meses = round(inversion / beneficio_mes, 1) if (inversion > 0 and beneficio_mes > 0) else None

    return {
        "beneficio_extra_anual":      round(beneficio_anual, 2),
        "beneficio_extra_mensual":    round(beneficio_mes, 2),
        "roi_inversion_pct":          roi_inversion_pct,        # 69%
        "crecimiento_ingresos_pct":   crecimiento_ingresos_pct, # 31.7%
        "payback_meses":              payback_meses,            # 17.4 meses
    }