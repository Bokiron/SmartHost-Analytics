import pandas as pd
import requests
import os
import time
from pathlib import Path

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
# Path(__file__) obtiene la ruta absoluta de este script (src/script_imagenes_portada.py)
# .resolve().parent sube un nivel para apuntar a la carpeta src/
BASE_DIR = Path(__file__).resolve().parent

# Construimos las rutas a data/ subiendo otro nivel desde src/ hasta la raíz del proyecto
CSV_PATH = BASE_DIR.parent / 'data' / 'listingV3.csv'
IMG_DIR  = BASE_DIR.parent / 'data' / 'Front_Images'
# CSV donde guardaremos los IDs de apartamentos cuya imagen no se pudo descargar,
# para limpiarlos posteriormente en el notebook de EDA

TIMEOUT = 10   # Segundos máximos esperando respuesta del servidor antes de abortar
SLEEP   = 0.2  # Pausa entre peticiones para no saturar el servidor y evitar bloqueos de IP

# ── PREPARACIÓN ───────────────────────────────────────────────────────────────
# Crea la carpeta de imágenes si no existe. parents=True crea carpetas intermedias
# si fueran necesarias. exist_ok=True evita error si ya existe.
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)

# Cargamos solo las columnas que necesitamos: el ID del listing (para nombrar el archivo)
# y la URL de la foto de portada. Eliminamos filas sin URL con dropna().
# ⚠️  Sin .head() → descarga TODAS las filas del CSV
df = pd.read_csv(CSV_PATH)[['id', 'picture_url']].dropna()
#df = df.head() # cabecera para probar funcionamiento

print(f"📋 Total de imágenes a descargar: {len(df)}")

# Cabecera HTTP que simula un navegador real. Sin esto, Airbnb puede devolver
# error 403 Forbidden al detectar que la petición viene de un script automático.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
}

# ── DESCARGA ──────────────────────────────────────────────────────────────────
descargadas = 0
fallidas    = 0

for _, row in df.iterrows():
    listing_id = row['id']
    url        = row['picture_url']

    # ── Extracción segura de extensión ────────────────────────────────────────
    # Problema: algunas URLs no tienen extensión reconocible al final (terminan
    # en un UUID sin .jpeg/.jpg). url.split('.')[-1] extraería en esos casos
    # algo como "com/pictures/.../uuid", que contiene barras '/' y rompe la ruta.
    # Solución: tomamos solo la última parte del path de la URL (antes de '?'),
    # verificamos si tiene extensión válida y si no, asignamos .jpeg por defecto.
    url_path   = url.split('?')[0]          # eliminamos parámetros de query
    url_ending = url_path.split('/')[-1]    # último segmento del path (ej: "foto.jpeg")
    ext_candidate = url_ending.split('.')[-1].lower() if '.' in url_ending else ''

    # Solo aceptamos extensiones de imagen conocidas; cualquier otra → jpeg por defecto
    ext      = ext_candidate if ext_candidate in ('jpeg', 'jpg', 'png', 'webp') else 'jpeg'
    filename = os.path.join(IMG_DIR, f"{listing_id}.{ext}")

    # Si ya existe en disco, la saltamos (reanudación del proceso sin repetir trabajo)
    if os.path.exists(filename):
        descargadas += 1
        continue

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)

        descargadas += 1
        time.sleep(SLEEP)

    except requests.exceptions.RequestException as e:
        # Error de red, timeout o HTTP error (404, 403, 500...)
        # Registramos el ID para limpiarlo después en el notebook
        print(f"  ❌ Error en ID {listing_id}: {e}")
        fallidas += 1

    if descargadas % 100 == 0 and descargadas > 0:
        print(f"  ✅ {descargadas} / {len(df)} descargadas...")

print(f"\n🏁 Proceso completado: {descargadas} descargadas, {fallidas} fallidas.")
print(f"📁 Imágenes guardadas en: {IMG_DIR}")
