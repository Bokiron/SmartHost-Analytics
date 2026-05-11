"""
00_resize_images.py
===================
Script de pre-procesado ONE-SHOT para SmartHost Analytics — Fase 3 CNN.

Convierte todas las imágenes de portada de Airbnb (resolución original variable)
al formato estándar de entrada de MobileNetV2: 224×224 píxeles.

Proceso:
Paso 1 — Resize(256)      : escala el lado más corto a 256px manteniendo proporción
Paso 2 — CenterCrop(224)  : recorta el cuadrado central de 224×224 px

EJECUCIÓN (una sola vez, antes de entrenar):
python scripts/00_resize_images.py

OUTPUT:
data/Front_Images_224/  →  5.667 archivos .jpg de exactamente 224×224 px
"""

import pathlib
import sys
import csv
import random
from PIL import Image, ImageFile

# ── Evitar crashes con imágenes truncadas o parcialmente corruptas ──────────
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None   # Suprimir DecompressionBombWarning (fotos >89Mpx)

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR   = BASE_DIR / "data" / "Front_Images"        # imágenes originales
DST_DIR   = BASE_DIR / "data" / "Front_Images_224"    # destino 224x224
SYNC_CSV  = BASE_DIR / "data" / "listingV5_CNN.csv"   # ← CSV ya sincronizado (5.667 filas)

# Parámetros de transformación (deben coincidir con torchvision en el notebook)
RESIZE_TO = 256   # Paso 1: lado más corto → 256 px
CROP_TO   = 224   # Paso 2: recorte central → 224×224 px
QUALITY   = 95    # Calidad JPEG de salida (95 = prácticamente sin pérdida)

# ============================================================
# VALIDACIONES PREVIAS
# ============================================================
if not SRC_DIR.exists():
    print(f"[ERROR] Carpeta de imágenes originales no encontrada: {SRC_DIR}")
    sys.exit(1)

if not SYNC_CSV.exists():
    print(f"[ERROR] CSV sincronizado no encontrado: {SYNC_CSV}")
    print("        Ejecuta primero la Sección 1 del notebook 04_ModeloCNN.ipynb")
    sys.exit(1)

# Leer los IDs válidos del CSV sincronizado
ids_validos = set()
with open(SYNC_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            ids_validos.add(int(float(row["id"])))
        except (ValueError, KeyError):
            pass

print(f"IDs válidos en CSV sincronizado : {len(ids_validos):,}")

# Crear carpeta de destino
DST_DIR.mkdir(parents=True, exist_ok=True)
print(f"Carpeta destino                 : {DST_DIR}")

# ============================================================
# INVENTARIO DE IMÁGENES FUENTE
# ============================================================
# Deduplicar con set de stems para evitar contar la misma imagen dos veces
# en sistemas case-insensitive (Windows/macOS) donde *.jpg y *.JPG
# pueden devolver los mismos ficheros físicos
stems_vistos = set()
image_files  = []

for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
    for f in SRC_DIR.glob(ext):
        if f.stem.lower() not in stems_vistos:
            stems_vistos.add(f.stem.lower())
            image_files.append(f)

# Filtrar solo los IDs que están en el CSV sincronizado
image_files_sync = []
for f in image_files:
    try:
        if int(f.stem) in ids_validos:
            image_files_sync.append(f)
    except ValueError:
        pass

print(f"Imágenes encontradas en disco   : {len(image_files):,}")
print(f"Imágenes a procesar (en CSV)    : {len(image_files_sync):,}")
print()

# ============================================================
# FUNCIÓN DE TRANSFORMACIÓN
# ============================================================
def resize_and_crop(img: Image.Image) -> Image.Image:
    """
    Replica exactamente el pipeline torchvision:
        1. Resize(256)     → lado más corto = 256px, proporción intacta
        2. CenterCrop(224) → cuadrado central de 224×224 px

    Ejemplo con imagen 1920×1080 (16:9):
        Paso 1: factor = 256/1080 ≈ 0.237 → nueva resolución: 455×256
        Paso 2: crop desde el centro → 224×224
            · Altura: corta 16px arriba + 16px abajo
            · Anchura: corta 115px izq + 116px der
    """
    w, h = img.size

    # Paso 1: escalar lado más corto a RESIZE_TO
    if w < h:
        new_w = RESIZE_TO
        new_h = round(h * RESIZE_TO / w)
    else:
        new_h = RESIZE_TO
        new_w = round(w * RESIZE_TO / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Paso 2: recortar cuadrado central de CROP_TO×CROP_TO
    w2, h2 = img.size
    left   = (w2 - CROP_TO) // 2
    top    = (h2 - CROP_TO) // 2
    right  = left + CROP_TO
    bottom = top  + CROP_TO

    img = img.crop((left, top, right, bottom))
    return img  # 224×224 exacto

# ============================================================
# BUCLE DE PROCESADO
# ============================================================
procesadas  = 0
ya_existian = 0
errores     = 0

total         = len(image_files_sync)
intervalo_log = max(1, total // 20)   # log cada ~5% de progreso

print("=" * 60)
print("  INICIANDO REDIMENSIONADO")
print(f"  Proceso: Resize({RESIZE_TO}) → CenterCrop({CROP_TO}×{CROP_TO})")
print("=" * 60)

for i, src_path in enumerate(image_files_sync, start=1):

    # Normalizar siempre el nombre de salida a {id}.jpg minúsculas
    dst_path = DST_DIR / f"{int(src_path.stem)}.jpg"

    # Saltar si ya existe (permite reanudar si el script se interrumpe)
    if dst_path.exists():
        ya_existian += 1
        if i % intervalo_log == 0:
            print(f"  [{i/total*100:5.1f}%] {i:>5}/{total}  omitida (ya existe): {src_path.name}")
        continue

    try:
        with Image.open(src_path) as img:
            img_out = resize_and_crop(img.convert("RGB"))
            img_out.save(dst_path, "JPEG", quality=QUALITY, optimize=True)
            procesadas += 1
    except Exception as e:
        print(f"  [ERROR] {src_path.name}: {e}")
        errores += 1
        continue

    if i % intervalo_log == 0:
        print(f"  [{i/total*100:5.1f}%] {i:>5}/{total}  procesada : {src_path.name}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print()
print("=" * 60)
print("  RESUMEN DE REDIMENSIONADO")
print("=" * 60)
print(f"  Total imágenes procesadas  : {procesadas:>6,}")
print(f"  Ya existían (omitidas)     : {ya_existian:>6,}")
print(f"  Errores                    : {errores:>6,}")
print(f"  Destino                    : {DST_DIR}")
print("=" * 60)

if errores > 0:
    print(f"\n  AVISO: {errores} imagen(es) fallaron.")
    print("  Estas filas deberán eliminarse del CSV sincronizado.")
    print("  Consulta la Sección 1 del notebook para re-sincronizar.")
else:
    print("\n  Todas las imágenes procesadas correctamente.")
    print("  Siguiente paso: actualizar IMG_DIR en el notebook a:")
    print(f"  data/Front_Images_224/")

# ============================================================
# VERIFICACIÓN RÁPIDA (muestra 3 imágenes aleatorias)
# ============================================================
muestras = random.sample(list(DST_DIR.glob("*.jpg")), min(3, procesadas + ya_existian))
print()
print("  Verificación aleatoria de tamaños en destino:")
for m in muestras:
    with Image.open(m) as img_check:
        assert img_check.size == (CROP_TO, CROP_TO), \
            f"ERROR: {m.name} tiene tamaño {img_check.size}, esperado ({CROP_TO}, {CROP_TO})"
        print(f"    OK  {m.name:30s} → {img_check.size}")
print()
print("  Verificacion completada. Todas las muestras son 224x224.")