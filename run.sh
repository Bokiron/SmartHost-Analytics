#!/bin/bash

# 1. Arrancar el backend (FastAPI) en segundo plano usando el símbolo '&'
# IMPORTANTE: Reemplaza "app.main:app" por la ruta real a tu instancia de FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Esperar 3 segundos para asegurar que el backend de PyTorch se ha cargado en memoria
sleep 3

# 3. Arrancar el frontend (Streamlit) en primer plano en el puerto que exige Hugging Face
# IMPORTANTE: Reemplaza "frontend/app.py" por la ruta de tu archivo de Streamlit
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0