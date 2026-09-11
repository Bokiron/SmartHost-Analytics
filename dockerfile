# 1. Usar una imagen oficial de Python ligera (usaremos 3.12, que es la de tu entorno local)
FROM python:3.12-slim

# 2. Hugging Face exige crear un usuario "no root" por seguridad
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiamos el requirements.txt que limpiamos antes
COPY --chown=user:user requirements.txt .

# 5. Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamos todo tu código al contenedor
COPY --chown=user:user . .

# 7. Exponemos el puerto oficial de Hugging Face Spaces
EXPOSE 7860

# 8. Damos permisos de ejecución al script bash y lo ejecutamos
RUN chmod +x run.sh
CMD ["./run.sh"]