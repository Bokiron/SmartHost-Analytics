**Autor:** David Menéndez Rodríguez  
**Institución:** IES Azarquiel (Curso de Especialización en Inteligencia Artificial y Big Data)  
**Profesor/Tutor:** Sebastián Rubio Valero  
**Fecha:** Mayo 2026  

---

## Resumen del Proyecto y Propuesta de Valor

**SmartHost Analytics** es una plataforma de Inteligencia Artificial diseñada específicamente para inversores del mercado de apartamentos turísticos en la ciudad de Málaga. Los modelos tradicionales de valoración automatizada (AVMs) sufren de **"ceguera estética"**, calculando precios basándose exclusivamente en parámetros numéricos estáticos (metros cuadrados, habitaciones, coordenadas). 

Este sistema implementa una **arquitectura de Redes Neuronales Multimodales (Late Fusion)** que no solo procesa las métricas lógicas del inmueble, sino que evalúa el impacto del "atractivo visual" de su fotografía de portada. Gracias a esto, la plataforma actúa como un **Sistema de Soporte a Decisiones Financieras**, permitiendo a los inversores simular proyectos de interiorismo o reformas estéticas y cuantificar su **Retorno de Inversión (ROI)** de forma analítica antes de gastar un solo euro en la obra física.

---

## Arquitectura del Sistema

El sistema opera mediante dos flujos paralelos de extracción que convergen en una etapa de decisión final:

                  ┌───────────────────────────┐
                  │ Foto de Portada (224x224) │
                  └─────────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   ResNet34 (CNN)    │
                     └──────────┬──────────┘
                                │ (Vector Estético)
                                ▼ [512 dim]


┌──────────────────┐     ┌──────────┴──────────┐     ┌──────────────────┐
│ Características  │────>│Capa de Concatenación│────>│ Red Densa (MLP)  │────> [Precio Predicho €]
│ Tabulares (35)   │     └─────────────────────┘     │  (256->128->32)  │
└──────────────────┘            [547 dim]            └──────────────────┘


1. **Rama Visual (CNN - Computer Vision):** Utiliza *Transfer Learning* basado en la arquitectura **ResNet34** pre-entrenada en ImageNet. Se le eliminó la cabeza de clasificación original para extraer un vector de características latentes (*embeddings*) de **512 dimensiones** que codifican los niveles de luminosidad, modernidad del mobiliario y calidad estética.
2. **Rama Tabular (DNN - Deep Neural Network):** Una red Perceptrón Multicapa (MLP) que procesa 35 variables numéricas y categóricas optimizadas mediante ingeniería de características, tales como distancias geodésicas (*Haversine*) a la playa y al centro histórico, el índice de hacinamiento por estancia y puntuaciones de equipamiento (*amenities_score*).
3. **Fusión Multimodal (Late Fusion):** Ambos vectores se concatenan en una capa intermedia de **547 conexiones** que alimenta a bloques densos finales regulados con `BatchNorm1d` y `Dropout` para predecir el precio objetivo por noche mediante una activación lineal.

---

## Resultados y Métricas del Modelo

Durante la fase de experimentación y optimización científica empleando una GPU **NVIDIA GeForce RTX 4070**, se comparó el rendimiento del modelo en sus diferentes iteraciones:

| Configuración del Modelo | Val MAE | Train MAE | Gap (Overfitting) | R² (Coef. Determinación) | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tasador Base Tabular Puro (Baseline)** | 24.84€ | - | - | 0.570 | Servía de benchmark a ciegas |
| **Multimodal Original (D=0.3/0.2, WD=0)** | **21.47€** | **13.48€** | **7.99€** | **0.692** | **Modelo Ganador (Mejor precisión)** |
| Multimodal Opt 1+2 (D=0.5/0.4, WD=1e-3) | 22.56€ | 18.63€ | 3.93€ | 0.672 | Mejor gap, pero peor MAE real |
| Multimodal Opt 3 (Arch. Pequeña, WD=1e-3) | 22.73€ | 17.81€ | 4.92€ | 0.683 | Comportamiento equilibrado |

**Conclusión Científica:** La inclusión de la rama visual de ResNet34 redujo el error absoluto medio en **más de 3.30€ por noche** y aumentó el coeficiente de determinación hasta un **R² = 0.692**. Esto demuestra matemáticamente que la estética del apartamento es capaz de explicar de forma directa cerca del 10% de la varianza del precio de mercado vacacional en Málaga.

---

## Estructura del Repositorio

El proyecto sigue una organización limpia y desacoplada siguiendo los estándares de producción de la industria de software:

```text
SmartHostAnalytics/
├── .gitignore               # Exclusión de archivos binarios pesados e imágenes
├── README.md                # Portada y documentación del repositorio
├── requirements.txt         # Listado de dependencias congeladas de producción
│
├── data/                    # Almacenamiento local de datos (Ignorado en Git)
│   ├── listingV5.csv        # Dataset enriquecido tras la limpieza y EDA
│   └── Front_Images_224/    # Banco de imágenes estandarizadas tras el procesamiento
│
├── notebooks/               # Cuadernos experimentales de Jupyter
│   ├── 01_EDA_y_Limpieza.ipynb
│   ├── 02_EDA.ipynb         # Análisis exploratorio y pruebas estadísticas ANOVA
│   ├── 03_ModeloTabular.ipynb    # Entrenamiento del Cerebro Tabular Base
│   └── 04_03_ModeloCNN_ResNet34.ipynb  # Extracción de Embeddings con ResNet34
│
├── src/                     # Scripts autónomos de automatización de datos
│   └── script_resize_images_224.py  # Pipeline de redimensionado geométrico y CenterCrop
│
├── backend/                 # Microservicio de Inferencia (API RESTful)
│   ├── main.py              # Entry point del servidor web FastAPI
│   ├── core/
│   │   └── loader.py        # Lifespan handler de carga asíncrona de pesos .pt y .pkl
│   ├── nn_models/
│   │   └── networks.py      # Declaración de clases nn.Module de PyTorch
│   └── routers/
│       └── predict.py       # Endpoint POST /predict con lógica Multipart/Form-data
│
└── app/                     # Interfaz Gráfica del Usuario (Frontend)
    ├── app.py               # Archivo principal de renderizado de Streamlit multipágina
    ├── config.py            # Constantes de conexión HTTP hacia el backend
    └── pages/
        └── 1_Simulador_ROI.py # Panel de carga de fotografías y cálculo financiero

```

---

## Guía de Instalación y Configuración

### 1. Clonar el repositorio y preparar el entorno

Abre una terminal en tu entorno local y ejecuta:

```bash
git clone [https://github.com/Bokiron/SmartHostAnalytics.git](https://github.com/Bokiron/SmartHost-Analytics)
cd SmartHostAnalytics

```

Crea e instala un entorno virtual aislado (ejemplo con `venv` de Python):

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

```

### 2. Ejecución del Preprocesamiento de Imágenes (One-Shot)

Si dispones del dataset de imágenes en bruto dentro de `data/Front_Images/`, ejecuta el script de estandarización geométrica antes de iniciar los servicios:

```bash
python src/script_resize_images_224.py

```

*Este script aplicará un escalado proporcional y un CenterCrop automático de 224x224 píxeles, depurando imágenes corruptas.*

---

## Despliegue en Entorno Local

El sistema requiere el arranque independiente de sus dos capas para mantener la arquitectura cliente-servidor desacoplada:

### Paso A: Arrancar el Servidor Backend (FastAPI)

Navega a la carpeta SmartHost-Analytics e inicia el servidor Uvicorn:

```bash
uvicorn app.backend.main:app --reload --port 8000
```

*Una vez iniciado, puedes consultar la documentación interactiva autogenerada del endpoint en `http://127.0.0.1:8000/docs` (Swagger UI).*

### Paso B: Arrancar la Interfaz de Usuario (Streamlit)

Abre otra terminal diferente, activa el entorno virtual y ejecuta el frontend:

```bash
streamlit run app/frontend/app.py

```

*La aplicación web se abrirá automáticamente en tu navegador predeterminado bajo la dirección `http://localhost:8501`.*

---

## Guía de Uso de la Aplicación

1. **Configuración del Inmueble:** En la barra lateral izquierda del frontend, introduce las especificaciones de la vivienda (Barrio oficial de Málaga, tipología de cuarto, capacidad, baños, camas y activa las casillas de las amenities disponibles).
2. **Estado Actual:** Sube la fotografía de portada actual del inmueble. El sistema llamará asíncronamente al backend y mostrará en pantalla la tasación multimodal en base a su estética actual, proyectando la ocupación media y los ingresos anuales esperados.
3. **Simulador de Reforma (ROI):** Define el presupuesto estimado para un proyecto de interiorismo (ej. 3.500€) y sube una fotografía o render de inspiración de alta calidad lumínica y estética (estilo catálogo). La Inteligencia Artificial recalculará instantáneamente la nueva tasación, reflejando cuántos meses tardará el inversor en amortizar los costes de la reforma en base al incremento de ingresos netos por noche.

---

## Limitaciones y Advertencias Técnicas

* **Carácter Orientativo:** Este sistema está diseñado como una herramienta consultiva de soporte. Debido a la naturaleza no lineal de los espacios latentes en PyTorch, micro-cambios aislados en las amenities pueden no reflejar variaciones mecánicas inmediatas. El modelo evalúa el inmueble de forma holística.
* **Price Cap:** El modelo ha sido entrenado restringiendo el techo de mercado a **500€/noche** para mitigar la distorsión de gradientes provocada por valores atípicos, por lo que no es apto para tasar propiedades vacacionales de ultra-lujo.

