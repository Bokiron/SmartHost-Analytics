# frontend/components/sidebar.py
import streamlit as st


def render_sidebar() -> tuple[dict, object, bool]:
    """Renderiza el sidebar y devuelve (form, foto, calcular)."""

    with st.sidebar:
        st.header(" Datos del apartamento")

        st.subheader(" Foto de portada")
        foto = st.file_uploader("Sube la foto actual del piso", type=["jpg", "jpeg", "png"])
        if foto:
            st.image(foto, caption="Foto actual", use_container_width=True)

        st.divider()

        st.subheader(" Datos del anfitrión")
        host_response_time = st.selectbox(
            "Tiempo de respuesta",
            ["within an hour", "within a few hours", "within a day", "a few days or more"],
        )
        host_response_rate = st.slider("Tasa de respuesta", 0.0, 1.0, 0.95, 0.01)
        host_is_superhost  = st.checkbox("¿Es Superhost?", value=True)

        st.divider()

        st.subheader(" Ubicación")
        distancia_centro = st.number_input("Distancia al centro (km)", 0.0, 20.0, 0.5, 0.1)
        distancia_playa  = st.number_input("Distancia a la playa (km)", 0.0, 20.0, 1.2, 0.1)
        longitude        = st.number_input("Longitud", -5.0, -4.0, -4.42, 0.001, format="%.3f")

        st.divider()

        st.subheader(" Características")
        room_type    = st.selectbox(
            "Tipo de alojamiento",
            ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        )
        accommodates = st.number_input("Capacidad (personas)", 1, 16, 4)
        bedrooms     = st.number_input("Habitaciones", 0, 10, 2)
        beds         = st.number_input("Camas", 0, 10, 2)
        bathrooms    = st.number_input("Baños", 0.0, 10.0, 1.0, 0.5)

        st.divider()

        st.subheader(" Servicios")

        st.markdown("<hr style='margin: 4px 0; border-color: transparent'>", unsafe_allow_html=True) # Variables individuales del modelo + score
        col_a, col_b = st.columns(2)
        with col_a:
            has_cooking_basics   = st.checkbox("🍳 Cocina básica",  value=True)
            has_tv               = st.checkbox("📺 TV",             value=True)
            has_air_conditioning = st.checkbox("❄️ Aire acond.",    value=True)
            has_washer           = st.checkbox("🫧 Lavadora",       value=True)
        with col_b:
            has_heating          = st.checkbox("🔥 Calefacción",    value=True)
            has_freezer          = st.checkbox("🧊 Congelador",     value=False)
            has_coffee_maker     = st.checkbox("☕ Cafetera",       value=True)
            has_balcony          = st.checkbox("🌿 Balcón/terraza", value=False)

        st.markdown("<hr style='margin: 4px 0; border-color: transparent'>", unsafe_allow_html=True) # Solo suman al amenities score
        col_c, col_d = st.columns(2)
        with col_c:
            has_kitchen    = st.checkbox("🍽️ Cocina completa", value=True)
            has_hair_dryer = st.checkbox("💨 Secador pelo",    value=True)
            has_iron       = st.checkbox("👔 Plancha",          value=True)
            has_bed_linens = st.checkbox("🛏️ Ropa de cama",   value=True)
        with col_d:
            has_microwave    = st.checkbox("📡 Microondas", value=True)
            has_refrigerator = st.checkbox("🧃 Nevera",     value=True)
            has_toaster      = st.checkbox("🍞 Tostadora",  value=False)
        
        st.markdown("<hr style='margin: 4px 0; border-color: transparent'>", unsafe_allow_html=True) # Básicos (se dan por supuestos, no afectan al modelo)
        col_e, col_f = st.columns(2)
        with col_e:
            st.checkbox("🍽️ Vajilla",        value=True,  disabled=True)
            st.checkbox("🪝 Perchas",         value=True,  disabled=True)
            st.checkbox("🚿 Agua caliente",   value=True,  disabled=True)
        with col_f:
            st.checkbox("🧴 Champú",          value=True,  disabled=True)
            st.checkbox("📶 WiFi",            value=True,  disabled=True)
            st.checkbox("🧻 Esenciales",      value=True,  disabled=True)

        st.markdown("<hr style='margin: 4px 0; border-color: transparent'>", unsafe_allow_html=True)
        private_bathroom = st.checkbox("Baño privado", value=True)

        st.divider()

        st.subheader("⭐ Valoraciones")
        reviews_per_month     = st.number_input("Reseñas por mes", 0.0, 20.0, 1.5, 0.1)
        number_of_reviews_ltm = st.number_input("Reseñas últimos 12 meses", 0, 200, 12)
        rating      = st.slider("Valoración global",         0.0, 5.0, 4.8, 0.1)
        accuracy    = st.slider("Precisión del anuncio",     0.0, 5.0, 4.7, 0.1)
        cleanliness = st.slider("Limpieza",                  0.0, 5.0, 4.9, 0.1)
        location    = st.slider("Ubicación",                 0.0, 5.0, 4.6, 0.1)
        value       = st.slider("Relación calidad/precio",   0.0, 5.0, 4.5, 0.1)

        st.divider()
        calcular = st.button("🔍 Calcular precio y ROI", use_container_width=True, type="primary")

    form = {
        "host_response_time": host_response_time, "host_response_rate": host_response_rate,
        "host_is_superhost": host_is_superhost, "distancia_centro": distancia_centro,
        "distancia_playa": distancia_playa, "longitude": longitude,
        "room_type": room_type, "accommodates": accommodates, "bedrooms": bedrooms,
        "beds": beds, "bathrooms": bathrooms,
        "has_cooking_basics": has_cooking_basics, "has_tv": has_tv,
        "has_air_conditioning": has_air_conditioning, "has_washer": has_washer,
        "has_heating": has_heating, "has_freezer": has_freezer,
        "has_coffee_maker": has_coffee_maker, "has_balcony_or_terrace": has_balcony,
        "has_kitchen": has_kitchen, "has_hair_dryer": has_hair_dryer,
        "has_iron": has_iron, "has_bed_linens": has_bed_linens,
        "has_microwave": has_microwave, "has_refrigerator": has_refrigerator,
        "has_toaster": has_toaster, "private_bathroom": private_bathroom,
        "reviews_per_month": reviews_per_month, "number_of_reviews_ltm": number_of_reviews_ltm,
        "rating": rating, "accuracy": accuracy, "cleanliness": cleanliness,
        "location": location, "value": value,
    }

    return form, foto, calcular