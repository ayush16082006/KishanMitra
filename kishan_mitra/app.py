
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from kishan_mitra.advisor import get_advisor_response
from kishan_mitra.config import DATASET_PATH
from kishan_mitra.geocoding import get_coordinates
from kishan_mitra.logging_config import setup_logging
from kishan_mitra.prediction import predict_one
from kishan_mitra.soil import analyze_soil_ph
from kishan_mitra.weather import get_weather


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

logger = setup_logging()

st.set_page_config(
    page_title="Kishan Mitra",
    page_icon="🌾",
    layout="wide",
)

logger.info("Kishan Mitra application started.")


# ============================================================
# 2. SESSION STATE
# ============================================================

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "advisor_response" not in st.session_state:
    st.session_state.advisor_response = None


# ============================================================
# 3. LOAD LOCATION AND CROP DATA
# ============================================================

@st.cache_data
def load_location_data():
    """Load valid states, districts, and crops from the dataset."""

    logger.info("Loading location and crop data.")

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    data = pd.read_csv(
        DATASET_PATH,
        usecols=["State Name", "Dist Name", "Crop"],
    )

    for column in ["State Name", "Dist Name", "Crop"]:
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(
        subset=["State Name", "Dist Name", "Crop"]
    )

    for column in ["State Name", "Dist Name", "Crop"]:
        data = data[data[column] != ""]

    if data.empty:
        raise ValueError(
            "The dataset contains no valid location or crop records."
        )

    logger.info(
        "Location and crop data loaded: %s records.",
        len(data),
    )

    return data


# ============================================================
# 4. WEATHER NORMALIZATION
# ============================================================

def normalize_weather_data(raw_weather):
    """
    Convert the weather module's response into the field names
    used by this application and the AI Advisor.
    """

    if not isinstance(raw_weather, dict):
        raise ValueError("Weather data is unavailable.")

    # Support the current modular weather.py response format.
    # Also accept the names used by the original application.
    aliases = {
        "temperature": ("temperature_c", "temperature"),
        "humidity": ("humidity_pct", "humidity"),
        "wind_speed": ("wind_speed_m_s", "wind_speed"),
        "rainfall": ("precipitation_mm", "rainfall"),
    }

    normalized = {}

    for target, possible_keys in aliases.items():
        value = None

        for key in possible_keys:
            if raw_weather.get(key) is not None:
                value = raw_weather[key]
                break

        if value is None:
            raise ValueError(
                f"Weather data is missing the field: {target}"
            )

        normalized[target] = float(value)

    return normalized


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("🌾 Kishan Mitra")

st.markdown(
    """
    Estimate crop yield using machine learning, review soil and
    weather conditions, and request practical farming guidance
    from the AI Advisor.
    """
)

st.divider()


# ============================================================
# 6. LOAD DATA SAFELY
# ============================================================

try:
    location_data = load_location_data()

except Exception:
    logger.exception("Failed to load location and crop data.")

    st.error(
        "The agricultural dataset could not be loaded. "
        "Please check the application configuration and logs."
    )
    st.stop()


# ============================================================
# 7. SIDEBAR MODEL SELECTION
# ============================================================

st.sidebar.header("⚙️ Model Settings")

model_options = {
    "Extra Trees (Default)": "extra_trees_candidate",
    "Random Forest (Comparison)": "random_forest",
}

selected_model_label = st.sidebar.selectbox(
    "Prediction model",
    options=list(model_options.keys()),
    index=0,
)

selected_model = model_options[selected_model_label]

if selected_model == "extra_trees_candidate":
    st.sidebar.info(
        "Using Extra Trees, the preferred model based on the "
        "recorded evaluation results."
    )
else:
    st.sidebar.info(
        "Using Random Forest for comparison."
    )

st.sidebar.caption(
    "Predictions are estimates and are not guarantees of actual harvest."
)


# ============================================================
# 8. APPLICATION TABS
# ============================================================

prediction_tab, advisor_tab = st.tabs(
    ["🔮 Prediction", "💬 AI Advisor"]
)


# ============================================================
# 9. PREDICTION TAB
# ============================================================

with prediction_tab:

    st.header("1. Location and Crop")

    states = sorted(
        location_data["State Name"].unique().tolist()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox(
            "State",
            options=states,
            key="farm_state",
        )

    district_data = location_data[
        location_data["State Name"] == state
    ]

    districts = sorted(
        district_data["Dist Name"].unique().tolist()
    )

    with col2:
        district = st.selectbox(
            "District",
            options=districts,
            key="farm_district",
        )

    crop_data = district_data[
        district_data["Dist Name"] == district
    ]

    crops = sorted(
        crop_data["Crop"].unique().tolist()
    )

    with col3:
        crop = st.selectbox(
            "Crop",
            options=crops,
            key="farm_crop",
        )

    st.header("2. Farm Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Farm area (hectares)",
            min_value=0.1,
            max_value=100000.0,
            value=1.0,
            step=0.1,
        )

        soil_type = st.selectbox(
            "Soil type",
            [
                "Not specified",
                "Sandy",
                "Clay",
                "Loamy",
                "Silt",
                "Black soil",
                "Alluvial soil",
                "Other",
            ],
        )

    with col2:
        irrigation = st.selectbox(
            "Irrigation method",
            [
                "Not specified",
                "Drip",
                "Sprinkler",
                "Flood",
                "Rainfed",
                "Other",
            ],
        )

        previous_crop = st.text_input(
            "Previous crop (optional)",
            placeholder="e.g., wheat",
        )

    st.header("3. Soil and Weather Conditions")

    weather_mode = st.radio(
        "Weather data source",
        [
            "Fetch current weather",
            "Enter weather manually",
        ],
        horizontal=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input(
            "Prediction year",
            min_value=1966,
            max_value=2100,
            value=datetime.now().year,
            step=1,
        )

        soil_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            help="Use a measured soil pH value when available.",
        )

    with col2:
        if weather_mode == "Enter weather manually":
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=28.0,
                step=0.5,
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
            )
        else:
            st.info(
                "Temperature and humidity will be obtained "
                "from the current weather service."
            )

    with col3:
        if weather_mode == "Enter weather manually":
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=5.0,
            )

            wind_speed = st.number_input(
                "Wind speed (m/s)",
                min_value=0.0,
                max_value=150.0,
                value=2.0,
                step=0.5,
            )
        else:
            st.info(
                "Rainfall and wind speed will be obtained "
                "from the current weather service."
            )

    solar_radiation = st.number_input(
        "Solar radiation (MJ/m²/day)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=0.5,
        help=(
            "Enter a measured or reliable local value when available. "
            "The current weather service does not necessarily provide it."
        ),
    )

    if int(year) > 2017:
        st.warning(
            "The training dataset ends in 2017. A prediction for "
            "a later year is outside the model's historical training "
            "period, and its reliability has not been established."
        )

    st.divider()

    predict_clicked = st.button(
        "🔮 Predict Crop Yield",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        # --------------------------------------------
        # A. Validate soil pH
        # --------------------------------------------

        try:
            soil_analysis = analyze_soil_ph(float(soil_ph))

        except (ValueError, TypeError) as exc:
            st.error(f"Please check the soil pH value: {exc}")
            st.stop()

        # --------------------------------------------
        # B. Get weather data
        # --------------------------------------------

        weather_data = None

        if weather_mode == "Fetch current weather":

            try:
                with st.spinner("Fetching current weather..."):

                    # Get geographic coordinates for the selected location.
                    coordinates = get_coordinates(
                        district,
                        state,
                    )

                    latitude = coordinates["latitude"]
                    longitude = coordinates["longitude"]

                    # Fetch current weather using numeric coordinates.
                    raw_weather = get_weather(
                        latitude,
                        longitude,
                    )

                    weather_data = normalize_weather_data(
                        raw_weather
                    )

            except Exception:
                logger.exception(
                    "Current weather retrieval failed | state=%s | "
                    "district=%s",
                    state,
                    district,
                )

                st.error(
                    "Current weather could not be retrieved. "
                    "Switch to 'Enter weather manually' and enter "
                    "reliable weather values to continue."
                )
                st.stop()

        else:
            weather_data = {
                "temperature": float(temperature),
                "humidity": float(humidity),
                "rainfall": float(rainfall),
                "wind_speed": float(wind_speed),
            }
        # --------------------------------------------
        # C. Build the model input
        # --------------------------------------------

        input_data = {
            "State Name": state,
            "Dist Name": district,
            "Crop": crop,
            "Year": int(year),
            "Temperature_C": weather_data["temperature"],
            "Humidity_%": weather_data["humidity"],
            "pH": float(soil_ph),
            "Rainfall_mm": weather_data["rainfall"],
            "Wind_Speed_m_s": weather_data["wind_speed"],
            "Solar_Radiation_MJ_m2_day": float(solar_radiation),
        }

        logger.info(
            "Prediction requested | model=%s | crop=%s | "
            "state=%s | district=%s | year=%s",
            selected_model,
            crop,
            state,
            district,
            int(year),
        )

        # --------------------------------------------
        # D. Run prediction
        # --------------------------------------------

        try:
            with st.spinner("Predicting crop yield..."):

                predicted_yield = float(
                    predict_one(
                        input_data,
                        model_name=selected_model,
                    )
                )

            if not pd.notna(predicted_yield):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            if predicted_yield < 0:
                logger.warning(
                    "The model returned a negative yield estimate: %s",
                    predicted_yield,
                )

            total_production = predicted_yield * float(area)

            # Save all relevant data for the AI Advisor.
            st.session_state.prediction_data = {
                "state": state,
                "district": district,
                "crop": crop,
                "area": float(area),
                "year": int(year),
                "soil_ph": float(soil_ph),
                "soil_type": soil_type,
                "irrigation": irrigation,
                "previous_crop": previous_crop.strip()
                if previous_crop.strip()
                else None,
                "weather": weather_data,
                "solar_radiation": float(solar_radiation),
                "yield_pred": predicted_yield,
                "total_production": total_production,
                "model_name": selected_model,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            # Clear advice for the previous prediction.
            st.session_state.advisor_response = None

            logger.info(
                "Prediction successful | model=%s | crop=%s | "
                "state=%s | district=%s",
                selected_model,
                crop,
                state,
                district,
            )

        except Exception:
            logger.exception(
                "Prediction failed | model=%s | crop=%s | "
                "state=%s | district=%s",
                selected_model,
                crop,
                state,
                district,
            )

            st.error(
                "Prediction failed. Check the model artifacts, "
                "input configuration, and application logs."
            )

    # ------------------------------------------------
    # E. Display saved prediction result
    # ------------------------------------------------

    prediction = st.session_state.prediction_data

    if prediction is not None:

        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated yield",
                f"{prediction['yield_pred']:,.2f} kg/ha",
            )

        with col2:
            st.metric(
                "Estimated total production",
                f"{prediction['total_production']:,.2f} kg",
            )

        st.caption(
            f"{prediction['crop']} · {prediction['district']}, "
            f"{prediction['state']} · "
            f"{prediction['area']:,.2f} hectares"
        )

        st.caption(
            f"Model: {prediction['model_name']} · "
            f"Prediction year: {prediction['year']} · "
            f"Generated: {prediction['timestamp']}"
        )

        st.info(
            "This is a model-based estimate, not a guaranteed harvest. "
            "Total production is calculated by multiplying estimated "
            "yield per hectare by the entered farm area."
        )

        if prediction["year"] > 2017:
            st.warning(
                "This prediction is for a year beyond the historical "
                "training data. Treat the result cautiously."
            )

        with st.expander("🌦️ Weather details"):
            weather = prediction["weather"]

            st.write(
                f"**Temperature:** {weather['temperature']:.1f} °C"
            )
            st.write(
                f"**Humidity:** {weather['humidity']:.1f}%"
            )
            st.write(
                f"**Rainfall:** {weather['rainfall']:.2f} mm"
            )
            st.write(
                f"**Wind speed:** {weather['wind_speed']:.2f} m/s"
            )
            st.write(
                f"**Solar radiation entered:** "
                f"{prediction['solar_radiation']:.2f} MJ/m²/day"
            )

        with st.expander("🌱 Soil analysis"):
            st.write(f"**Measured pH:** {prediction['soil_ph']:.1f}")
            st.write(f"**Soil type:** {prediction['soil_type']}")

            # Display the existing module's result without
            # assuming a particular return structure.
            saved_prediction = st.session_state.get("prediction_data")

            if saved_prediction:
                soil_analysis = analyze_soil_ph(
                float(saved_prediction["soil_ph"])
                )
                st.write(soil_analysis)
            else:
                st.info("Run a prediction first to view soil pH analysis.")

        st.caption(
            "Soil pH alone does not establish overall soil fertility. "
"Use a soil test for nutrient-specific decisions."
        )

        st.success(
            "Prediction saved for the AI Advisor. Open the AI Advisor "
            "tab when you are ready to request farming guidance."
        )


# ============================================================
# 10. AI ADVISOR TAB
# ============================================================

with advisor_tab:

    st.header("💬 AI Agricultural Advisor")

    prediction = st.session_state.prediction_data

    if prediction is None:

        st.info(
            "Make a crop-yield prediction in the Prediction tab first. "
            "The AI Advisor uses that prediction and its associated "
            "farm, soil, and weather information."
        )

    else:

        st.subheader("Current farm analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Crop",
                prediction["crop"].title(),
            )

        with col2:
            st.metric(
                "Estimated yield",
                f"{prediction['yield_pred']:,.2f} kg/ha",
            )

        with col3:
            st.metric(
                "Soil pH",
                f"{prediction['soil_ph']:.1f}",
            )

        st.write(
            f"**Location:** {prediction['district']}, "
            f"{prediction['state']}"
        )

        st.write(
            f"**Farm area:** {prediction['area']:.2f} hectares"
        )

        st.write(
            f"**Soil type:** {prediction['soil_type']}"
        )

        st.write(
            f"**Irrigation:** {prediction['irrigation']}"
        )

        st.write(
            f"**Previous crop:** "
            f"{prediction['previous_crop'] or 'Not specified'}"
        )

        st.divider()

        if st.button(
            "💬 Get AI Farming Advice",
            type="primary",
            use_container_width=True,
        ):

            weather = prediction["weather"]

            try:
                with st.spinner(
                    "The AI Advisor is preparing farming guidance..."
                ):

                    advice = get_advisor_response(
                        crop=prediction["crop"],
                        state=prediction["state"],
                        district=prediction["district"],
                        predicted_yield_kg_per_ha=prediction["yield_pred"],
                        soil_ph=prediction["soil_ph"],
                        temperature_c=weather["temperature"],
                        humidity_pct=weather["humidity"],
                        rainfall_mm=weather["rainfall"],
                        wind_speed_m_s=weather["wind_speed"],
                        solar_radiation_mj_m2_day=(
                            prediction["solar_radiation"]
                        ),
                        soil_type=prediction["soil_type"],
                        irrigation=prediction["irrigation"],
                        previous_crop=prediction["previous_crop"],
                    )

                if not isinstance(advice, str) or not advice.strip():
                    raise ValueError(
                        "The AI Advisor returned an empty response."
                    )

                st.session_state.advisor_response = advice

                logger.info(
                    "AI advice generated successfully | crop=%s | "
                    "state=%s | district=%s",
                    prediction["crop"],
                    prediction["state"],
                    prediction["district"],
                )

            except Exception:
                logger.exception(
                    "AI Advisor request failed | crop=%s | "
                    "state=%s | district=%s",
                    prediction["crop"],
                    prediction["state"],
                    prediction["district"],
                )

                st.error(
                    "The AI Advisor could not generate advice. "
                    "Check the application logs and AI service "
                    "configuration, then try again."
                )

        if st.session_state.advisor_response:

            st.subheader("🌱 Your farming guidance")

            st.markdown(
                st.session_state.advisor_response
            )

            st.caption(
                "AI-generated guidance is informational. Verify "
                "fertilizer, pesticide, and other high-impact decisions "
                "with a soil test and a local agricultural expert."
            )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌾 Kishan Mitra | Crop Yield Prediction and AI Farming Advisor"
)
