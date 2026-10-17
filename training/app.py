import streamlit as st
import requests
import pickle
import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Load trained model and preprocessors
@st.cache_data
def load_models():
    with open('crop_yield_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('categorical_columns.pkl', 'rb') as f:
        cat_cols = pickle.load(f)
    with open('numerical_columns.pkl', 'rb') as f:
        num_cols = pickle.load(f)
    return model, label_encoders, scaler, feature_columns, cat_cols, num_cols

model, label_encoders, scaler, feature_columns, cat_cols, num_cols = load_models()

# API Keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# Initialize LangChain ChatGroq
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.7
    )

llm = initialize_llm()


# Create a safer, concise agricultural advisory prompt
prompt_template = ChatPromptTemplate.from_template("""
You are Kishan Mitra, a careful and farmer-friendly agricultural advisor
for Indian farmers.

Give practical advice in simple language. Keep the complete response short,
clear, and suitable for a farmer with limited technical knowledge.

FARM DATA:
- Location: {location}
- Crop: {crop}
- Area: {area} hectares
- Model-predicted yield: {predicted_yield} kg/ha
- Soil pH: {soil_ph}
- Temperature: {temperature}°C
- Humidity: {humidity}%
- Rainfall: {rainfall} mm
- Wind speed: {wind_speed} m/s
- Soil type: {soil_type}
- Previous crop: {previous_crop}

IMPORTANT RULES:
1. Treat the predicted yield as an estimate, not a guaranteed actual yield.
2. Never promise a 10% yield increase. Explain that results depend on
   weather, crop variety, soil conditions, pests, and farm management.
3. Do not invent fertilizer doses, local prices, water requirements,
   yield gains, or ROI figures.
4. Recommend exact fertilizer quantities only when supported by reliable,
   crop-specific and region-appropriate guidance. Otherwise, advise a soil
   test and consultation with a local Krishi Vigyan Kendra (KVK) or
   agriculture officer before deciding the dose.
5. Do not recommend pesticides, growth regulators, or chemical treatments
   without a clear need and reliable crop-specific guidance.
6. If any weather or farm information is missing, say it is unavailable.
   Never assume missing values are zero.
7. Consider the stated crop, location, soil type, irrigation method,
   and previous crop when making recommendations.
8. Clearly distinguish general suggestions from verified recommendations.
9. Do not fabricate sources, market prices, or financial returns.
10. If a recommendation depends on the crop's growth stage, ask the farmer
    for that stage or explain that it is needed.

RESPONSE FORMAT:

### 🌱 Farm Summary
Briefly summarize the crop and predicted yield. State that the prediction
is an estimate.

### ✅ Top 3 Actions
Give the three most useful practical actions for this farm.
Include timing where known. Avoid unsupported exact doses.

### 💧 Water and Soil
Give brief irrigation and soil-management advice suited to the available
farm information.

### 💰 Cost and Yield
Mention potential costs or yield benefits only when reliable figures are
available. Otherwise, explain what information is needed to estimate them.
Never claim a guaranteed 10% increase or a definite ROI.

### ⚠️ Important Note
Briefly mention any essential soil test, local agricultural advice, or
missing information needed before applying fertilizers or chemicals.

Use simple language. Avoid unnecessarily long tables. Do not repeat the
same recommendation in multiple sections.
""")


def get_coordinates(district_name, state_name=None):
    """Get latitude and longitude for a given district"""
    if state_name:
        location = f"{district_name}, {state_name}, India"
    else:
        location = f"{district_name}, India"
    
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
    headers = {'User-Agent': 'CropYieldPredictionApp/1.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        else:
            raise Exception("District not found")
    except Exception as e:
        st.error(f"Error fetching coordinates: {e}")
        return None, None


def get_weather(state, district):
    """Fetch current weather data for a district and state."""
    if not WEATHER_API_KEY:
        st.warning("Weather service is not configured.")
        return None

    location = f"{district}, {state}, India"
    url = "https://api.weatherapi.com/v1/current.json"

    try:
        response = requests.get(
            url,
            params={
                "key": WEATHER_API_KEY,
                "q": location
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # WeatherAPI may return an API error even with an HTTP response.
        if data.get("error"):
            st.warning("Weather data is unavailable for this location.")
            return None

        current = data.get("current")
        if not current:
            st.warning("Weather data is currently unavailable.")
            return None

        return {
            "temperature": current["temp_c"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_kph"] / 3.6,
            "rainfall": current.get("precip_mm", 0.0)
        }

    except requests.exceptions.Timeout:
        st.warning("The weather service took too long to respond.")
        return None

    except requests.exceptions.RequestException:
        st.warning("Unable to connect to the weather service.")
        return None

    except (ValueError, KeyError, TypeError):
        st.warning("The weather service returned unexpected data.")
        return None


def predict_yield(state_name, district_name, crop, area_ha, weather_data, soil_ph):
    """Predict crop yield"""
    new_data_dict = {
        "State Name": state_name,
        "Dist Name": district_name,
        "Year": 2025,
        "Crop": crop,
        "Area_ha": area_ha,
        "Temperature_C": weather_data["temperature"],
        "Humidity_%": weather_data["humidity"],
        "Rainfall_mm": weather_data["rainfall"],
        "Wind_Speed_m_s": weather_data["wind_speed"],
        "pH": soil_ph
    }
    new_df = pd.DataFrame([new_data_dict])

    # Apply label encoding
    for col in cat_cols:
        if col in new_df.columns:
            new_df[col] = label_encoders[col].transform(new_df[col])

    # Scale numerical values
    new_df[num_cols] = scaler.transform(new_df[num_cols])

    return model.predict(new_df[feature_columns])[0]

def get_ai_advice( prediction_data):
    """Get AI advice using LangChain"""
    try:
        # Create the chain
        chain = prompt_template | llm
        
        # Get response
        response = chain.invoke({
            "location": f"{prediction_data['district']}, {prediction_data['state']}",
            "crop": prediction_data['crop'],
            "area": prediction_data['area'],
            "predicted_yield": f"{prediction_data['yield_pred']:.2f}",
            "soil_ph": f"{prediction_data['soil_ph']:.1f}",
            "temperature": prediction_data['weather']['temperature'],
            "humidity": prediction_data['weather']['humidity'],
            "rainfall": prediction_data['weather']['rainfall'],
            "wind_speed": f"{prediction_data['weather']['wind_speed']:.2f}",
            "soil_type": prediction_data.get('soil_type', 'Not specified'),
            "irrigation": prediction_data.get('irrigation', 'Not specified'),
            "previous_crop": prediction_data.get('previous_crop', 'Not specified'),
            
        })
        
        return response.content
    except Exception as e:
        return f"Error getting AI response: {e}"

# ------------------- Streamlit UI --------------------
st.title("🌾 Kishan Mitra")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "💬 AI Advisor"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location")
        state = st.text_input("State", placeholder="e.g., Punjab")
        district = st.text_input("District", placeholder="e.g., Ludhiana")

    with col2:
        st.subheader("🌱 Crop Info")
        crop = st.selectbox("Crop", ["rice", "cotton", "maize", "chickpea"])
        area = st.number_input("Area (hectares)", min_value=0.1, value=1.0)

    st.subheader("🌱 Additional Info")
    col3, col4 = st.columns(2)
    
    with col3:
        soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=11.0, value=6.5, step=0.1)
        soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Loamy", "Silt", "Not specified"])
    
    with col4:
        irrigation = st.selectbox("Irrigation", ["Drip", "Sprinkler", "Flood", "Rainfed", "Not specified"])
        previous_crop = st.text_input("Previous Crop", placeholder="Optional")

    if st.button("🔮 Predict Yield", type="primary"):
        if state and district and crop and area > 0:
            with st.spinner("Getting prediction..."):
                lat, lon = get_coordinates(district, state)
                
                if lat and lon:
                    weather_data = get_weather(state, district)
                    
                    if weather_data:
                        try:
                            yield_pred = predict_yield(state, district, crop, area, weather_data, soil_ph)
                            
                            # Store data for AI advisor
                            st.session_state.prediction_data = {
                                'state': state,
                                'district': district,
                                'crop': crop,
                                'area': area,
                                'soil_ph': soil_ph,
                                'soil_type': soil_type,
                                'irrigation': irrigation,
                                'previous_crop': previous_crop if previous_crop else "Not specified",
                                'weather': weather_data,
                                'yield_pred': yield_pred,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Display results
                            st.success("✅ Prediction Complete!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🌾 Predicted Yield", f"{yield_pred:.2f} kg/ha")
                            with col2:
                                st.metric("📦 Total Production", f"{yield_pred * area:.2f} kg")
                            
                            with st.expander("🌦️ Weather Data"):
                                st.write(f"🌡️ Temperature: {weather_data['temperature']}°C")
                                st.write(f"💧 Humidity: {weather_data['humidity']}%")
                                st.write(f"🌧️ Rainfall: {weather_data['rainfall']} mm")
                                st.write(f"💨 Wind Speed: {weather_data['wind_speed']:.2f} m/s")
                            
                            st.info("💬 Now you can use the AI Advisor tab for farming recommendations!")
                            
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                    else:
                        st.error("Could not fetch weather data")
                else:
                    st.error("Location not found")
        else:
            st.warning("Please fill all required fields")

with tab2:
    st.subheader("💬 AI Agricultural Advisor")
    
    if st.session_state.prediction_data is None:
        st.info("Please make a prediction first in the Prediction tab")
    else:
        # Show current data summary
        data = st.session_state.prediction_data
        st.write(f"**Current Analysis:** {data['crop']} in {data['district']}, {data['state']}")
        st.write(f"**Predicted Yield:** {data['yield_pred']:.2f} kg/ha | **Soil pH:** {data['soil_ph']:.1f}")
        
        st.markdown("---")
        
        # Quick question buttons

        
        
    response = get_ai_advice(st.session_state.prediction_data)
    st.markdown(f"**🤖 AI Advisor:**\n\n{response}")
        
st.markdown("---")
        
       
        

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 AI Crop Yield Prediction with LangChain Advisory System</p>
</div>
""", unsafe_allow_html=True)