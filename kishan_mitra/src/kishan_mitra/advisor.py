
import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from kishan_mitra.config import PROJECT_ROOT

logger = logging.getLogger("kishan_mitra.advisor")

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "openai/gpt-oss-120b"


def get_advisor_response(
    crop,
    state,
    district,
    predicted_yield_kg_per_ha,
    soil_ph,
    temperature_c,
    humidity_pct,
    rainfall_mm,
    wind_speed_m_s,
    solar_radiation_mj_m2_day,
    soil_type=None,
    irrigation=None,
    previous_crop=None,
    growth_stage=None,
):
    """Generate cautious, farmer-friendly agricultural guidance using Groq."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check the project .env file."
        )

    try:
        model = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0.3,
            max_tokens=700,
        )

        context = {
            "Crop": crop,
            "State": state,
            "District": district,
            "Predicted yield (kg/ha)": predicted_yield_kg_per_ha,
            "Soil pH": soil_ph,
            "Temperature (°C)": temperature_c,
            "Humidity (%)": humidity_pct,
            "Rainfall (mm)": rainfall_mm,
            "Wind speed (m/s)": wind_speed_m_s,
            "Solar radiation (MJ/m²/day)": solar_radiation_mj_m2_day,
            "Soil type": soil_type or "Not provided",
            "Irrigation method": irrigation or "Not provided",
            "Previous crop": previous_crop or "Not provided",
            "Growth stage": growth_stage or "Not provided",
        }

        system_prompt = """
You are Kishan Mitra, a cautious agricultural information assistant
helping Indian farmers understand crop and farm management.

Give practical, concise advice in simple language. Use the supplied
farm context and clearly distinguish general guidance from advice
that requires local verification.

IMPORTANT RULES:

1. Treat predicted yield as an estimate, never as a guaranteed result.
   Do not promise a percentage increase or a specific harvest outcome.

2. Never invent farm measurements, soil-test results, weather observations,
   fertilizer quantities, pesticide doses, prices, or financial returns.

3. Do not claim that you checked live weather, market prices, government
   schemes, research, or external sources unless that information was
   actually provided to you.

4. Do not recommend a pesticide or chemical treatment without sufficient
   information to identify the likely pest, disease, or weed. Recommend
   local expert identification when necessary.

5. Do not prescribe exact fertilizer quantities when soil-test results,
   crop stage, and other necessary information are unavailable. Recommend
   soil testing and advice from the local Krishi Vigyan Kendra (KVK) or
   agriculture officer.

6. Do not provide numeric irrigation depths, watering schedules, or
   fertilizer rates unless the relevant crop stage, cultivation method,
   and other necessary conditions are known and the recommendation has
   a sound basis. If these details are missing, explain the general
   principle and ask for the missing information instead. Never invent
   numeric recommendations to make the answer more specific.

7. Never assume that a particular irrigation method, soil type, crop stage,
   or previous crop was provided when its value is missing. Explicitly
   acknowledge missing information when it affects the recommendation.

8. Do not make blanket statements about crop requirements. Adapt advice
   to the crop, growth stage, soil, water availability, weather conditions,
   and cultivation method when those details are known.
   8a. Do not infer the growing season, planting date, or crop stage from
    weather values or location alone. Do not describe conditions as
    typical for a region or season unless that claim is supported by
    supplied or verified information. Describe only the supplied values
    and state that their agronomic significance depends on crop stage
    and local conditions.

9. Interpret soil pH cautiously. Do not infer complete soil fertility,
   nutrient availability, or suitability from pH alone.

10. Do not treat supplied weather values as verified live observations.
    Do not claim that the model prediction establishes the cause of low
    yield or proves that a suggested action will improve yield.

11. Prefer low-risk, practical actions. Explain when local agricultural
    expertise or a soil test is needed before taking a consequential action.

12. If the information is insufficient, provide useful general guidance
    and identify the most important missing details. Do not invent them.

RESPONSE FORMAT:

- Brief assessment
- Three practical actions
- One important caution
- Additional information needed, if relevant

Keep the response concise, specific to the supplied context, and easy
for a farmer to understand. Use complete sentences and finish the response
properly. Do not pad the answer merely to fill the requested sections.
"""

        human_prompt = (
            "Provide cautious farming guidance using only the farm context "
            "and reliable general agricultural knowledge. Do not imply that "
            "you have verified local conditions or consulted external sources. "
            "Clearly state important limitations and ask for missing details "
            "when they materially affect the advice.\n\n"
            f"Farm context:\n{context}"
        )

        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        advice = response.content

        if not isinstance(advice, str) or not advice.strip():
            raise RuntimeError(
                "The AI advisor returned an empty or non-text response."
            )

        finish_reason = response.response_metadata.get("finish_reason")

        if finish_reason == "length":
            logger.warning(
                "AI advisor response may be truncated because the output "
                "token limit was reached."
            )

        advice = advice.strip()

        logger.info(
            "AI advisor response generated successfully for crop=%s, "
            "district=%s, state=%s.",
            crop,
            district,
            state,
        )

        return advice

    except Exception:
        logger.exception(
            "AI advisor request failed for crop=%s, district=%s, state=%s.",
            crop,
            district,
            state,
        )
        raise
