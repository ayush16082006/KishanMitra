
import logging
import math


logger = logging.getLogger("kishan_mitra.soil")


def analyze_soil_ph(soil_ph):
    """
    Validate soil pH and return a basic interpretation.

    This function does not diagnose soil fertility or calculate
    fertilizer requirements.
    """
    try:
        soil_ph = float(soil_ph)
    except (TypeError, ValueError) as error:
        raise ValueError("Soil pH must be a numeric value.") from error

    if not math.isfinite(soil_ph) or not 0 <= soil_ph <= 14:
        raise ValueError("Soil pH must be between 0 and 14.")

    if soil_ph < 6.5:
        category = "Acidic"
        interpretation = (
            "The measured pH is acidic. Crop suitability and any "
            "need for soil amendments depend on the crop and a soil test."
        )
    elif soil_ph <= 7.5:
        category = "Near neutral"
        interpretation = (
            "The measured pH is near neutral. This alone does not "
            "confirm that nutrients are sufficient for the crop."
        )
    else:
        category = "Alkaline"
        interpretation = (
            "The measured pH is alkaline. Crop suitability and "
            "nutrient availability depend on the crop and soil conditions."
        )

    logger.info("Soil pH analysis completed; category=%s.", category)

    return {
        "soil_ph": soil_ph,
        "category": category,
        "interpretation": interpretation,
        "fertility_assessed": False,
    }
