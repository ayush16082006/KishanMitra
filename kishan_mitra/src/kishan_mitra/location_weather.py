
import logging

from kishan_mitra.geocoding import get_coordinates
from kishan_mitra.weather import get_weather


logger = logging.getLogger("kishan_mitra.location_weather")


def get_location_weather(district, state):
    """
    Resolve a district and state, then retrieve current weather.

    Returns:
        dict containing coordinates and current weather measurements.
    """
    logger.info(
        "Starting location and weather lookup for district=%s, state=%s.",
        district,
        state,
    )

    coordinates = get_coordinates(district, state)

    weather = get_weather(
        coordinates["latitude"],
        coordinates["longitude"],
    )

    result = {
        "district": district,
        "state": state,
        **coordinates,
        **weather,
    }

    logger.info(
        "Location and weather lookup completed for district=%s, state=%s.",
        district,
        state,
    )

    return result
