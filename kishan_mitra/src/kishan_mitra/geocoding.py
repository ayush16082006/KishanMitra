
import logging

import requests


logger = logging.getLogger("kishan_mitra.geocoding")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Identify the application when contacting Nominatim.
USER_AGENT = "KishanMitra/1.0 (agricultural crop yield assistance)"


def get_coordinates(district, state, country="India"):
    """
    Return latitude and longitude for a district and state.

    Returns:
        dict: {"latitude": float, "longitude": float}

    Raises:
        ValueError: If the location is empty or no result is found.
        requests.RequestException: If the request fails.
    """
    district = str(district).strip()
    state = str(state).strip()
    country = str(country).strip()

    if not district or not state or not country:
        raise ValueError(
            "District, state, and country must not be empty."
        )

    location = f"{district}, {state}, {country}"

    params = {
        "q": location,
        "format": "jsonv2",
        "limit": 1,
    }

    headers = {
        "User-Agent": USER_AGENT,
    }

    logger.info("Geocoding location: %s", location)

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()
        results = response.json()

    except requests.Timeout:
        logger.exception("Geocoding request timed out.")
        raise

    except requests.RequestException:
        logger.exception("Geocoding network request failed.")
        raise

    except ValueError:
        logger.exception("Geocoding service returned invalid JSON.")
        raise

    if not isinstance(results, list) or not results:
        logger.warning("No geocoding result found for %s.", location)
        raise ValueError(
            f"Could not find coordinates for {location}."
        )

    try:
        latitude = float(results[0]["lat"])
        longitude = float(results[0]["lon"])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        logger.exception("Geocoding response has invalid coordinates.")
        raise ValueError(
            "The geocoding service returned invalid coordinates."
        ) from error

    if (
        not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        logger.error("Geocoding returned out-of-range coordinates.")
        raise ValueError(
            "The geocoding service returned out-of-range coordinates."
        )

    logger.info(
        "Geocoding successful for %s.",
        location,
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
    }
