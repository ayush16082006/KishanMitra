
import logging
import os

import requests
import streamlit as st
from dotenv import load_dotenv


logger = logging.getLogger("kishan_mitra.weather")

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"
REQUEST_TIMEOUT = 10


def get_weather(latitude, longitude):
    """
    Fetch current weather for geographic coordinates.

    Args:
        latitude: Geographic latitude.
        longitude: Geographic longitude.

    Returns:
        dict containing:
            temperature_c: Temperature in degrees Celsius.
            humidity_pct: Relative humidity in percent.
            wind_speed_m_s: Wind speed in metres per second.
            precipitation_mm: Current precipitation in millimetres.

    Raises:
        ValueError: If coordinates or API configuration are invalid.
        requests.RequestException: If the API request fails.
    """
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Latitude and longitude must be numeric."
        ) from error

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("WEATHER_API_KEY")
        except (FileNotFoundError, RuntimeError):
            api_key = None

    if not api_key:
        logger.error("WEATHER_API_KEY is not configured.")
        raise ValueError(
            "Weather API key is missing. Check the project .env file."
        )

    params = {
        "key": api_key,
        "q": f"{latitude},{longitude}",
    }

    logger.info("Requesting current weather for supplied coordinates.")

    try:
        response = requests.get(
            WEATHER_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.Timeout:
        logger.exception("Weather API request timed out.")
        raise

    except requests.RequestException:
        logger.exception("Weather API request failed.")
        raise

    except ValueError:
        logger.exception("Weather API returned invalid JSON.")
        raise

    try:
        current = payload["current"]

        temperature_c = float(current["temp_c"])
        humidity_pct = float(current["humidity"])
        wind_kph = float(current["wind_kph"])
        precipitation_mm = float(current["precip_mm"])

    except (KeyError, TypeError, ValueError) as error:
        # Do not log the complete response; it may contain sensitive data.
        logger.error("Weather API response is missing valid weather fields.")
        raise ValueError(
            "Weather API returned incomplete or invalid weather data."
        ) from error

    values = [
        temperature_c,
        humidity_pct,
        wind_kph,
        precipitation_mm,
    ]

    if not all(__import__("math").isfinite(value) for value in values):
        logger.error("Weather API returned non-finite numeric values.")
        raise ValueError(
            "Weather API returned invalid numeric weather values."
        )

    if not 0 <= humidity_pct <= 100:
        raise ValueError("Weather API returned invalid humidity.")

    if wind_kph < 0 or precipitation_mm < 0:
        raise ValueError(
            "Weather API returned a negative wind or precipitation value."
        )

    weather = {
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "wind_speed_m_s": wind_kph / 3.6,
        "precipitation_mm": precipitation_mm,
    }

    logger.info("Current weather retrieved successfully.")

    return weather
