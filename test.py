import requests

def get_soil_ph(lat, lon):
    """
    Fetch soil pH (0–5 cm depth) from SoilGrids API for given lat/lon.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,  # Use input lon
        "lat": lat,  # Use input lat
        "property": "phh2o",
        "depth": "0-5cm",
        "value": "mean"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"SoilGrids API error: {response.status_code} - {response.text}")

        soil_data = response.json()
        print("API Response:", soil_data)  # Log response for debugging

        # Check if 'properties' and 'layers' exist
        if "properties" not in soil_data or not soil_data["properties"].get("layers"):
            raise Exception("No soil data found for the given coordinates.")

        # Extract soil pH at 0–5 cm
        ph_layer = soil_data["properties"]["layers"][0]
        if "depths" not in ph_layer or not ph_layer["depths"]:
            raise Exception("No depth data found in the response.")

        ph_value = ph_layer["depths"][0]["values"]["mean"]
        if ph_value is None:
            raise Exception("No mean pH value available for the given coordinates.")

        # SoilGrids pH values are often scaled (e.g., pH * 10), so divide by 10
        return ph_value / 10.0

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Error parsing API response: {str(e)}")

if __name__ == "__main__":
    lat, lon = 83.980642,  25.543123
    try:
        ph = get_soil_ph(lat, lon)
        print(f"Soil pH (0–5 cm) : {ph}")
    except Exception as e:
        print(f"Error: {str(e)}")