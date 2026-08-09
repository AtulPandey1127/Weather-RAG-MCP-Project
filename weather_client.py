"""
Indian weather data client.

Uses Open-Meteo for free weather observations and forecasts.
Locations are resolved through OpenStreetMap Nominatim.

The normalized output is designed for the project's RAG pipeline
and PostgreSQL/Lakebase weather_documents table.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple

import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

USER_AGENT = "weather-rag-india/1.0"

REQUEST_TIMEOUT = 20
LOCATION_DELAY_SECONDS = 1


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def geocode_location(
    location: str,
) -> Optional[Tuple[float, float, str]]:
    """
    Resolve an Indian city/district to latitude, longitude and display name.

    Also accepts direct coordinates in the form:

        "22.5726,88.3639"
    """

    location = location.strip()

    # Direct latitude/longitude input
    if "," in location:
        parts = [part.strip() for part in location.split(",")]

        if len(parts) == 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])

                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon, f"{lat},{lon}"

            except ValueError:
                pass

    params = {
        "q": f"{location}, India",
        "format": "json",
        "limit": 1,
        "countrycodes": "in",
    }

    headers = {
        "User-Agent": USER_AGENT,
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    result = results[0]

    return (
        float(result["lat"]),
        float(result["lon"]),
        result.get("display_name", location),
    )


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def fetch_weather(
    latitude: float,
    longitude: float,
) -> Dict:
    """
    Fetch current weather and a 7-day forecast.

    Open-Meteo does not require an API key.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "sunrise",
                "sunset",
                "precipitation_sum",
                "rain_sum",
                "precipitation_probability_max",
                "wind_speed_10m_max",
            ]
        ),
        "timezone": "auto",
        "forecast_days": 7,
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------------------------
# Weather code conversion
# ---------------------------------------------------------------------------

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather_description(code: Optional[int]) -> str:
    """Convert Open-Meteo WMO weather code to readable text."""

    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(code, "Unknown weather condition")


# ---------------------------------------------------------------------------
# Stable document IDs
# ---------------------------------------------------------------------------

def _make_id(prefix: str, raw: Dict) -> str:
    """
    Create a deterministic document ID.

    This allows repeated ingestion without creating duplicates.
    """

    canonical = json.dumps(
        {
            "prefix": prefix,
            **raw,
        },
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_current_weather(
    weather: Dict,
    location_label: str,
) -> Dict:
    """
    Convert current Open-Meteo weather into a RAG document.
    """

    current = weather.get("current", {})

    weather_code = current.get("weather_code")

    headline = (
        f"Current weather in {location_label}"
    )

    narrative = (
        f"Current weather for {location_label}. "
        f"Temperature: {current.get('temperature_2m')} °C. "
        f"Feels like: {current.get('apparent_temperature')} °C. "
        f"Humidity: {current.get('relative_humidity_2m')}%. "
        f"Condition: {weather_description(weather_code)}. "
        f"Precipitation: {current.get('precipitation')} mm. "
        f"Cloud cover: {current.get('cloud_cover')}%. "
        f"Wind speed: {current.get('wind_speed_10m')} km/h. "
        f"Wind direction: {current.get('wind_direction_10m')}°."
    )

    raw = {
        "location": location_label,
        "source": "open-meteo",
        "type": "current",
        "time": current.get("time"),
    }

    return {
        "id": _make_id("current", raw),
        "location": location_label,
        "source_type": "current",
        "headline": headline,
        "narrative_text": narrative,
        "issued_at": current.get("time"),
        "payload": weather,
        "synced_at": None,
    }


def normalize_daily_forecasts(
    weather: Dict,
    location_label: str,
) -> List[Dict]:
    """
    Convert daily Open-Meteo forecasts into individual RAG documents.
    """

    daily = weather.get("daily", {})

    dates = daily.get("time", [])

    documents = []

    for index, date in enumerate(dates):

        weather_code = _get_index(
            daily.get("weather_code"),
            index,
        )

        max_temp = _get_index(
            daily.get("temperature_2m_max"),
            index,
        )

        min_temp = _get_index(
            daily.get("temperature_2m_min"),
            index,
        )

        precipitation = _get_index(
            daily.get("precipitation_sum"),
            index,
        )

        rain = _get_index(
            daily.get("rain_sum"),
            index,
        )

        precipitation_probability = _get_index(
            daily.get("precipitation_probability_max"),
            index,
        )

        wind = _get_index(
            daily.get("wind_speed_10m_max"),
            index,
        )

        condition = weather_description(weather_code)

        headline = (
            f"Weather forecast for {location_label} on {date}"
        )

        narrative = (
            f"Weather forecast for {location_label} on {date}. "
            f"Condition: {condition}. "
            f"Minimum temperature: {min_temp} °C. "
            f"Maximum temperature: {max_temp} °C. "
            f"Total precipitation: {precipitation} mm. "
            f"Rainfall: {rain} mm. "
            f"Maximum precipitation probability: "
            f"{precipitation_probability}%. "
            f"Maximum wind speed: {wind} km/h."
        )

        raw = {
            "location": location_label,
            "date": date,
            "type": "daily_forecast",
        }

        documents.append(
            {
                "id": _make_id("forecast", raw),
                "location": location_label,
                "source_type": "forecast",
                "headline": headline,
                "narrative_text": narrative,
                "issued_at": date,
                "payload": {
                    "date": date,
                    "weather_code": weather_code,
                    "condition": condition,
                    "temperature_min_c": min_temp,
                    "temperature_max_c": max_temp,
                    "precipitation_mm": precipitation,
                    "rain_mm": rain,
                    "precipitation_probability": precipitation_probability,
                    "max_wind_speed_kmh": wind,
                },
                "synced_at": None,
            }
        )

    return documents


def _get_index(
    values: Optional[List],
    index: int,
):
    """Safely retrieve a list value."""

    if not values or index >= len(values):
        return None

    return values[index]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def upsert_documents(
    documents: List[Dict],
) -> int:
    """
    Upsert normalized weather documents into PostgreSQL/Lakebase.
    """
    import lakebase
    if not documents:
        return 0

    sql = """
        INSERT INTO weather_documents (
            id,
            location,
            source_type,
            headline,
            narrative_text,
            issued_at,
            payload,
            synced_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            now()
        )
        ON CONFLICT (id)
        DO UPDATE SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            payload = EXCLUDED.payload,
            synced_at = now()
    """

    params = []

    for document in documents:
        params.append(
            (
                document.get("id"),
                document.get("location"),
                document.get("source_type"),
                document.get("headline"),
                document.get("narrative_text"),
                document.get("issued_at"),
                json.dumps(
                    document.get("payload") or {},
                    default=str,
                ),
            )
        )

    with lakebase.get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.executemany(
                sql,
                params,
            )

            connection.commit()

            return cursor.rowcount


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def sync_locations(
    locations: List[str],
    limit: int = 50,
) -> int:
    """
    Fetch and store weather data for Indian locations.

    `limit` is retained for backward compatibility with the
    previous NWS implementation.
    """

    del limit

    total = 0

    for location in locations:

        try:
            resolved = geocode_location(location)

            if not resolved:
                print(
                    f"Could not resolve location: {location}"
                )
                continue

            latitude, longitude, label = resolved

            print(
                f"Fetching weather for {label} "
                f"({latitude}, {longitude})"
            )

            weather = fetch_weather(
                latitude,
                longitude,
            )

            documents = []

            documents.append(
                normalize_current_weather(
                    weather,
                    label,
                )
            )

            documents.extend(
                normalize_daily_forecasts(
                    weather,
                    label,
                )
            )

            count = upsert_documents(documents)

            total += count

            print(
                f"Synced {count} documents for {label}"
            )

        except requests.RequestException as exc:

            print(
                f"Weather API error for "
                f"{location}: {exc}"
            )

        except Exception as exc:

            print(
                f"Unexpected error for "
                f"{location}: {exc}"
            )

        # Respect Nominatim/Open-Meteo usage.
        time.sleep(LOCATION_DELAY_SECONDS)

    return total
