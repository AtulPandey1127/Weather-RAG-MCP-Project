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
        "addressdetails": 1,
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


def geocode_location_details(
    location: str,
) -> Optional[Dict]:
    """
    Resolve an Indian location and return structured geographic
    and administrative metadata.

    Returns:

        {
            "latitude": float,
            "longitude": float,
            "display_name": str,
            "state": str | None,
            "district": str | None,
        }
    """

    location = location.strip()

    # Direct coordinates
    if "," in location:
        parts = [part.strip() for part in location.split(",")]

        if len(parts) == 2:
            try:
                latitude = float(parts[0])
                longitude = float(parts[1])

                if (
                    -90 <= latitude <= 90
                    and -180 <= longitude <= 180
                ):
                    return {
                        "latitude": latitude,
                        "longitude": longitude,
                        "display_name": location,
                        "state": None,
                        "district": None,
                    }

            except ValueError:
                pass

    params = {
        "q": f"{location}, India",
        "format": "json",
        "limit": 1,
        "countrycodes": "in",
        "addressdetails": 1,
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

    address = result.get("address", {})

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result.get(
            "display_name",
            location,
        ),
        "state": address.get("state"),
        "district": (
            address.get("state_district")
            or address.get("district")
            or address.get("county")
        ),
    }


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


def weather_description(
    code: Optional[int],
) -> str:
    """
    Convert Open-Meteo WMO weather code to readable text.
    """

    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(
        code,
        "Unknown weather condition",
    )


def weather_severity(
    code: Optional[int],
) -> str:
    """
    Classify weather conditions into application-level severity.

    This is NOT an official IMD warning classification.
    """

    if code is None:
        return "unknown"

    if code in {95, 96, 99}:
        return "severe"

    if code in {65, 67, 82, 86}:
        return "high"

    if code in {61, 63, 80, 81}:
        return "moderate"

    return "normal"


# ---------------------------------------------------------------------------
# Stable document IDs
# ---------------------------------------------------------------------------

def _make_id(
    prefix: str,
    raw: Dict,
) -> str:
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
# Normalization helpers
# ---------------------------------------------------------------------------

def _get_index(
    values: Optional[List],
    index: int,
):
    """
    Safely retrieve a list value.
    """

    if not values or index >= len(values):
        return None

    return values[index]


# ---------------------------------------------------------------------------
# Current weather normalization
# ---------------------------------------------------------------------------

def normalize_current_weather(
    weather: Dict,
    location_label: str,
    location_details: Optional[Dict] = None,
) -> Dict:
    """
    Convert current Open-Meteo weather into a structured RAG document.
    """

    current = weather.get(
        "current",
        {},
    )

    weather_code = current.get(
        "weather_code"
    )

    latitude = (
        location_details.get("latitude")
        if location_details
        else None
    )

    longitude = (
        location_details.get("longitude")
        if location_details
        else None
    )

    state = (
        location_details.get("state")
        if location_details
        else None
    )

    district = (
        location_details.get("district")
        if location_details
        else None
    )

    current_time = current.get(
        "time"
    )

    forecast_date = (
        current_time[:10]
        if current_time
        else None
    )

    severity = weather_severity(
        weather_code
    )

    condition = weather_description(
        weather_code
    )

    headline = (
        f"Current weather in {location_label}"
    )

    narrative = (
        f"Current weather for {location_label}. "
        f"State: {state or 'Unknown'}. "
        f"District: {district or 'Unknown'}. "
        f"Temperature: "
        f"{current.get('temperature_2m')} °C. "
        f"Feels like: "
        f"{current.get('apparent_temperature')} °C. "
        f"Humidity: "
        f"{current.get('relative_humidity_2m')}%. "
        f"Condition: {condition}. "
        f"Severity: {severity}. "
        f"Precipitation: "
        f"{current.get('precipitation')} mm. "
        f"Cloud cover: "
        f"{current.get('cloud_cover')}%. "
        f"Wind speed: "
        f"{current.get('wind_speed_10m')} km/h. "
        f"Wind direction: "
        f"{current.get('wind_direction_10m')}°."
    )

    raw = {
        "location": location_label,
        "source": "open-meteo",
        "type": "current",
        "time": current_time,
    }

    return {
        "id": _make_id(
            "current",
            raw,
        ),
        "location": location_label,
        "state": state,
        "district": district,
        "latitude": latitude,
        "longitude": longitude,
        "source": "open-meteo",
        "source_type": "current",
        "headline": headline,
        "narrative_text": narrative,
        "forecast_date": forecast_date,
        "temperature_min_c": None,
        "temperature_max_c": current.get(
            "temperature_2m"
        ),
        "rainfall_mm": current.get(
            "precipitation"
        ),
        "precipitation_probability": None,
        "weather_code": weather_code,
        "severity": severity,
        "issued_at": current_time,
        "payload": weather,
        "synced_at": None,
    }


# ---------------------------------------------------------------------------
# Daily forecast normalization
# ---------------------------------------------------------------------------

def normalize_daily_forecasts(
    weather: Dict,
    location_label: str,
    location_details: Optional[Dict] = None,
) -> List[Dict]:
    """
    Convert daily Open-Meteo forecasts into structured RAG documents.
    """

    daily = weather.get(
        "daily",
        {},
    )

    dates = daily.get(
        "time",
        [],
    )

    documents = []

    latitude = (
        location_details.get("latitude")
        if location_details
        else None
    )

    longitude = (
        location_details.get("longitude")
        if location_details
        else None
    )

    state = (
        location_details.get("state")
        if location_details
        else None
    )

    district = (
        location_details.get("district")
        if location_details
        else None
    )

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
            daily.get(
                "precipitation_probability_max"
            ),
            index,
        )

        wind = _get_index(
            daily.get("wind_speed_10m_max"),
            index,
        )

        condition = weather_description(
            weather_code
        )

        severity = weather_severity(
            weather_code
        )

        headline = (
            f"Weather forecast for "
            f"{location_label} on {date}"
        )

        narrative = (
            f"Weather forecast for "
            f"{location_label} on {date}. "
            f"State: {state or 'Unknown'}. "
            f"District: {district or 'Unknown'}. "
            f"Condition: {condition}. "
            f"Severity: {severity}. "
            f"Minimum temperature: "
            f"{min_temp} °C. "
            f"Maximum temperature: "
            f"{max_temp} °C. "
            f"Total precipitation: "
            f"{precipitation} mm. "
            f"Rainfall: "
            f"{rain} mm. "
            f"Maximum precipitation probability: "
            f"{precipitation_probability}%. "
            f"Maximum wind speed: "
            f"{wind} km/h."
        )

        raw = {
            "location": location_label,
            "date": date,
            "type": "daily_forecast",
        }

        documents.append(
            {
                "id": _make_id(
                    "forecast",
                    raw,
                ),
                "location": location_label,
                "state": state,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "source": "open-meteo",
                "source_type": "forecast",
                "headline": headline,
                "narrative_text": narrative,
                "forecast_date": date,
                "temperature_min_c": min_temp,
                "temperature_max_c": max_temp,
                "rainfall_mm": rain,
                "precipitation_probability": (
                    precipitation_probability
                ),
                "weather_code": weather_code,
                "severity": severity,
                "issued_at": date,
                "payload": {
                    "date": date,
                    "weather_code": weather_code,
                    "condition": condition,
                    "temperature_min_c": min_temp,
                    "temperature_max_c": max_temp,
                    "precipitation_mm": precipitation,
                    "rain_mm": rain,
                    "precipitation_probability": (
                        precipitation_probability
                    ),
                    "max_wind_speed_kmh": wind,
                },
                "synced_at": None,
            }
        )

    return documents


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def upsert_documents(
    documents: List[Dict],
) -> int:
    """
    Upsert normalized weather documents into PostgreSQL/Lakebase.

    Lakebase is imported lazily so weather API functionality can be
    tested locally without Databricks credentials.
    """

    import lakebase

    if not documents:
        return 0

    sql = """
        INSERT INTO weather_documents (
            id,
            location,
            state,
            district,
            latitude,
            longitude,
            source,
            source_type,
            headline,
            narrative_text,
            forecast_date,
            temperature_min_c,
            temperature_max_c,
            rainfall_mm,
            precipitation_probability,
            weather_code,
            severity,
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
            %s,
            %s,
            %s,
            %s,
            %s,
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
            state = EXCLUDED.state,
            district = EXCLUDED.district,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            source = EXCLUDED.source,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            forecast_date = EXCLUDED.forecast_date,
            temperature_min_c = EXCLUDED.temperature_min_c,
            temperature_max_c = EXCLUDED.temperature_max_c,
            rainfall_mm = EXCLUDED.rainfall_mm,
            precipitation_probability =
                EXCLUDED.precipitation_probability,
            weather_code = EXCLUDED.weather_code,
            severity = EXCLUDED.severity,
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
                document.get("state"),
                document.get("district"),
                document.get("latitude"),
                document.get("longitude"),
                document.get(
                    "source",
                    "open-meteo",
                ),
                document.get("source_type"),
                document.get("headline"),
                document.get("narrative_text"),
                document.get("forecast_date"),
                document.get("temperature_min_c"),
                document.get("temperature_max_c"),
                document.get("rainfall_mm"),
                document.get(
                    "precipitation_probability"
                ),
                document.get("weather_code"),
                document.get("severity"),
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

            location_details = (
                geocode_location_details(
                    location
                )
            )

            if not location_details:

                print(
                    f"Could not resolve location: "
                    f"{location}"
                )

                continue

            latitude = location_details[
                "latitude"
            ]

            longitude = location_details[
                "longitude"
            ]

            label = location_details[
                "display_name"
            ]

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
                    location_details,
                )
            )

            documents.extend(
                normalize_daily_forecasts(
                    weather,
                    label,
                    location_details,
                )
            )

            count = upsert_documents(
                documents
            )

            total += count

            print(
                f"Synced {count} documents "
                f"for {label}"
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
        time.sleep(
            LOCATION_DELAY_SECONDS
        )

    return total
