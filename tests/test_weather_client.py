import pytest

import weather_client


def test_geocode_coordinates_directly():
    result = weather_client.geocode_location(
        "22.5726,88.3639"
    )

    assert result is not None

    latitude, longitude, label = result

    assert latitude == pytest.approx(22.5726)
    assert longitude == pytest.approx(88.3639)
    assert label == "22.5726,88.3639"


def test_geocode_location_details_coordinates():
    result = weather_client.geocode_location_details(
        "22.5726,88.3639"
    )

    assert result is not None
    assert result["latitude"] == pytest.approx(22.5726)
    assert result["longitude"] == pytest.approx(88.3639)


def test_weather_description():
    assert (
        weather_client.weather_description(0)
        == "Clear sky"
    )

    assert (
        weather_client.weather_description(95)
        == "Thunderstorm"
    )

    assert (
        weather_client.weather_description(None)
        == "Unknown"
    )


def test_weather_severity():
    assert (
        weather_client.weather_severity(0)
        == "normal"
    )

    assert (
        weather_client.weather_severity(63)
        == "moderate"
    )

    assert (
        weather_client.weather_severity(65)
        == "high"
    )

    assert (
        weather_client.weather_severity(95)
        == "severe"
    )

    assert (
        weather_client.weather_severity(None)
        == "unknown"
    )


def test_normalize_daily_forecast():
    weather = {
        "daily": {
            "time": ["2026-08-10"],
            "weather_code": [61],
            "temperature_2m_max": [32.0],
            "temperature_2m_min": [26.0],
            "precipitation_sum": [5.2],
            "rain_sum": [4.8],
            "precipitation_probability_max": [70],
            "wind_speed_10m_max": [12.5],
        }
    }

    location_details = {
        "latitude": 22.5726,
        "longitude": 88.3639,
        "state": "West Bengal",
        "district": "Kolkata",
    }

    documents = weather_client.normalize_daily_forecasts(
        weather,
        "Kolkata",
        location_details,
    )

    assert len(documents) == 1

    document = documents[0]

    assert document["location"] == "Kolkata"
    assert document["forecast_date"] == "2026-08-10"
    assert document["temperature_min_c"] == 26.0
    assert document["temperature_max_c"] == 32.0
    assert document["weather_code"] == 61
    assert document["severity"] == "moderate"
    assert document["source"] == "open-meteo"
