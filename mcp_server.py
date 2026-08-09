"""
Indian Weather RAG MCP Server.

Exposes the project's weather and RAG capabilities
through the Model Context Protocol.

Tools:
    get_weather
    search_weather
    ask_weather
    sync_weather
    database_health

Transport:
    stdio

No paid API key is required.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

import lakebase
import rag_service
import weather_client


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "indian-weather-rag"
)


# ---------------------------------------------------------------------------
# Tool: get_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def get_weather(
    location: str,
) -> dict[str, Any]:
    """
    Get current weather and a 7-day forecast for an Indian location.

    Examples:
        Kolkata
        Delhi
        Mumbai
        Bengaluru
        22.5726,88.3639
    """

    if not location or not location.strip():
        raise ValueError(
            "location cannot be empty"
        )

    location = location.strip()

    details = (
        weather_client.geocode_location_details(
            location
        )
    )

    if not details:
        return {
            "success": False,
            "error": (
                f"Could not resolve location: "
                f"{location}"
            ),
        }

    latitude = details["latitude"]
    longitude = details["longitude"]

    weather = (
        weather_client.fetch_weather(
            latitude,
            longitude,
        )
    )

    current = weather.get(
        "current",
        {},
    )

    daily = weather.get(
        "daily",
        {},
    )

    return {
        "success": True,
        "location": details,
        "current": current,
        "daily": daily,
    }


# ---------------------------------------------------------------------------
# Tool: search_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def search_weather(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search the Indian weather knowledge base using
    hybrid vector + BM25 retrieval.

    This does NOT generate an LLM answer.
    It returns the retrieved weather documents.
    """

    if not query or not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    top_k = max(
        1,
        min(
            20,
            int(top_k),
        ),
    )

    documents = (
        rag_service.retrieve_weather(
            query.strip(),
            top_k,
        )
    )

    return {
        "success": True,
        "query": query.strip(),
        "retrieval": "hybrid",
        "documents": documents,
        "count": len(documents),
    }


# ---------------------------------------------------------------------------
# Tool: ask_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_weather(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Ask a natural-language Indian weather question.

    Uses:

        vector retrieval
        +
        BM25
        ↓
        RRF
        ↓
        Ollama
        ↓
        grounded answer
    """

    if not query or not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    top_k = max(
        1,
        min(
            20,
            int(top_k),
        ),
    )

    return (
        rag_service.answer_weather_question(
            query=query.strip(),
            top_k=top_k,
        )
    )


# ---------------------------------------------------------------------------
# Tool: sync_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def sync_weather(
    locations: list[str],
) -> dict[str, Any]:
    """
    Fetch and store fresh weather data for Indian locations.

    Example:

        ["Kolkata", "Delhi", "Mumbai"]
    """

    if not locations:
        raise ValueError(
            "locations cannot be empty"
        )

    cleaned_locations = [
        location.strip()
        for location in locations
        if location
        and location.strip()
    ]

    if not cleaned_locations:
        raise ValueError(
            "No valid locations supplied"
        )

    count = (
        weather_client.sync_locations(
            cleaned_locations
        )
    )

    return {
        "success": True,
        "locations": cleaned_locations,
        "documents_synced": count,
    }


# ---------------------------------------------------------------------------
# Tool: database_health
# ---------------------------------------------------------------------------

@mcp.tool()
def database_health() -> dict[str, Any]:
    """
    Check whether the configured PostgreSQL/Lakebase
    database is reachable.
    """

    try:

        connected = (
            lakebase.check_connection()
        )

        return {
            "success": connected,
            "backend": (
                lakebase.DATABASE_BACKEND
            ),
            "status": (
                "ok"
                if connected
                else "unavailable"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "backend": (
                lakebase.DATABASE_BACKEND
            ),
            "status": "error",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )
