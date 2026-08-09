"""
Indian Weather RAG API.

Provides:

    GET  /healthz
    POST /weather/ask
    POST /weather/sync

Architecture:

    Open-Meteo
        ↓
    weather_client
        ↓
    Lakebase / PostgreSQL
        ↓
    Hybrid RAG
        ├── pgvector
        └── BM25
        ↓
    RRF
        ↓
    Ollama
        ↓
    Grounded answer

No paid LLM API is required.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

import lakebase
import rag_service
import weather_client


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    "weather-rag"
)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(
    __name__
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route(
    "/healthz",
    methods=["GET"],
)
def healthz():
    """
    Basic application health check.

    This endpoint does NOT access Lakebase or Databricks.
    Therefore it can be used to verify that the application
    starts locally without database credentials.
    """

    return jsonify(
        {
            "status": "ok",
            "service": "indian-weather-rag",
        }
    )


# ---------------------------------------------------------------------------
# Weather RAG
# ---------------------------------------------------------------------------

@app.route(
    "/weather/ask",
    methods=["POST"],
)
def weather_ask():
    """
    Ask a weather question using the hybrid RAG pipeline.

    Request:

        {
            "query": "Will it rain in Kolkata tomorrow?",
            "top_k": 5
        }

    Response:

        {
            "answer": "...",
            "sources": [],
            "retrieved_documents": 5,
            "model": "llama3.2:3b",
            "retrieval": "hybrid"
        }

    This endpoint requires the weather database and embeddings
    to be available.
    """

    body = (
        request.get_json(
            silent=True
        )
        or {}
    )

    query = body.get(
        "query"
    )

    if not query or not isinstance(
        query,
        str,
    ):
        return jsonify(
            {
                "error": (
                    "Missing or invalid "
                    "'query' in request body"
                )
            }
        ), 400

    query = query.strip()

    if not query:
        return jsonify(
            {
                "error": (
                    "Query cannot be empty"
                )
            }
        ), 400

    try:

        top_k = int(
            body.get(
                "top_k",
                5,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return jsonify(
            {
                "error": (
                    "'top_k' must be "
                    "an integer"
                )
            }
        ), 400

    top_k = max(
        1,
        min(
            20,
            top_k,
        ),
    )

    try:

        result = (
            rag_service
            .answer_weather_question(
                query=query,
                top_k=top_k,
            )
        )

        return jsonify(
            result
        )

    except Exception as exc:

        logger.exception(
            "Weather RAG request failed"
        )

        return jsonify(
            {
                "error": (
                    "Failed to generate "
                    "weather answer"
                ),
                "details": str(exc),
            }
        ), 500


# ---------------------------------------------------------------------------
# Weather synchronization
# ---------------------------------------------------------------------------

@app.route(
    "/weather/sync",
    methods=["POST"],
)
def weather_sync():
    """
    Fetch and store weather data for Indian locations.

    Request:

        {
            "locations": [
                "Kolkata",
                "Delhi",
                "Mumbai"
            ]
        }

    This endpoint requires Lakebase credentials.
    """

    body = (
        request.get_json(
            silent=True
        )
        or {}
    )

    locations = body.get(
        "locations"
    )

    if (
        not isinstance(
            locations,
            list,
        )
        or not locations
    ):
        return jsonify(
            {
                "error": (
                    "Missing or invalid "
                    "'locations' list"
                )
            }
        ), 400

    cleaned_locations = []

    for location in locations:

        if not isinstance(
            location,
            str,
        ):
            continue

        location = location.strip()

        if location:
            cleaned_locations.append(
                location
            )

    if not cleaned_locations:

        return jsonify(
            {
                "error": (
                    "No valid locations "
                    "were provided"
                )
            }
        ), 400

    try:

        # Database initialization is intentionally
        # performed only when the sync endpoint is called.
        lakebase.ensure_weather_tables(
            embedding_dim=384
        )

        synced = (
            weather_client
            .sync_locations(
                cleaned_locations
            )
        )

        return jsonify(
            {
                "status": "success",
                "synced": synced,
                "locations": (
                    cleaned_locations
                ),
            }
        )

    except Exception as exc:

        logger.exception(
            "Weather synchronization failed"
        )

        return jsonify(
            {
                "error": (
                    "Weather synchronization "
                    "failed"
                ),
                "details": str(exc),
            }
        ), 500


# ---------------------------------------------------------------------------
# 404 handler
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    """
    Return JSON for unknown endpoints.
    """

    return jsonify(
        {
            "error": "Endpoint not found",
        }
    ), 404


# ---------------------------------------------------------------------------
# 405 handler
# ---------------------------------------------------------------------------

@app.errorhandler(405)
def method_not_allowed(error):
    """
    Return JSON for unsupported HTTP methods.
    """

    return jsonify(
        {
            "error": "Method not allowed",
        }
    ), 405


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(error):
    """
    Catch unexpected application errors.
    """

    logger.exception(
        "Unhandled application error"
    )

    return jsonify(
        {
            "error": "Internal server error",
            "details": str(error),
        }
    ), 500


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    host = os.getenv(
        "FLASK_RUN_HOST",
        "0.0.0.0",
    )

    port = int(
        os.getenv(
            "FLASK_RUN_PORT",
            "8000",
        )
    )

    app.run(
        debug=True,
        host=host,
        port=port,
    )
