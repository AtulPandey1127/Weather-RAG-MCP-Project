"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL stored in a Databricks
secret scope.

The database contains:
    - weather_documents
    - weather_embeddings

The weather_documents table stores structured Indian-weather
metadata for metadata filtering and RAG retrieval.
"""

import base64
import os
from contextlib import contextmanager

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.rows import dict_row
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Databricks / Lakebase configuration
# ---------------------------------------------------------------------------

_w = WorkspaceClient()

_SCOPE = os.environ.get(
    "LAKEBASE_SECRET_SCOPE",
    "database",
)

_KEY = os.environ.get(
    "LAKEBASE_SECRET_KEY",
    "lakebase-url",
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _lakebase_url() -> str:
    """
    Fetch and decode the Lakebase PostgreSQL connection URL
    from the Databricks secret scope.
    """

    secret = _w.secrets.get_secret(
        scope=_SCOPE,
        key=_KEY,
    )

    return base64.b64decode(
        secret.value
    ).decode("utf-8")


@contextmanager
def get_connection():
    """
    Yield a raw psycopg connection with dict_row factory.
    """

    conn = psycopg.connect(
        _lakebase_url(),
        row_factory=dict_row,
    )

    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """
    Return a SQLAlchemy engine for Lakebase.
    """

    return create_engine(
        _lakebase_url()
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def run_query(
    sql: str,
    params: tuple | dict | None = None,
) -> list[dict]:
    """
    Run a read query against Lakebase.

    Returns:
        List of rows represented as dictionaries.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                params,
            )

            return cur.fetchall()


def run_write(
    sql: str,
    params: tuple | dict | None = None,
) -> int:
    """
    Run an INSERT/UPDATE/DELETE/DDL statement.

    Returns:
        Number of affected rows when available.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                params,
            )

            conn.commit()

            return cur.rowcount


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def ensure_weather_tables(
    embedding_dim: int = 384,
) -> None:
    """
    Create and migrate the weather database tables.

    The default embedding dimension is 384 because the project
    currently uses all-MiniLM-L6-v2.
    """

    # -----------------------------------------------------------------------
    # pgvector
    # -----------------------------------------------------------------------

    run_write(
        "CREATE EXTENSION IF NOT EXISTS vector;"
    )

    # -----------------------------------------------------------------------
    # Weather documents
    # -----------------------------------------------------------------------

    run_write(
        """
        CREATE TABLE IF NOT EXISTS weather_documents (

            id TEXT PRIMARY KEY,

            location TEXT NOT NULL,

            state TEXT,

            district TEXT,

            latitude DOUBLE PRECISION,

            longitude DOUBLE PRECISION,

            source TEXT NOT NULL DEFAULT 'open-meteo',

            source_type TEXT NOT NULL,

            headline TEXT,

            narrative_text TEXT,

            forecast_date DATE,

            temperature_min_c DOUBLE PRECISION,

            temperature_max_c DOUBLE PRECISION,

            rainfall_mm DOUBLE PRECISION,

            precipitation_probability DOUBLE PRECISION,

            weather_code INTEGER,

            severity TEXT,

            issued_at TIMESTAMPTZ,

            payload JSONB NOT NULL,

            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # -----------------------------------------------------------------------
    # Migrations for existing installations
    # -----------------------------------------------------------------------
    #
    # CREATE TABLE IF NOT EXISTS does not modify an existing table.
    # These statements make the schema backward-compatible if the table
    # already existed using the older weather schema.
    # -----------------------------------------------------------------------

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS state TEXT
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS district TEXT
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'open-meteo'
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS forecast_date DATE
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS temperature_min_c DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS temperature_max_c DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS rainfall_mm DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS precipitation_probability DOUBLE PRECISION
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS weather_code INTEGER
        """
    )

    run_write(
        """
        ALTER TABLE weather_documents
        ADD COLUMN IF NOT EXISTS severity TEXT
        """
    )

    # -----------------------------------------------------------------------
    # Weather document indexes
    # -----------------------------------------------------------------------

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_location
        ON weather_documents (location)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_state
        ON weather_documents (state)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_district
        ON weather_documents (district)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_source
        ON weather_documents (source)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_source_type
        ON weather_documents (source_type)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_forecast_date
        ON weather_documents (forecast_date)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_precipitation_probability
        ON weather_documents (
            precipitation_probability
        )
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_documents_issued_at
        ON weather_documents (issued_at)
        """
    )

    # -----------------------------------------------------------------------
    # Weather embeddings
    # -----------------------------------------------------------------------

    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS weather_embeddings (

            id TEXT PRIMARY KEY,

            document_id TEXT NOT NULL,

            chunk_index INT NOT NULL,

            chunk_text TEXT NOT NULL,

            embedding VECTOR({embedding_dim}) NOT NULL,

            model_name TEXT NOT NULL,

            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # -----------------------------------------------------------------------
    # Embedding indexes
    # -----------------------------------------------------------------------

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_embeddings_document_id
        ON weather_embeddings (document_id)
        """
    )

    run_write(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_embeddings_embedding
        ON weather_embeddings
        USING hnsw (
            embedding vector_cosine_ops
        )
        """
    )
