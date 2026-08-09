"""
Lakebase (Databricks-managed Postgres) connection helper.

Databricks authentication is initialized lazily so that the application
can start locally without Lakebase credentials.

Lakebase credentials are only required when a database operation is used.
"""

from __future__ import annotations

import base64
import os
from contextlib import contextmanager

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.rows import dict_row
from sqlalchemy import create_engine


_SCOPE = os.environ.get(
    "LAKEBASE_SECRET_SCOPE",
    "database",
)

_KEY = os.environ.get(
    "LAKEBASE_SECRET_KEY",
    "lakebase-url",
)


# ---------------------------------------------------------------------------
# Lazy Databricks client
# ---------------------------------------------------------------------------

_workspace_client: WorkspaceClient | None = None


def _get_workspace_client() -> WorkspaceClient:
    """
    Create the Databricks WorkspaceClient only when Lakebase access
    is actually required.
    """

    global _workspace_client

    if _workspace_client is None:
        _workspace_client = WorkspaceClient()

    return _workspace_client


# ---------------------------------------------------------------------------
# Lakebase URL
# ---------------------------------------------------------------------------

def _lakebase_url() -> str:
    """
    Fetch and decode the Lakebase connection URL from Databricks secrets.
    """

    workspace_client = _get_workspace_client()

    secret = workspace_client.secrets.get_secret(
        scope=_SCOPE,
        key=_KEY,
    )

    return base64.b64decode(
        secret.value
    ).decode("utf-8")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """
    Yield a PostgreSQL connection.

    Databricks authentication is triggered only when this function
    is actually called.
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
    Execute a read query.
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
    Execute an INSERT/UPDATE/DELETE/DDL statement.
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
    Create the weather tables and indexes.

    This function requires working Lakebase credentials.
    """

    run_write(
        "CREATE EXTENSION IF NOT EXISTS vector;"
    )

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
    # Existing-table migrations
    # -----------------------------------------------------------------------

    migration_columns = [
        (
            "state",
            "TEXT",
        ),
        (
            "district",
            "TEXT",
        ),
        (
            "latitude",
            "DOUBLE PRECISION",
        ),
        (
            "longitude",
            "DOUBLE PRECISION",
        ),
        (
            "source",
            "TEXT DEFAULT 'open-meteo'",
        ),
        (
            "forecast_date",
            "DATE",
        ),
        (
            "temperature_min_c",
            "DOUBLE PRECISION",
        ),
        (
            "temperature_max_c",
            "DOUBLE PRECISION",
        ),
        (
            "rainfall_mm",
            "DOUBLE PRECISION",
        ),
        (
            "precipitation_probability",
            "DOUBLE PRECISION",
        ),
        (
            "weather_code",
            "INTEGER",
        ),
        (
            "severity",
            "TEXT",
        ),
    ]

    for column_name, column_type in migration_columns:

        run_write(
            f"""
            ALTER TABLE weather_documents
            ADD COLUMN IF NOT EXISTS
            {column_name} {column_type}
            """
        )

    # -----------------------------------------------------------------------
    # Weather indexes
    # -----------------------------------------------------------------------

    indexes = [
        (
            "idx_weather_documents_location",
            "location",
        ),
        (
            "idx_weather_documents_state",
            "state",
        ),
        (
            "idx_weather_documents_district",
            "district",
        ),
        (
            "idx_weather_documents_source",
            "source",
        ),
        (
            "idx_weather_documents_source_type",
            "source_type",
        ),
        (
            "idx_weather_documents_forecast_date",
            "forecast_date",
        ),
        (
            "idx_weather_documents_precipitation_probability",
            "precipitation_probability",
        ),
        (
            "idx_weather_documents_issued_at",
            "issued_at",
        ),
    ]

    for index_name, column_name in indexes:

        run_write(
            f"""
            CREATE INDEX IF NOT EXISTS
            {index_name}
            ON weather_documents ({column_name})
            """
        )

    # -----------------------------------------------------------------------
    # Embeddings
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
