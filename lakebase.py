"""
Database connection helper.

Supports two backends:

1. Local PostgreSQL + pgvector
   Used for development and testing.

2. Databricks Lakebase
   Used for managed/production deployment.

Backend selection:

    DATABASE_BACKEND=local
        -> DATABASE_URL

    DATABASE_BACKEND=lakebase
        -> Databricks secret

Local default:

    postgresql://weather_user:weather_password@localhost:5432/weather_rag
"""

from __future__ import annotations

import base64
import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_BACKEND = os.environ.get(
    "DATABASE_BACKEND",
    "local",
).lower()


LOCAL_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://weather_user:weather_password@localhost:5432/weather_rag",
)


LAKEBASE_SECRET_SCOPE = os.environ.get(
    "LAKEBASE_SECRET_SCOPE",
    "database",
)


LAKEBASE_SECRET_KEY = os.environ.get(
    "LAKEBASE_SECRET_KEY",
    "lakebase-url",
)


# ---------------------------------------------------------------------------
# Lazy Databricks client
# ---------------------------------------------------------------------------

_workspace_client = None


def _get_workspace_client():
    """
    Lazily create the Databricks WorkspaceClient.

    This is intentionally NOT executed when lakebase.py is imported.
    """

    global _workspace_client

    if _workspace_client is None:

        from databricks.sdk import WorkspaceClient

        _workspace_client = WorkspaceClient()

    return _workspace_client


# ---------------------------------------------------------------------------
# Lakebase URL
# ---------------------------------------------------------------------------

def _get_lakebase_url() -> str:
    """
    Fetch the Lakebase PostgreSQL URL from Databricks secrets.
    """

    workspace_client = (
        _get_workspace_client()
    )

    secret = (
        workspace_client
        .secrets
        .get_secret(
            scope=LAKEBASE_SECRET_SCOPE,
            key=LAKEBASE_SECRET_KEY,
        )
    )

    return base64.b64decode(
        secret.value
    ).decode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    """
    Return the active database connection URL.
    """

    if DATABASE_BACKEND == "local":

        return LOCAL_DATABASE_URL

    if DATABASE_BACKEND == "lakebase":

        return _get_lakebase_url()

    raise ValueError(
        "Unsupported DATABASE_BACKEND: "
        f"{DATABASE_BACKEND}. "
        "Use 'local' or 'lakebase'."
    )


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """
    Yield a PostgreSQL connection using dict rows.
    """

    connection = psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    )

    try:

        yield connection

    finally:

        connection.close()


# ---------------------------------------------------------------------------
# SQLAlchemy engine
# ---------------------------------------------------------------------------

def get_engine():
    """
    Return a SQLAlchemy engine for the active database.
    """

    database_url = get_database_url()

    # SQLAlchemy's psycopg driver.
    if database_url.startswith(
        "postgresql://"
    ):

        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    elif database_url.startswith(
        "postgres://"
    ):

        database_url = database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    return create_engine(
        database_url
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def run_query(
    sql: str,
    params: tuple | dict | None = None,
) -> list[dict]:
    """
    Execute a SELECT query and return rows as dictionaries.
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                sql,
                params,
            )

            return cursor.fetchall()


def run_write(
    sql: str,
    params: tuple | dict | None = None,
) -> int:
    """
    Execute an INSERT, UPDATE, DELETE, or DDL statement.
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                sql,
                params,
            )

            connection.commit()

            return cursor.rowcount


# ---------------------------------------------------------------------------
# Database health
# ---------------------------------------------------------------------------

def check_connection() -> bool:
    """
    Check whether the configured database is reachable.
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT 1"
            )

            result = cursor.fetchone()

            return bool(
                result
            )


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def ensure_weather_tables(
    embedding_dim: int = 384,
) -> None:
    """
    Create the weather tables, indexes, and pgvector index.
    """

    # -----------------------------------------------------------------------
    # pgvector
    # -----------------------------------------------------------------------

    run_write(
        """
        CREATE EXTENSION IF NOT EXISTS vector;
        """
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

            source TEXT NOT NULL
                DEFAULT 'open-meteo',

            source_type TEXT NOT NULL,

            headline TEXT,

            narrative_text TEXT,

            forecast_date DATE,

            temperature_min_c DOUBLE PRECISION,

            temperature_max_c DOUBLE PRECISION,

            rainfall_mm DOUBLE PRECISION,

            precipitation_probability
                DOUBLE PRECISION,

            weather_code INTEGER,

            severity TEXT,

            issued_at TIMESTAMPTZ,

            payload JSONB NOT NULL,

            synced_at TIMESTAMPTZ
                NOT NULL DEFAULT now()
        )
        """
    )

    # -----------------------------------------------------------------------
    # Existing-table migrations
    # -----------------------------------------------------------------------

    migrations = [
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

    for column_name, column_type in migrations:

        run_write(
            f"""
            ALTER TABLE weather_documents
            ADD COLUMN IF NOT EXISTS
            {column_name} {column_type}
            """
        )

    # -----------------------------------------------------------------------
    # Weather document indexes
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
    # Weather embeddings
    # -----------------------------------------------------------------------

    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS weather_embeddings (

            id TEXT PRIMARY KEY,

            document_id TEXT NOT NULL,

            chunk_index INT NOT NULL,

            chunk_text TEXT NOT NULL,

            embedding VECTOR({embedding_dim})
                NOT NULL,

            model_name TEXT NOT NULL,

            created_at TIMESTAMPTZ
                NOT NULL DEFAULT now()
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
        ON weather_embeddings (
            document_id
        )
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


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "Database backend:",
        DATABASE_BACKEND,
    )

    print(
        "Database URL:",
        (
            "configured"
            if DATABASE_BACKEND == "lakebase"
            else LOCAL_DATABASE_URL
        ),
    )

    try:

        if check_connection():

            print(
                "Database connection: OK"
            )

    except Exception as exc:

        print(
            "Database connection: FAILED"
        )

        print(
            exc
        )
