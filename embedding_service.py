"""
Weather document embedding service.

Pipeline:

    weather_documents
          ↓
       chunking
          ↓
    MiniLM embeddings
          ↓
    weather_embeddings
          ↓
       pgvector

Uses the same 384-dimensional embedding model as rag_service.py.

No paid API is required.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from sentence_transformers import SentenceTransformer

import lakebase


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.environ.get(
    "WEATHER_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = 384

CHUNK_SIZE = int(
    os.environ.get(
        "WEATHER_CHUNK_SIZE",
        "500",
    )
)

CHUNK_OVERLAP = int(
    os.environ.get(
        "WEATHER_CHUNK_OVERLAP",
        "100",
    )
)


logger = logging.getLogger(
    __name__
)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split weather text into overlapping chunks.

    Character-based chunking is sufficient for the relatively
    short structured weather documents used by this project.
    """

    if not text:
        return []

    text = " ".join(
        text.split()
    )

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(text):
            break

        start = (
            end - overlap
        )

    return chunks


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate 384-dimensional embeddings using MiniLM.
    """

    if not texts:
        return []

    vectors = _embedding_model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return [
        [
            float(value)
            for value in vector
        ]
        for vector in vectors
    ]


# ---------------------------------------------------------------------------
# Stable embedding IDs
# ---------------------------------------------------------------------------

def make_embedding_id(
    document_id: str,
    chunk_index: int,
    chunk_text_value: str,
) -> str:
    """
    Create a deterministic embedding ID.

    This makes ingestion idempotent.
    """

    raw = (
        f"{document_id}:"
        f"{chunk_index}:"
        f"{chunk_text_value}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Vector formatting
# ---------------------------------------------------------------------------

def vector_literal(
    vector: list[float],
) -> str:
    """
    Convert a Python vector into PostgreSQL pgvector syntax.
    """

    return (
        "["
        + ",".join(
            repr(float(value))
            for value in vector
        )
        + "]"
    )


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

def get_documents(
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Retrieve weather documents that can be embedded.
    """

    sql = """
        SELECT
            id,
            location,
            state,
            district,
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
            issued_at

        FROM weather_documents

        ORDER BY synced_at DESC

        LIMIT %s
    """

    return lakebase.run_query(
        sql,
        (
            limit,
        ),
    )


# ---------------------------------------------------------------------------
# Document text construction
# ---------------------------------------------------------------------------

def build_document_text(
    document: dict[str, Any],
) -> str:
    """
    Build retrieval-friendly text from structured weather metadata.

    Keeping metadata inside the text improves BM25 retrieval while
    the structured columns remain available for filtering.
    """

    fields = [
        (
            "Location",
            document.get(
                "location"
            ),
        ),
        (
            "State",
            document.get(
                "state"
            ),
        ),
        (
            "District",
            document.get(
                "district"
            ),
        ),
        (
            "Source",
            document.get(
                "source"
            ),
        ),
        (
            "Source type",
            document.get(
                "source_type"
            ),
        ),
        (
            "Forecast date",
            document.get(
                "forecast_date"
            ),
        ),
        (
            "Temperature minimum C",
            document.get(
                "temperature_min_c"
            ),
        ),
        (
            "Temperature maximum C",
            document.get(
                "temperature_max_c"
            ),
        ),
        (
            "Rainfall mm",
            document.get(
                "rainfall_mm"
            ),
        ),
        (
            "Precipitation probability",
            document.get(
                "precipitation_probability"
            ),
        ),
        (
            "Weather code",
            document.get(
                "weather_code"
            ),
        ),
        (
            "Severity",
            document.get(
                "severity"
            ),
        ),
        (
            "Headline",
            document.get(
                "headline"
            ),
        ),
        (
            "Weather information",
            document.get(
                "narrative_text"
            ),
        ),
    ]

    parts = []

    for name, value in fields:

        if value is None:
            continue

        parts.append(
            f"{name}: {value}"
        )

    return "\n".join(
        parts
    )


# ---------------------------------------------------------------------------
# Store embeddings
# ---------------------------------------------------------------------------

def store_embeddings(
    document: dict[str, Any],
    chunks: list[str],
    vectors: list[list[float]],
) -> int:
    """
    Replace existing embeddings for a document and insert
    the current chunks.

    Replacing the document's embeddings prevents stale chunks
    from remaining after weather data is updated.
    """

    document_id = document[
        "id"
    ]

    # Remove old embeddings for this document.
    lakebase.run_write(
        """
        DELETE FROM weather_embeddings
        WHERE document_id = %s
        """,
        (
            document_id,
        ),
    )

    if not chunks:
        return 0

    if len(chunks) != len(vectors):
        raise ValueError(
            "Number of chunks and vectors must match"
        )

    sql = """
        INSERT INTO weather_embeddings (
            id,
            document_id,
            chunk_index,
            chunk_text,
            embedding,
            model_name
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s::vector,
            %s
        )
        ON CONFLICT (id)
        DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name
    """

    params = []

    for index, (
        chunk,
        vector,
    ) in enumerate(
        zip(
            chunks,
            vectors,
        )
    ):

        embedding_id = make_embedding_id(
            document_id,
            index,
            chunk,
        )

        params.append(
            (
                embedding_id,
                document_id,
                index,
                chunk,
                vector_literal(
                    vector
                ),
                EMBEDDING_MODEL,
            )
        )

    with lakebase.get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.executemany(
                sql,
                params,
            )

        connection.commit()

    return len(
        params
    )


# ---------------------------------------------------------------------------
# Embed one document
# ---------------------------------------------------------------------------

def embed_document(
    document: dict[str, Any],
) -> int:
    """
    Chunk and embed a single weather document.
    """

    text = build_document_text(
        document
    )

    chunks = chunk_text(
        text
    )

    if not chunks:

        logger.warning(
            "Skipping empty document: %s",
            document.get("id"),
        )

        return 0

    vectors = generate_embeddings(
        chunks
    )

    count = store_embeddings(
        document,
        chunks,
        vectors,
    )

    logger.info(
        "Embedded document %s into %d chunks",
        document.get("id"),
        count,
    )

    return count


# ---------------------------------------------------------------------------
# Full indexing
# ---------------------------------------------------------------------------

def index_documents(
    limit: int = 1000,
) -> dict[str, int]:
    """
    Generate embeddings for weather documents.

    Returns ingestion statistics.
    """

    documents = get_documents(
        limit=limit
    )

    documents_processed = 0
    chunks_created = 0

    for document in documents:

        count = embed_document(
            document
        )

        documents_processed += 1
        chunks_created += count

    return {
        "documents_processed": (
            documents_processed
        ),
        "chunks_created": (
            chunks_created
        ),
    }


# ---------------------------------------------------------------------------
# Index a specific document
# ---------------------------------------------------------------------------

def index_document_by_id(
    document_id: str,
) -> int:
    """
    Re-index one weather document.
    """

    rows = lakebase.run_query(
        """
        SELECT
            id,
            location,
            state,
            district,
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
            issued_at

        FROM weather_documents

        WHERE id = %s
        """,
        (
            document_id,
        ),
    )

    if not rows:
        raise ValueError(
            f"Document not found: {document_id}"
        )

    return embed_document(
        rows[0]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    result = index_documents()

    print(
        "Embedding index complete:"
    )

    print(
        f"Documents processed: "
        f"{result['documents_processed']}"
    )

    print(
        f"Chunks created: "
        f"{result['chunks_created']}"
    )
