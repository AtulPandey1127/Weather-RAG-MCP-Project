"""
Hybrid RAG service for Indian weather question answering.

Pipeline:

    query
      ↓
    query embedding
      ↓
    vector retrieval
      +
    BM25 retrieval
      ↓
    Reciprocal Rank Fusion
      ↓
    context construction
      ↓
    local Ollama LLM
      ↓
    grounded answer + citations

No paid LLM API is required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import lakebase


logger = logging.getLogger(
    __name__
)


# ============================================================================
# Configuration
# ============================================================================

EMBEDDING_MODEL = os.environ.get(
    "WEATHER_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Smaller local model to reduce RAM usage.
OLLAMA_MODEL = os.environ.get(
    "WEATHER_LLM_MODEL",
    "llama3.2:1b",
)

DEFAULT_TOP_K = int(
    os.environ.get(
        "WEATHER_RAG_TOP_K",
        "5",
    )
)

VECTOR_CANDIDATES = int(
    os.environ.get(
        "WEATHER_VECTOR_CANDIDATES",
        "20",
    )
)

BM25_CANDIDATES = int(
    os.environ.get(
        "WEATHER_BM25_CANDIDATES",
        "20",
    )
)

BM25_CORPUS_LIMIT = int(
    os.environ.get(
        "WEATHER_BM25_CORPUS_LIMIT",
        "5000",
    )
)

RRF_K = int(
    os.environ.get(
        "WEATHER_RRF_K",
        "60",
    )
)


# ============================================================================
# Embedding model
# ============================================================================

_embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================================
# Query embedding
# ============================================================================

def embed_query(
    query: str,
) -> list[float]:
    """
    Convert a user query into the embedding space
    used by weather documents.
    """

    vector = _embedding_model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]

    return [
        float(value)
        for value in vector
    ]


# ============================================================================
# Vector retrieval
# ============================================================================

def retrieve_vector(
    query: str,
    top_k: int = VECTOR_CANDIDATES,
) -> list[dict[str, Any]]:
    """
    Retrieve semantically similar weather chunks using pgvector.
    """

    top_k = max(
        1,
        min(
            100,
            int(top_k),
        ),
    )

    vector = embed_query(
        query
    )

    vector_literal = (
        "["
        + ",".join(
            repr(float(value))
            for value in vector
        )
        + "]"
    )

    sql = """
        SELECT
            d.id AS document_id,
            d.location,
            d.state,
            d.district,
            d.source,
            d.source_type,
            d.headline,
            d.forecast_date,
            d.temperature_min_c,
            d.temperature_max_c,
            d.rainfall_mm,
            d.precipitation_probability,
            d.weather_code,
            d.severity,
            d.issued_at,
            e.chunk_index,
            e.chunk_text,

            (
                e.embedding <=> %s::vector
            ) AS distance

        FROM weather_embeddings e

        JOIN weather_documents d
            ON d.id = e.document_id

        ORDER BY distance ASC

        LIMIT %s
    """

    rows = lakebase.run_query(
        sql,
        (
            vector_literal,
            top_k,
        ),
    )

    results = []

    for row in rows:

        distance = row.get(
            "distance"
        )

        similarity = (
            None
            if distance is None
            else 1.0 - float(distance)
        )

        results.append(
            {
                **row,
                "similarity": similarity,
            }
        )

    return results


# ============================================================================
# BM25 retrieval
# ============================================================================

def _tokenize(
    text: str,
) -> list[str]:
    """
    Lightweight tokenizer for BM25.
    """

    return [
        token.lower()
        for token in text.split()
        if token.strip()
    ]


def retrieve_bm25(
    query: str,
    limit: int = BM25_CANDIDATES,
) -> list[dict[str, Any]]:
    """
    Retrieve keyword-relevant weather chunks using BM25.
    """

    corpus_sql = """
        SELECT
            d.id AS document_id,
            d.location,
            d.state,
            d.district,
            d.source,
            d.source_type,
            d.headline,
            d.forecast_date,
            d.temperature_min_c,
            d.temperature_max_c,
            d.rainfall_mm,
            d.precipitation_probability,
            d.weather_code,
            d.severity,
            d.issued_at,
            e.chunk_index,
            e.chunk_text

        FROM weather_embeddings e

        JOIN weather_documents d
            ON d.id = e.document_id

        ORDER BY d.forecast_date DESC NULLS LAST

        LIMIT %s
    """

    rows = lakebase.run_query(
        corpus_sql,
        (
            BM25_CORPUS_LIMIT,
        ),
    )

    if not rows:
        return []

    corpus = []

    for row in rows:

        text = " ".join(
            [
                str(
                    row.get("location")
                    or ""
                ),
                str(
                    row.get("state")
                    or ""
                ),
                str(
                    row.get("district")
                    or ""
                ),
                str(
                    row.get("headline")
                    or ""
                ),
                str(
                    row.get("source_type")
                    or ""
                ),
                str(
                    row.get("severity")
                    or ""
                ),
                str(
                    row.get("chunk_text")
                    or ""
                ),
            ]
        )

        corpus.append(
            _tokenize(text)
        )

    if not corpus:
        return []

    bm25 = BM25Okapi(
        corpus
    )

    query_tokens = _tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        if len(results) >= limit:
            break

        score = float(
            scores[index]
        )

        if score <= 0:
            continue

        row = dict(
            rows[index]
        )

        row[
            "bm25_score"
        ] = score

        results.append(
            row
        )

    return results


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Combine vector and BM25 rankings using RRF.
    """

    fused: dict[
        str,
        dict[str, Any]
    ] = {}

    # ------------------------------------------------------------------------
    # Vector ranking
    # ------------------------------------------------------------------------

    for rank, document in enumerate(
        vector_results,
        start=1,
    ):

        document_id = document.get(
            "document_id"
        )

        if not document_id:
            continue

        if document_id not in fused:

            fused[
                document_id
            ] = {
                **document,
                "vector_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }

        fused[
            document_id
        ][
            "vector_rank"
        ] = rank

        fused[
            document_id
        ][
            "rrf_score"
        ] += (
            1.0
            / (
                RRF_K
                + rank
            )
        )

    # ------------------------------------------------------------------------
    # BM25 ranking
    # ------------------------------------------------------------------------

    for rank, document in enumerate(
        bm25_results,
        start=1,
    ):

        document_id = document.get(
            "document_id"
        )

        if not document_id:
            continue

        if document_id not in fused:

            fused[
                document_id
            ] = {
                **document,
                "vector_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }

        fused[
            document_id
        ][
            "bm25_rank"
        ] = rank

        fused[
            document_id
        ][
            "rrf_score"
        ] += (
            1.0
            / (
                RRF_K
                + rank
            )
        )

    results = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return results[
        :max(
            1,
            min(
                20,
                int(top_k),
            ),
        )
    ]


# ============================================================================
# Hybrid retrieval
# ============================================================================

def retrieve_weather(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Hybrid weather retrieval.

    Combines:

        Dense vector retrieval
        BM25 lexical retrieval
        Reciprocal Rank Fusion
    """

    top_k = max(
        1,
        min(
            20,
            int(top_k),
        ),
    )

    vector_results = retrieve_vector(
        query,
        VECTOR_CANDIDATES,
    )

    bm25_results = retrieve_bm25(
        query,
        BM25_CANDIDATES,
    )

    return reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k,
    )


# ============================================================================
# Context construction
# ============================================================================

def build_context(
    documents: list[dict[str, Any]],
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    Convert retrieved documents into an LLM context block.
    """

    context_parts = []

    sources = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        citation_id = (
            f"S{index}"
        )

        location = (
            document.get(
                "location"
            )
            or "Unknown location"
        )

        state = (
            document.get(
                "state"
            )
            or "Unknown state"
        )

        district = (
            document.get(
                "district"
            )
            or "Unknown district"
        )

        headline = (
            document.get(
                "headline"
            )
            or "Weather information"
        )

        source = (
            document.get(
                "source"
            )
            or "unknown"
        )

        source_type = (
            document.get(
                "source_type"
            )
            or "unknown"
        )

        forecast_date = (
            document.get(
                "forecast_date"
            )
            or "unknown"
        )

        severity = (
            document.get(
                "severity"
            )
            or "unknown"
        )

        chunk_text = (
            document.get(
                "chunk_text"
            )
            or ""
        )

        context_parts.append(
            f"""
[{citation_id}]
Location: {location}
State: {state}
District: {district}
Source: {source}
Source type: {source_type}
Forecast date: {forecast_date}
Severity: {severity}
Headline: {headline}

Weather information:
{chunk_text}
""".strip()
        )

        sources.append(
            {
                "citation": citation_id,
                "document_id": document.get(
                    "document_id"
                ),
                "location": location,
                "state": state,
                "district": district,
                "source": source,
                "source_type": source_type,
                "forecast_date": forecast_date,
                "severity": severity,
                "similarity": document.get(
                    "similarity"
                ),
                "bm25_score": document.get(
                    "bm25_score"
                ),
                "rrf_score": document.get(
                    "rrf_score"
                ),
            }
        )

    return (
        "\n\n---\n\n".join(
            context_parts
        ),
        sources,
    )


# ============================================================================
# Grounded generation
# ============================================================================

SYSTEM_PROMPT = """
You are an Indian weather intelligence assistant.

Answer questions using ONLY the weather information
provided in the retrieved context.

Rules:

1. Do not invent weather facts.

2. Do not use outside knowledge to fill missing
   weather information.

3. If the retrieved context is insufficient,
   explicitly say that the available weather data
   is insufficient.

4. Cite factual claims using source identifiers
   such as [S1], [S2], etc.

5. Prefer recent information when multiple weather
   records are available.

6. Clearly distinguish current conditions from forecasts.

7. Do not describe an Open-Meteo condition severity
   as an official government warning.

8. Do not claim that a condition is an official IMD
   warning unless the retrieved source explicitly says so.

9. Keep answers concise and useful.

10. Use Celsius and kilometres per hour unless the
    retrieved data specifies otherwise.
"""


def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generate a grounded answer using local Ollama.

    No paid API key is required.
    """

    user_prompt = f"""
User question:

{query}

Retrieved weather context:

{context}

Answer the user's question using ONLY the
retrieved weather context.

Include citations such as [S1] and [S2].
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )


# ============================================================================
# Complete RAG pipeline
# ============================================================================

def answer_weather_question(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Execute the complete hybrid RAG pipeline.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    query = query.strip()

    documents = retrieve_weather(
        query=query,
        top_k=top_k,
    )

    if not documents:

        return {
            "answer": (
                "I could not find relevant "
                "weather information in the "
                "knowledge base."
            ),
            "sources": [],
            "retrieved_documents": 0,
            "model": OLLAMA_MODEL,
            "retrieval": "hybrid",
        }

    context, sources = (
        build_context(
            documents
        )
    )

    answer = generate_answer(
        query=query,
        context=context,
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_documents": len(
            documents
        ),
        "model": OLLAMA_MODEL,
        "retrieval": "hybrid",
    }


# ============================================================================
# Simple CLI test
# ============================================================================

if __name__ == "__main__":

    print(
        "RAG service configuration:"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Ollama model: "
        f"{OLLAMA_MODEL}"
    )

    print(
        f"Default top-k: "
        f"{DEFAULT_TOP_K}"
    )

    print(
        "RAG service loaded successfully."
    )
