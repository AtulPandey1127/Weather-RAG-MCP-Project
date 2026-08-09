"""
RAG service for weather question answering.

Pipeline:

query
  -> embedding
  -> vector retrieval
  -> context construction
  -> LLM
  -> grounded answer + citations
"""

import logging
import os
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer

import lakebase

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get(
    "WEATHER_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

LLM_MODEL = os.environ.get(
    "WEATHER_LLM_MODEL",
    "gpt-5-mini",
)

DEFAULT_TOP_K = int(os.environ.get("WEATHER_RAG_TOP_K", "5"))

_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
_openai_client = OpenAI()


def embed_query(query: str) -> list[float]:
    """Convert the user's question into the same vector space as documents."""

    vector = _embedding_model.encode(
        [query],
        show_progress_bar=False,
    )[0]

    return [float(value) for value in vector]


def retrieve_weather(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve the most semantically relevant weather chunks.
    """

    top_k = max(1, min(20, int(top_k)))

    vector = embed_query(query)

    vector_literal = (
        "["
        + ",".join(repr(float(value)) for value in vector)
        + "]"
    )

    sql = """
        SELECT
            d.id AS document_id,
            d.location,
            d.source_type,
            d.headline,
            d.issued_at,
            e.chunk_index,
            e.chunk_text,
            (e.embedding <=> %s::vector) AS distance
        FROM weather_embeddings e
        JOIN weather_documents d
            ON d.id = e.document_id
        ORDER BY distance ASC
        LIMIT %s
    """

    rows = lakebase.run_query(
        sql,
        (vector_literal, top_k),
    )

    results = []

    for row in rows:
        distance = row.get("distance")

        similarity = (
            None
            if distance is None
            else 1.0 - float(distance)
        )

        results.append(
            {
                "document_id": row.get("document_id"),
                "location": row.get("location"),
                "source_type": row.get("source_type"),
                "headline": row.get("headline"),
                "issued_at": row.get("issued_at"),
                "chunk_index": row.get("chunk_index"),
                "chunk_text": row.get("chunk_text"),
                "similarity": similarity,
            }
        )

    return results


def build_context(
    documents: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert retrieved documents into an LLM context block.

    Returns:
        context text
        source metadata
    """

    context_parts = []
    sources = []

    for index, document in enumerate(documents, start=1):

        citation_id = f"S{index}"

        location = document.get("location") or "Unknown location"
        headline = document.get("headline") or "Weather information"
        source_type = document.get("source_type") or "unknown"
        chunk_text = document.get("chunk_text") or ""

        context_parts.append(
            f"""
[{citation_id}]
Location: {location}
Source type: {source_type}
Headline: {headline}

Weather information:
{chunk_text}
""".strip()
        )

        sources.append(
            {
                "citation": citation_id,
                "document_id": document.get("document_id"),
                "location": location,
                "source_type": source_type,
                "headline": headline,
                "similarity": document.get("similarity"),
            }
        )

    return "\n\n---\n\n".join(context_parts), sources


SYSTEM_PROMPT = """
You are a weather intelligence assistant.

Your job is to answer questions using ONLY the weather information
provided in the retrieved context.

Rules:

1. Do not invent weather facts.
2. Do not use outside knowledge to fill missing weather information.
3. If the retrieved context does not contain enough information,
   explicitly say that the available weather data is insufficient.
4. Cite factual claims using the source identifiers provided in the
   context, for example [S1] or [S2].
5. Prefer the most relevant and recent information when multiple
   sources disagree.
6. Distinguish between forecasts and active weather alerts.
7. Do not claim that weather information is live unless the retrieved
   data indicates that it is current.
8. Keep the answer concise but useful.
"""


def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generate a grounded answer from retrieved weather context.
    """

    user_prompt = f"""
User question:

{query}

Retrieved weather context:

{context}

Answer the user's question using only this context.
Include citations such as [S1] and [S2] for factual claims.
"""

    response = _openai_client.responses.create(
        model=LLM_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    return response.output_text.strip()


def answer_weather_question(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Complete RAG pipeline.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query = query.strip()

    documents = retrieve_weather(
        query=query,
        top_k=top_k,
    )

    if not documents:
        return {
            "answer": (
                "I could not find relevant weather information "
                "in the knowledge base."
            ),
            "sources": [],
            "retrieved_documents": 0,
            "model": LLM_MODEL,
        }

    context, sources = build_context(documents)

    answer = generate_answer(
        query=query,
        context=context,
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_documents": len(documents),
        "model": LLM_MODEL,
    }
