import rag_service


def test_answer_weather_question():

    fake_documents = [
        {
            "document_id": "doc1",
            "location": "Kolkata",
            "state": "West Bengal",
            "district": "Kolkata",
            "source": "open-meteo",
            "source_type": "forecast",
            "forecast_date": "2026-08-10",
            "severity": "moderate",
            "headline": "Kolkata weather",
            "chunk_text": (
                "Temperature will be 32 °C "
                "with a high chance of rain."
            ),
            "similarity": 0.85,
            "bm25_score": 5.0,
            "rrf_score": 0.03,
        }
    ]

    original_retrieve = (
        rag_service.retrieve_weather
    )

    original_generate = (
        rag_service.generate_answer
    )

    try:

        rag_service.retrieve_weather = (
            lambda query, top_k: fake_documents
        )

        rag_service.generate_answer = (
            lambda query, context:
            "Kolkata may receive rain tomorrow. [S1]"
        )

        result = (
            rag_service.answer_weather_question(
                "Will it rain in Kolkata tomorrow?",
                top_k=5,
            )
        )

        assert "answer" in result
        assert "sources" in result
        assert "retrieved_documents" in result
        assert "model" in result
        assert "retrieval" in result

        assert (
            result["answer"]
            == "Kolkata may receive rain tomorrow. [S1]"
        )

        assert result["retrieved_documents"] == 1
        assert result["retrieval"] == "hybrid"
        assert len(result["sources"]) == 1

    finally:

        rag_service.retrieve_weather = (
            original_retrieve
        )

        rag_service.generate_answer = (
            original_generate
        )


def test_empty_query_is_rejected():

    try:

        rag_service.answer_weather_question(
            "",
            top_k=5,
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError as exc:

        assert (
            "Query cannot be empty"
            in str(exc)
        )


def test_no_documents_returns_fallback():

    original_retrieve = (
        rag_service.retrieve_weather
    )

    try:

        rag_service.retrieve_weather = (
            lambda query, top_k: []
        )

        result = (
            rag_service.answer_weather_question(
                "Unknown weather question",
                top_k=5,
            )
        )

        assert (
            result["retrieved_documents"]
            == 0
        )

        assert result["sources"] == []

        assert (
            "could not find relevant"
            in result["answer"].lower()
        )

        assert result["retrieval"] == "hybrid"

    finally:

        rag_service.retrieve_weather = (
            original_retrieve
        )
