import rag_service


def test_rrf_combines_vector_and_bm25_results():

    vector_results = [
        {
            "document_id": "doc1",
            "similarity": 0.90,
        },
        {
            "document_id": "doc2",
            "similarity": 0.80,
        },
    ]

    bm25_results = [
        {
            "document_id": "doc2",
            "bm25_score": 8.0,
        },
        {
            "document_id": "doc3",
            "bm25_score": 7.0,
        },
    ]

    results = rag_service.reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k=3,
    )

    assert len(results) == 3

    document_ids = [
        result["document_id"]
        for result in results
    ]

    assert "doc1" in document_ids
    assert "doc2" in document_ids
    assert "doc3" in document_ids


def test_shared_document_gets_score_from_both_rankers():

    vector_results = [
        {
            "document_id": "doc1",
        },
    ]

    bm25_results = [
        {
            "document_id": "doc1",
        },
    ]

    results = rag_service.reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result["document_id"] == "doc1"
    assert result["vector_rank"] == 1
    assert result["bm25_rank"] == 1
    assert result["rrf_score"] > 0


def test_rrf_respects_top_k():

    vector_results = [
        {"document_id": "doc1"},
        {"document_id": "doc2"},
        {"document_id": "doc3"},
        {"document_id": "doc4"},
    ]

    bm25_results = []

    results = rag_service.reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k=2,
    )

    assert len(results) == 2


def test_build_context_creates_citations():

    documents = [
        {
            "document_id": "doc1",
            "location": "Kolkata",
            "state": "West Bengal",
            "district": "Kolkata",
            "source": "open-meteo",
            "source_type": "forecast",
            "forecast_date": "2026-08-10",
            "severity": "moderate",
            "headline": "Weather forecast for Kolkata",
            "chunk_text": "Rain is expected.",
            "similarity": 0.82,
            "bm25_score": 4.5,
            "rrf_score": 0.03,
        }
    ]

    context, sources = rag_service.build_context(
        documents
    )

    assert "[S1]" in context
    assert "Kolkata" in context
    assert "Rain is expected." in context

    assert len(sources) == 1
    assert sources[0]["citation"] == "S1"
    assert sources[0]["document_id"] == "doc1"
