import app


def test_healthz():

    client = app.app.test_client()

    response = client.get(
        "/healthz"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert (
        data["service"]
        == "indian-weather-rag"
    )


def test_weather_ask_missing_query():

    client = app.app.test_client()

    response = client.post(
        "/weather/ask",
        json={},
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


def test_weather_ask_empty_query():

    client = app.app.test_client()

    response = client.post(
        "/weather/ask",
        json={
            "query": ""
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert (
    data["error"]
    == "Missing or invalid 'query' in request body"
)


def test_weather_ask_invalid_top_k():

    client = app.app.test_client()

    response = client.post(
        "/weather/ask",
        json={
            "query": "Weather in Kolkata?",
            "top_k": "abc",
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert (
        "'top_k' must be an integer"
        in data["error"]
    )


def test_weather_ask_success(monkeypatch):

    def fake_answer(
        query,
        top_k,
    ):
        return {
            "answer": (
                "Kolkata may receive rain. [S1]"
            ),
            "sources": [
                {
                    "citation": "S1",
                    "location": "Kolkata",
                }
            ],
            "retrieved_documents": 1,
            "model": "llama3.2:1b",
            "retrieval": "hybrid",
        }

    monkeypatch.setattr(
        app.rag_service,
        "answer_weather_question",
        fake_answer,
    )

    client = app.app.test_client()

    response = client.post(
        "/weather/ask",
        json={
            "query": (
                "What is the weather "
                "forecast for Kolkata?"
            ),
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert (
        data["answer"]
        == "Kolkata may receive rain. [S1]"
    )

    assert data["retrieved_documents"] == 1
    assert data["retrieval"] == "hybrid"


def test_weather_ask_database_error(
    monkeypatch,
):

    def fake_answer(
        query,
        top_k,
    ):
        raise RuntimeError(
            "database unavailable"
        )

    monkeypatch.setattr(
        app.rag_service,
        "answer_weather_question",
        fake_answer,
    )

    client = app.app.test_client()

    response = client.post(
        "/weather/ask",
        json={
            "query": "Weather in Kolkata?"
        },
    )

    assert response.status_code == 500

    data = response.get_json()

    assert (
        data["error"]
        == "Failed to generate weather answer"
    )


def test_unknown_endpoint():

    client = app.app.test_client()

    response = client.get(
        "/does-not-exist"
    )

    assert response.status_code == 404

    data = response.get_json()

    assert (
        data["error"]
        == "Endpoint not found"
    )
