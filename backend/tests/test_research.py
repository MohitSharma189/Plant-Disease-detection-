import asyncio

import httpx
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.main import app
from backend.app.services import research as research_service
from backend.app.services.research import ResearchProviderError


client = TestClient(app)


def test_research_returns_cited_publication_metadata(monkeypatch) -> None:
    async def fake_search(crop: str, disease: str, max_results: int) -> dict[str, object]:
        assert (crop, disease, max_results) == ("apple", "apple scab", 3)
        return {
            "provider": "Europe PMC / PubMed-indexed records",
            "query": "apple / apple scab",
            "status": "results",
            "searched_at": "2026-10-31T00:00:00+00:00",
            "results": [{
                "title": "Apple scab research",
                "authors": "A. Researcher",
                "journal": "Plant Pathology",
                "year": "2024",
                "doi": "10.1000/example",
                "url": "https://doi.org/10.1000/example",
                "abstract": "A retrieved abstract.",
            }],
            "message": "Retrieved publication metadata.",
        }

    monkeypatch.setattr(main, "search_publications", fake_search)
    response = client.post("/research", json={"crop": "apple", "disease": "apple scab", "max_results": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "results"
    assert payload["results"][0]["url"] == "https://doi.org/10.1000/example"
    assert payload["results"][0]["abstract"] == "A retrieved abstract."


def test_research_reports_provider_failure(monkeypatch) -> None:
    async def unavailable(*_args, **_kwargs):
        raise ResearchProviderError("Europe PMC could not complete the scholarly search.")

    monkeypatch.setattr(main, "search_publications", unavailable)
    response = client.post("/research", json={"crop": "apple", "disease": "apple scab"})

    assert response.status_code == 502
    assert "Europe PMC" in response.json()["detail"]


def test_research_limits_result_count() -> None:
    response = client.post("/research", json={"crop": "apple", "disease": "apple scab", "max_results": 100})

    assert response.status_code == 422


def test_publication_search_sanitizes_query_and_bounds_results(monkeypatch) -> None:
    requested_query = []
    payload = {
        "resultList": {
            "result": [
                {
                    "title": "Apple scab study",
                    "authorString": "A. Researcher",
                    "journalTitle": "Plant Pathology",
                    "pubYear": 2024,
                    "doi": "10.1000/example",
                    "abstractText": "x" * 6000,
                },
                {"title": "Duplicate record", "doi": "10.1000/example"},
                {"title": "PMID-linked study", "pmid": "12345678"},
            ]
        }
    }

    def handle_request(request: httpx.Request) -> httpx.Response:
        requested_query.append(request.url.params["query"])
        return httpx.Response(200, json=payload)

    original_client = research_service.httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)
    monkeypatch.setattr(
        research_service.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    result = asyncio.run(research_service.search_publications('apple "OR" disease', "scab", 5))

    assert requested_query == ['SRC:MED AND TITLE_ABS:("apple OR disease" AND "scab")']
    assert len(result["results"]) == 2
    assert result["results"][0]["url"] == "https://doi.org/10.1000/example"
    assert len(result["results"][0]["abstract"]) == 5000
    assert result["results"][1]["url"] == "https://europepmc.org/article/MED/12345678"


def test_publication_search_translates_network_failure(monkeypatch) -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider unavailable", request=request)

    original_client = research_service.httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)
    monkeypatch.setattr(
        research_service.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    try:
        asyncio.run(research_service.search_publications("apple", "scab"))
    except ResearchProviderError as exc:
        assert "could not complete" in str(exc)
    else:
        raise AssertionError("Expected the provider failure to be translated.")