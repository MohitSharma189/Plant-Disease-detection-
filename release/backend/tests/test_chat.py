import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.config import settings
from backend.app.main import app
from backend.app.schemas.analysis import ChatRequest
from backend.app.services import chat as chat_service


client = TestClient(app)


def test_chat_is_disabled_until_provider_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_api_url", "")
    monkeypatch.setattr(settings, "chat_model", "")
    monkeypatch.setattr(settings, "chat_api_key", None)

    status = client.get("/chat/status")
    response = client.post("/chat", json={"question": "नमस्ते, यह स्कोर क्या बताता है?"})

    assert status.status_code == 200
    assert status.json() == {
        "configured": False,
        "ready": False,
        "model": None,
        "message": "Configure a chat endpoint and model in the backend.",
    }
    assert response.status_code == 503
    assert "PLANT_CHAT_API_URL" in response.json()["detail"]


def test_chat_endpoint_returns_provider_answer_without_exposing_credentials(monkeypatch) -> None:
    async def fake_answer(request: ChatRequest) -> dict[str, object]:
        assert request.response_language == "hi-IN"
        return {
            "answer": "यह एक अनिश्चित मॉडल अनुमान है।",
            "provider": "Configured OpenAI-compatible provider",
            "model": "multilingual-model",
            "citations": [],
        }

    monkeypatch.setattr(settings, "chat_api_url", "https://example.test/v1/chat/completions")
    monkeypatch.setattr(settings, "chat_model", "multilingual-model")
    monkeypatch.setattr(main, "answer_question", fake_answer)
    response = client.post(
        "/chat",
        json={"question": "इस बीमारी की संभावना क्या है?", "response_language": "hi-IN"},
    )

    assert response.status_code == 200
    assert response.json()["answer"].startswith("यह")
    assert "api_key" not in response.json()


def test_chat_service_requests_same_language_and_preserves_sources(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def handle_request(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "यह अनुमान है, पुष्टि नहीं।"}}]},
        )

    original_client = chat_service.httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)
    monkeypatch.setattr(settings, "chat_api_url", "https://example.test/v1/chat/completions")
    monkeypatch.setattr(settings, "chat_api_key", "test-secret")
    monkeypatch.setattr(settings, "chat_model", "multilingual-model")
    monkeypatch.setattr(settings, "report_dir", tmp_path)
    monkeypatch.setattr(
        chat_service.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    request = ChatRequest(
        question="क्या यह पक्का रोग है?",
        response_language="hi-IN",
        assistant_notes="Use short explanations and define technical terms.",
        context={
            "predicted_class": "Apple___Apple_scab",
            "model_score": 0.81,
            "score_calibrated": False,
            "sources": [{"title": "Research record", "url": "https://doi.org/10.1000/example"}],
            "history": [
                {"role": "user" if index % 2 == 0 else "assistant", "text": f"turn-{index}"}
                for index in range(10)
            ],
        },
    )

    result = asyncio.run(chat_service.answer_question(request))

    messages = captured["body"]["messages"]
    assert "hi-IN" in messages[0]["content"]
    assert "same language" in messages[0]["content"]
    assert "answer directly" in messages[0]["content"]
    assert "concise" in messages[0]["content"]
    assert "https://doi.org/10.1000/example" in messages[-1]["content"]
    assert "Use short explanations and define technical terms." in messages[-1]["content"]
    assert "image itself is not sent" in messages[0]["content"]
    assert "Identify the user's intent" in messages[0]["content"]
    assert len(messages) == 6
    assert "turn-0" not in json.dumps(messages)
    assert "turn-4" not in json.dumps(messages)
    assert "turn-6" in json.dumps(messages)
    assert captured["body"]["max_tokens"] == 64
    assert captured["authorization"] == "Bearer test-secret"
    assert result["answer"] == "यह अनुमान है, पुष्टि नहीं।"
    assert result["citations"][0]["url"] == "https://doi.org/10.1000/example"
    assert result["learning_category"] == "research"
    assert result["learning_id"]
    assert "Prefer a concise answer" in messages[0]["content"]


def test_chat_status_reports_missing_local_model(monkeypatch) -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    original_client = chat_service.httpx.AsyncClient
    transport = httpx.MockTransport(handle_request)
    monkeypatch.setattr(settings, "chat_api_url", "http://127.0.0.1:11434/v1/chat/completions")
    monkeypatch.setattr(settings, "chat_model", "qwen2.5:7b")
    monkeypatch.setattr(
        chat_service.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    status = asyncio.run(chat_service.chat_provider_status())

    assert status["configured"] is True
    assert status["ready"] is False
    assert "not installed yet" in status["message"]


def test_resolve_local_question_handles_arithmetic_and_alphabet_questions() -> None:
    assert chat_service.resolve_local_question("What is 12 + 8 * 3?") == "36"
    assert chat_service.resolve_local_question("What is 25 divided by 5?") == "5"
    assert chat_service.resolve_local_question("What is fifteen plus twelve?") == "27"
    assert chat_service.resolve_local_question("What is the 5th letter of the alphabet?") == "E"
    assert chat_service.resolve_local_question("What is the 26th letter of the alphabet?") == "Z"
    assert chat_service.resolve_local_question("What is 9 - 4?") == "5"
    assert chat_service.resolve_local_question("What is 7 x 6?") == "42"
    assert chat_service.resolve_local_question("This is not an arithmetic question.") is None


def test_a2a_agent_card_advertises_message_capabilities() -> None:
    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["protocolVersion"] == "0.3.0"
    assert card["url"].endswith("/a2a")
    assert card["capabilities"]["streaming"] is False
    assert card["skills"][0]["id"] == "plant-health-and-general-qa"


def test_a2a_message_send_returns_completed_task(monkeypatch) -> None:
    async def fake_answer(request: ChatRequest) -> dict[str, object]:
        assert request.question == "What is 12 + 8 * 3?"
        assert request.response_language == "hi-IN"
        return {"answer": "36", "provider": "test", "model": "test", "citations": []}

    monkeypatch.setattr(main, "answer_question", fake_answer)
    response = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "message-1",
                    "role": "user",
                    "parts": [{"kind": "text", "text": "What is 12 + 8 * 3?"}],
                    "metadata": {"responseLanguage": "hi-IN"},
                }
            },
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert response.json()["id"] == "request-1"
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    assert result["artifacts"][0]["parts"][0]["text"] == "36"


def test_a2a_rejects_message_without_text_part() -> None:
    response = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "request-2",
            "method": "message/send",
            "params": {"message": {"parts": []}},
        },
    )

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32602


def test_a2a_rejects_oversized_question_as_invalid_params() -> None:
    response = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "request-3",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "a" * 1501}],
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32602