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


def test_chat_defaults_to_english_assistant_without_exposing_backend_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_api_url", "")
    monkeypatch.setattr(settings, "chat_model", "")
    monkeypatch.setattr(settings, "chat_api_key", None)

    status = client.get("/chat/status")
    non_english = client.post("/chat", json={"question": "नमस्ते, यह स्कोर क्या बताता है?"})

    assert status.status_code == 200
    assert status.json() == {
        "configured": False,
        "ready": True,
        "model": None,
        "message": "English assistant ready.",
    }
    assert non_english.status_code == 200
    assert "English only" in non_english.json()["answer"]
    assert "PLANT_CHAT_API_URL" not in non_english.json()["answer"]


def test_context_aware_assistant_handles_all_specimen_scenarios(monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_api_url", "")
    monkeypatch.setattr(settings, "chat_model", "")
    monkeypatch.setattr(settings, "chat_api_key", None)

    # 1. Low-confidence Stage A + zero candidate regions
    low_conf = client.post(
        "/chat",
        json={
            "question": "What does this score mean and why are there zero candidate regions?",
            "context": {
                "predicted_class": "Corn_(maize)___healthy",
                "display_label": "Corn (Maize) - Healthy",
                "is_healthy": True,
                "model_score": 0.266,
                "review_threshold": 0.60,
                "score_calibrated": False,
                "review_required": True,
                "stage_a_uncertain": True,
                "screening_mode": "single_leaf",
                "regions_checked": 12,
                "candidate_count": 0,
                "candidate_score_threshold": 0.60,
                "is_inconclusive": True,
            },
        },
    )
    assert low_conf.status_code == 200
    low_text = low_conf.json()["answer"]
    assert "Corn (Maize) - Healthy" in low_text
    assert "26.6%" in low_text
    assert "uncalibrated model score" in low_text.lower()
    assert "uncertain" in low_text.lower()
    assert "12 examined crops" in low_text
    assert "0 candidate regions" in low_text
    assert "never proves that the leaf or plant is healthy" in low_text.lower()

    # 2. High-confidence Stage A + candidate regions
    high_conf = client.post(
        "/chat",
        json={
            "question": "Explain the model score and candidate regions found.",
            "context": {
                "predicted_class": "Apple___Apple_scab",
                "display_label": "Apple - Apple Scab",
                "is_healthy": False,
                "model_score": 0.956,
                "review_threshold": 0.60,
                "score_calibrated": False,
                "review_required": False,
                "stage_a_uncertain": False,
                "screening_mode": "single_leaf",
                "regions_checked": 9,
                "candidate_count": 2,
                "candidate_score_threshold": 0.60,
                "candidate_regions": [
                    {
                        "predicted_class": "Apple___Apple_scab",
                        "display_label": "Apple - Apple Scab",
                        "confidence": 0.987,
                        "needs_review": False,
                    },
                    {
                        "predicted_class": "Apple___Apple_scab",
                        "display_label": "Apple - Apple Scab",
                        "confidence": 0.857,
                        "needs_review": False,
                    },
                ],
                "is_inconclusive": False,
            },
        },
    )
    assert high_conf.status_code == 200
    high_text = high_conf.json()["answer"]
    assert "95.6%" in high_text
    assert "not a confirmed diagnosis" in high_text.lower()
    assert "2 candidate region(s)" in high_text
    assert "98.7%" in high_text
    assert "85.7%" in high_text
    assert "not confirmed disease diagnoses or exact lesion boundaries" in high_text.lower()

    # 3. Stage A vs Stage B + Grad-CAM safeguard
    gradcam_q = client.post(
        "/chat",
        json={
            "question": "How does whole-leaf classification differ from crop screening, and does Grad-CAM segment lesions?",
            "context": {
                "predicted_class": "Tomato___Early_blight",
                "display_label": "Tomato - Early Blight",
                "model_score": 0.607,
                "review_threshold": 0.60,
                "screening_mode": "single_leaf",
                "regions_checked": 9,
                "candidate_count": 1,
                "candidate_regions": [
                    {
                        "predicted_class": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                        "display_label": "Grape - Leaf Blight (Isariopsis Leaf Spot)",
                        "confidence": 0.71,
                        "needs_review": False,
                    }
                ],
                "has_gradcam": True,
            },
        },
    )
    assert gradcam_q.status_code == 200
    gc_text = gradcam_q.json()["answer"]
    assert "stage a" in gc_text.lower() and "stage b" in gc_text.lower()
    assert "does not segment or identify actual disease lesions" in gc_text.lower()

    # 4. Missing specimen context (never invents prediction)
    missing_ctx = client.post(
        "/chat",
        json={"question": "What disease does my plant have?"},
    )
    assert missing_ctx.status_code == 200
    missing_text = missing_ctx.json()["answer"]
    assert "no specimen analysis results are loaded" in missing_text.lower()

    # 5. Empty input rejected cleanly
    empty_q = client.post("/chat", json={"question": ""})
    assert empty_q.status_code == 422


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