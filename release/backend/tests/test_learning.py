from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.services import learning


client = TestClient(app)


def test_feedback_updates_contextual_style_preferences_without_storing_text(monkeypatch, tmp_path) -> None:
    question = "Which paper explains this plant disease?"
    monkeypatch.setattr(settings, "report_dir", tmp_path)

    decision = learning.register_answer(
        question,
        {"predicted_class": "Apple___Apple_scab", "sources": [{"title": "Paper"}]},
    )

    assert decision["category"] == "research"
    assert decision["style"] == "concise"
    assert learning.record_feedback(decision["feedback_id"], helpful=True) is True
    assert learning.record_feedback(decision["feedback_id"], helpful=False) is False

    status = learning.learning_status()
    assert status["ratings"] == 1
    assert status["categories"]["research"]["helpful"] == 1
    assert question.encode() not in (tmp_path / "assistant_learning.sqlite3").read_bytes()

    learning.clear_learning()
    assert learning.learning_status()["ratings"] == 0


def test_learning_categories_distinguish_plant_analysis_and_general_questions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "report_dir", tmp_path)

    plant_decision = learning.register_answer(
        "What does this score mean?",
        {"predicted_class": "Apple___Apple_scab", "sources": []},
    )
    general_decision = learning.register_answer("How do I write a thank-you note?", {})

    assert plant_decision["category"] == "plant_analysis"
    assert general_decision["category"] == "general"
    assert general_decision["style"] == "concise"


def test_positive_rewards_influence_future_style_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "report_dir", tmp_path)
    decisions = [learning.register_answer("Explain a concept", {}) for _ in range(3)]

    assert [decision["style"] for decision in decisions] == ["concise", "step_by_step", "conversational"]
    assert learning.record_feedback(decisions[0]["feedback_id"], helpful=False)
    assert learning.record_feedback(decisions[1]["feedback_id"], helpful=True)
    assert learning.record_feedback(decisions[2]["feedback_id"], helpful=False)

    next_decision = learning.register_answer("Explain another concept", {})

    assert next_decision["style"] == "step_by_step"


def test_learning_feedback_status_and_reset_routes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "report_dir", tmp_path)
    decision = learning.register_answer("How do I interpret this source?", {"sources": ["paper"]})

    feedback = client.post(
        "/learning/feedback",
        json={"learning_id": decision["feedback_id"], "helpful": True},
    )
    status = client.get("/learning/status")

    assert feedback.status_code == 200
    assert feedback.json()["accepted"] is True
    assert status.json()["ratings"] == 1
    assert status.json()["categories"]["research"]["helpful"] == 1

    reset = client.delete("/learning/memory")

    assert reset.status_code == 200
    assert reset.json()["cleared"] is True
    assert client.get("/learning/status").json()["ratings"] == 0