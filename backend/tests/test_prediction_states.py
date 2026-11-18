from io import BytesIO
from typing import Any
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.models import format_class_metadata, ModelRegistry, ModelUnavailableError
from backend.app.services.screening import screen_plant_image


client = TestClient(app)


class MockRegistry:
    """Configurable mock registry for controlling prediction scores and classes."""
    def __init__(
        self,
        predicted_class: str = "Tomato___Early_blight",
        confidence: float = 0.85,
        review_threshold: float = 0.60,
        calibration_report: str | None = None,
        crop_responses: list[dict[str, Any]] | None = None,
        should_fail: bool = False,
    ) -> None:
        self.classifier = object()
        self.class_names = [
            "Apple___Apple_scab",
            "Apple___healthy",
            "Cherry_(including_sour)___healthy",
            "Tomato___Early_blight",
            "Tomato___healthy",
        ]
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.review_threshold = review_threshold
        self.calibration_report = calibration_report
        self.crop_responses = crop_responses or []
        self._crop_index = 0
        self.should_fail = should_fail
        self.image_size = 224

    def predict_class(self, image: Image.Image) -> dict[str, Any]:
        if self.should_fail:
            raise ModelUnavailableError("Simulated neural network runtime inference failure.")

        if self.crop_responses and self._crop_index < len(self.crop_responses):
            resp = self.crop_responses[self._crop_index]
            self._crop_index += 1
            pred_class = resp.get("predicted_class", self.predicted_class)
            conf = resp.get("confidence", self.confidence)
        else:
            pred_class = self.predicted_class
            conf = self.confidence

        needs_review = conf < self.review_threshold
        is_uncertain = needs_review
        meta = format_class_metadata(pred_class)
        conf_calibrated = self.calibration_report is not None

        if is_uncertain:
            pred_state = "LOW_CONFIDENCE_UNCERTAIN"
            pred_status = "Uncertain Prediction"
            state_tone = "amber"
            headline = f"Uncertain Prediction · Tentative match: {meta['display_label']}"
            explanation = (
                f"The uncalibrated model score ({conf:.1%}) is below the operational review threshold ({self.review_threshold:.0%}). "
                "The model could not produce a sufficiently reliable classification for this specimen. "
                "Tentative predictions must not be treated as a confirmed diagnosis or confirmation of plant health."
            )
            recommendation = (
                "Treat this result as tentative and seek confirmation from a qualified plant diagnostic clinic "
                "or local agricultural extension service. Consider retaking a sharper, well-lit close-up."
            )
        else:
            pred_state = "HIGH_CONFIDENCE"
            pred_status = "Model Prediction"
            state_tone = "good"
            headline = f"Model Prediction · {meta['display_label']}"
            explanation = (
                f"The uncalibrated model score ({conf:.1%}) meets the operational review threshold ({self.review_threshold:.0%}). "
                "This output represents an automated machine-learning screening prediction, not a certified clinical or laboratory diagnosis."
            )
            recommendation = (
                "Qualified agronomic verification is recommended before taking field-management or chemical actions."
            )

        return {
            "predicted_class": pred_class,
            "class_scores": {pred_class: conf},
            "confidence": conf,
            "needs_review": needs_review,
            "is_uncertain": is_uncertain,
            "review_threshold": self.review_threshold,
            "confidence_calibrated": conf_calibrated,
            "prediction_state": pred_state,
            "prediction_status": pred_status,
            "state_tone": state_tone,
            "headline": headline,
            "explanation": explanation,
            "recommendation": recommendation,
            "score_label": "Uncalibrated Model Score",
            "display_label": meta["display_label"],
            "crop_name": meta["crop"],
            "condition_name": meta["condition"],
            "is_healthy": meta["is_healthy"],
        }


def _create_jpeg_bytes(width: int = 256, height: int = 256, color: str = "green") -> bytes:
    buffer = BytesIO()
    img = Image.new("RGB", (width, height), color)
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Scenario 1: High-confidence healthy prediction
# -----------------------------------------------------------------------------
def test_scenario_1_high_confidence_healthy() -> None:
    registry = MockRegistry(predicted_class="Apple___healthy", confidence=0.92)
    img = Image.new("RGB", (256, 256), "green")
    res = screen_plant_image(img, registry, "single_leaf")

    pred = res["overall_prediction"]
    assert pred is not None
    assert pred["prediction_state"] == "HIGH_CONFIDENCE"
    assert pred["prediction_status"] == "Model Prediction"
    assert pred["state_tone"] == "good"
    assert pred["is_uncertain"] is False
    assert pred["is_healthy"] is True
    assert pred["display_label"] == "Apple - Healthy"
    assert pred["score_label"] == "Uncalibrated Model Score"
    assert pred["confidence"] == 0.92
    assert "meets the operational review threshold" in pred["explanation"]


# -----------------------------------------------------------------------------
# Scenario 2: High-confidence diseased prediction
# -----------------------------------------------------------------------------
def test_scenario_2_high_confidence_diseased() -> None:
    registry = MockRegistry(predicted_class="Tomato___Early_blight", confidence=0.88)
    img = Image.new("RGB", (256, 256), "green")
    res = screen_plant_image(img, registry, "single_leaf")

    pred = res["overall_prediction"]
    assert pred is not None
    assert pred["prediction_state"] == "HIGH_CONFIDENCE"
    assert pred["prediction_status"] == "Model Prediction"
    assert pred["state_tone"] == "good"
    assert pred["is_uncertain"] is False
    assert pred["is_healthy"] is False
    assert pred["display_label"] == "Tomato - Early Blight"
    assert pred["score_label"] == "Uncalibrated Model Score"
    assert pred["confidence"] == 0.88


# -----------------------------------------------------------------------------
# Scenario 3: Low-confidence healthy prediction (Crucial User Issue)
# -----------------------------------------------------------------------------
def test_scenario_3_low_confidence_healthy() -> None:
    # 28% Cherry healthy (matching user screenshot)
    registry = MockRegistry(predicted_class="Cherry_(including_sour)___healthy", confidence=0.28)
    img = Image.new("RGB", (256, 256), "green")
    res = screen_plant_image(img, registry, "single_leaf")

    pred = res["overall_prediction"]
    assert pred is not None
    assert pred["prediction_state"] == "LOW_CONFIDENCE_UNCERTAIN"
    assert pred["prediction_status"] == "Uncertain Prediction"
    assert pred["state_tone"] == "amber"
    assert pred["is_uncertain"] is True
    assert pred["needs_review"] is True
    assert pred["is_healthy"] is True
    assert pred["display_label"] == "Cherry (Sour) - Healthy"
    assert pred["score_label"] == "Uncalibrated Model Score"
    assert "Tentative match: Cherry (Sour) - Healthy" in pred["headline"]
    assert "below the operational review threshold" in pred["explanation"]
    assert "not be treated as a confirmed diagnosis or confirmation of plant health" in pred["explanation"]


# -----------------------------------------------------------------------------
# Scenario 4: Low-confidence diseased prediction
# -----------------------------------------------------------------------------
def test_scenario_4_low_confidence_diseased() -> None:
    registry = MockRegistry(predicted_class="Apple___Apple_scab", confidence=0.45)
    img = Image.new("RGB", (256, 256), "green")
    res = screen_plant_image(img, registry, "single_leaf")

    pred = res["overall_prediction"]
    assert pred is not None
    assert pred["prediction_state"] == "LOW_CONFIDENCE_UNCERTAIN"
    assert pred["prediction_status"] == "Uncertain Prediction"
    assert pred["state_tone"] == "amber"
    assert pred["is_uncertain"] is True
    assert pred["needs_review"] is True
    assert pred["display_label"] == "Apple - Apple Scab"
    assert "Tentative match" in pred["headline"]


# -----------------------------------------------------------------------------
# Scenario 5: No candidate screening regions
# -----------------------------------------------------------------------------
def test_scenario_5_no_candidate_screening_regions() -> None:
    # All windows return healthy or scores < 0.60
    registry = MockRegistry(predicted_class="Tomato___healthy", confidence=0.90)
    img = Image.new("RGB", (256, 256), "green")
    res = screen_plant_image(img, registry, "single_leaf")

    assert res["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
    assert res["screening_status"] == "No Candidate Regions Identified"
    assert res["screening_tone"] == "neutral"
    assert res["candidate_count"] == 0
    assert len(res["candidate_regions"]) == 0
    assert res["regions_checked"] > 0
    assert "Lack of candidate screening regions does not establish that the leaf or plant is healthy" in res["safety_disclaimer"]


# -----------------------------------------------------------------------------
# Scenario 6: Multiple candidate screening regions
# -----------------------------------------------------------------------------
def test_scenario_6_multiple_candidate_screening_regions() -> None:
    # Distinct crops returning >= 0.60 disease
    crops = [
        {"predicted_class": "Tomato___Early_blight", "confidence": 0.85},
        {"predicted_class": "Apple___Apple_scab", "confidence": 0.78},
        {"predicted_class": "Tomato___Early_blight", "confidence": 0.65},
    ]
    registry = MockRegistry(
        predicted_class="Tomato___Early_blight",
        confidence=0.85,
        crop_responses=crops,
    )
    img = Image.new("RGB", (320, 320), "green")
    res = screen_plant_image(img, registry, "whole_plant")

    assert res["screening_state"] == "CANDIDATES_IDENTIFIED"
    assert "Candidate Region(s) Identified" in res["screening_status"]
    assert res["screening_tone"] == "info"
    assert res["candidate_count"] >= 1
    first_candidate = res["candidate_regions"][0]
    assert first_candidate["display_label"] in ["Tomato - Early Blight", "Apple - Apple Scab"]
    assert first_candidate["score_label"] == "Uncalibrated Model Score"
    assert "box" in first_candidate
    assert "image" in first_candidate


# -----------------------------------------------------------------------------
# Scenario 7: Invalid image upload (empty / corrupted)
# -----------------------------------------------------------------------------
def test_scenario_7_invalid_image_upload() -> None:
    # Empty file
    resp_empty = client.post("/screen-plant", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp_empty.status_code == 422
    assert "empty" in resp_empty.json()["detail"].lower()

    # Corrupted text pretending to be JPEG
    resp_corrupt = client.post("/screen-plant", files={"file": ("corrupt.jpg", b"NOT AN IMAGE FILE CONTENT", "image/jpeg")})
    assert resp_corrupt.status_code == 422
    assert "not a valid image" in resp_corrupt.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Scenario 8: Unsupported image format / arbitrary text
# -----------------------------------------------------------------------------
def test_scenario_8_unsupported_image_format() -> None:
    resp_txt = client.post(
        "/predict",
        files={"file": ("data.txt", b"plain text payload cannot decode", "text/plain")},
    )
    assert resp_txt.status_code == 422
    assert "not a valid image" in resp_txt.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Scenario 9: Backend inference failure handling
# -----------------------------------------------------------------------------
def test_scenario_9_backend_inference_failure() -> None:
    registry = MockRegistry(should_fail=True)
    img = Image.new("RGB", (256, 256), "green")
    with pytest.raises(ModelUnavailableError, match="Simulated neural network runtime inference failure"):
        screen_plant_image(img, registry, "single_leaf")


# -----------------------------------------------------------------------------
# Scenario 10: Different image dimensions
# -----------------------------------------------------------------------------
def test_scenario_10_different_image_dimensions() -> None:
    registry = MockRegistry(predicted_class="Tomato___Early_blight", confidence=0.75)
    
    # Non-square rectangular dimensions: landscape, portrait, and large
    dimensions = [(640, 360), (360, 640), (800, 600), (256, 512)]
    for w, h in dimensions:
        img = Image.new("RGB", (w, h), "green")
        res = screen_plant_image(img, registry, "single_leaf")
        assert res["regions_checked"] > 0
        assert res["overlay"].startswith("data:image/jpeg;base64,")
        assert res["screening_state"] in ["NO_CANDIDATES_IDENTIFIED", "CANDIDATES_IDENTIFIED"]

