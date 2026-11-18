import logging

from PIL import Image
import pytest

from backend.app.services.screening import _windows, screen_plant_image


class StubRegistry:
    classifier = object()
    class_names = ["Tomato___healthy", "Tomato___Early_blight"]

    def __init__(self, predicted_class: str = "Tomato___Early_blight", confidence: float = 0.91) -> None:
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.calls = 0

    def predict_class(self, image: Image.Image) -> dict[str, object]:
        self.calls += 1
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_scores": {self.predicted_class: self.confidence},
        }


def test_single_leaf_returns_candidate_crops_and_original_overlay() -> None:
    registry = StubRegistry()
    result = screen_plant_image(Image.new("RGB", (256, 256), "green"), registry, "single_leaf")

    assert result["mode"] == "single_leaf"
    assert result["overall_prediction"]["predicted_class"] == "Tomato___Early_blight"
    assert result["regions_checked"] > 1
    assert result["candidate_count"] > 0
    assert result["overlay"].startswith("data:image/jpeg;base64,")
    assert all(region["image"].startswith("data:image/jpeg;base64,") for region in result["candidate_regions"])
    assert all(region["severity"] == "Not assessed from image" for region in result["candidate_regions"])


def test_healthy_predictions_are_not_shown_as_affected_candidates() -> None:
    result = screen_plant_image(
        Image.new("RGB", (320, 320), "green"),
        StubRegistry("Tomato___healthy", 0.94),
        "whole_plant",
    )

    assert result["candidate_count"] == 0
    assert result["candidate_regions"] == []
    assert "No candidates does not prove" in result["limitations"][2]


def test_low_score_is_not_reported_as_candidate() -> None:
    result = screen_plant_image(
        Image.new("RGB", (320, 320), "green"),
        StubRegistry("Tomato___Early_blight", 0.59),
        "whole_plant",
    )

    assert result["candidate_count"] == 0
    assert result["candidate_score_threshold"] == 0.60


def test_calibrated_scores_and_review_threshold_are_reported_without_claiming_ood() -> None:
    registry = StubRegistry(confidence=0.91)
    registry.calibration_report = "held-out calibration report"
    registry.review_threshold = 0.95

    result = screen_plant_image(Image.new("RGB", (256, 256), "green"), registry, "single_leaf")

    assert result["confidence_calibrated"] is True
    assert result["review_required"] is True
    assert result["unknown_detection_validated"] is False
    assert result["candidate_regions"][0]["confidence_calibrated"] is True
    assert result["candidate_regions"][0]["needs_review"] is True
    assert "not independent botanical" in result["candidate_regions"][0]["explanation"]
    assert "temperature-scaled" in result["candidate_regions"][0]["visible_evidence"]


def test_invalid_screening_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Mode must be"):
        screen_plant_image(Image.new("RGB", (256, 256)), StubRegistry(), "unsupported")


def test_wide_plant_screen_samples_across_full_image() -> None:
    boxes = _windows(Image.new("RGB", (4096, 256)), "whole_plant")

    assert len(boxes) <= 36
    assert min(box[0] for box in boxes) == 0
    assert max(box[2] for box in boxes) == 4096


def test_zero_candidate_safety_disclaimer_has_no_duplicate_heading_prefix() -> None:
    result = screen_plant_image(
        Image.new("RGB", (320, 320), "green"),
        StubRegistry("Tomato___Early_blight", 0.35),
        "single_leaf",
    )

    assert result["candidate_count"] == 0
    assert result["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
    assert result["screening_tone"] == "neutral"
    assert result["screening_message"] == "No candidate regions crossed the screening threshold in the examined image crops."
    assert result["screening_explanation"] == (
        "This screening result is inconclusive because no examined crop exceeded the operational screening threshold."
    )
    assert "sharper, well-lit image" in result["screening_recommendation"]
    assert not result["safety_disclaimer"].lower().startswith("important safety notice")
    assert result["safety_disclaimer"] == (
        "Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free. "
        "Candidate screening identifies image-crop regions whose disease-class scores exceed the operational threshold. "
        "These regions are candidate areas for review, not confirmed disease or lesion locations."
    )


def test_stage_b_diagnostic_logging_records_every_crop_and_suppression_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SequenceRegistry(StubRegistry):
        def __init__(self, responses: list[tuple[str, float]]) -> None:
            super().__init__()
            self._responses = responses
            self._index = 0

        def predict_class(self, image: Image.Image) -> dict[str, object]:
            label, score = self._responses[min(self._index, len(self._responses) - 1)]
            self._index += 1
            return {
                "predicted_class": label,
                "confidence": score,
                "class_scores": {label: score},
            }

    registry = SequenceRegistry(
        [
            ("Tomato___healthy", 0.88),
            ("Tomato___Early_blight", 0.41),
            ("Tomato___Early_blight", 0.82),
            ("Tomato___Early_blight", 0.79),
        ]
    )

    with caplog.at_level(logging.INFO, logger="backend.app.services.screening"):
        result = screen_plant_image(Image.new("RGB", (256, 256), "green"), registry, "whole_plant")

    crop_logs = [
        record.message
        for record in caplog.records
        if record.name == "backend.app.services.screening" and "Stage B crop " in record.message
    ]
    evaluated_logs = [msg for msg in crop_logs if "box=" in msg]
    assert len(evaluated_logs) == result["regions_checked"]
    assert any("suppressed_by=healthy_class_filter" in msg and "Tomato___healthy" in msg for msg in evaluated_logs)
    assert any("suppressed_by=below_threshold" in msg and "passed_threshold=False" in msg for msg in evaluated_logs)
    assert any("suppressed_by=none" in msg and "passed_threshold=True" in msg for msg in evaluated_logs)
    assert any("Stage B screening completed:" in record.message for record in caplog.records)


def test_undersized_image_raises_instead_of_reporting_zero_candidates() -> None:
    with pytest.raises(ValueError, match="No valid screening crops"):
        screen_plant_image(Image.new("RGB", (64, 64), "green"), StubRegistry(), "single_leaf")


def test_exact_threshold_boundary_and_nms_suppression(caplog: pytest.LogCaptureFixture) -> None:
    img = Image.new("RGB", (320, 320), "green")

    below = screen_plant_image(img, StubRegistry("Tomato___Early_blight", 0.5999), "whole_plant")
    assert below["candidate_count"] == 0
    assert below["screening_state"] == "NO_CANDIDATES_IDENTIFIED"

    with caplog.at_level(logging.INFO, logger="backend.app.services.screening"):
        exact = screen_plant_image(img, StubRegistry("Tomato___Early_blight", 0.60), "whole_plant")
    assert exact["candidate_count"] > 0
    assert exact["screening_state"] == "CANDIDATES_IDENTIFIED"
    assert any("suppressed_by=nms" in record.message for record in caplog.records)
