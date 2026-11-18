import os
import sys
import io
import time
import json
import hashlib
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path("backend").resolve()))

from starlette.testclient import TestClient
from app.main import app
from app.services.models import model_registry
from app.config import settings

def run_smoke_test():
    print("=" * 70)
    print("PHASE 6A.6: FINAL APPLICATION VERIFICATION SMOKE TEST")
    print("=" * 70)

    # 1. Model Registry and Checkpoint Verification
    model_registry.load()
    cp_path = settings.classifier_path
    sha = hashlib.sha256(cp_path.read_bytes()).hexdigest()
    expected_sha = "6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213"
    assert sha == expected_sha, f"SHA mismatch! Expected {expected_sha}, got {sha}"
    print(f"1. Model Checkpoint: {cp_path.name} | SHA-256 verified -> PASS")

    client = TestClient(app)

    # 2. Health Endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200, f"Health check failed with {resp_health.status_code}"
    h_data = resp_health.json()
    assert h_data["status"] == "ok" and h_data["inference_ready"] is True
    print(f"2. Health Endpoint: status={h_data['status']}, inference_ready={h_data['inference_ready']} -> PASS")

    # 3. Classes Endpoint
    resp_classes = client.get("/classes")
    assert resp_classes.status_code == 200
    c_data = resp_classes.json()
    assert len(c_data["classes"]) == 38
    print(f"3. Classes Endpoint: {len(c_data['classes'])} classes loaded -> PASS")

    # 4. Whole-Leaf Classification (/predict)
    healthy_leaf = "data/processed/PlantVillage/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"
    with open(healthy_leaf, "rb") as f:
        resp_pred = client.post("/predict", files={"file": ("healthy.jpg", f.read(), "image/jpeg")})
    assert resp_pred.status_code == 200
    p_data = resp_pred.json()["model"]
    assert p_data["predicted_class"] == "Apple___healthy"
    assert p_data["prediction_state"] == "HIGH_CONFIDENCE"
    print(f"4. Whole-Leaf Predict: {p_data['predicted_class']} ({p_data['confidence']*100:.1f}%), State: {p_data['prediction_state']} -> PASS")

    # 5. Single-Leaf Screening with No Candidates (Clean Leaf)
    with open(healthy_leaf, "rb") as f:
        resp_sl_clean = client.post("/screen-plant", files={"file": ("healthy.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp_sl_clean.status_code == 200
    sc_clean = resp_sl_clean.json()["screening"]
    assert sc_clean["candidate_count"] == 0
    assert sc_clean["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
    assert "Lack of candidate screening regions" in sc_clean["safety_disclaimer"]
    print(f"5. Single-Leaf Screening (Clean): Candidates={sc_clean['candidate_count']}, State={sc_clean['screening_state']}, Disclaimer Verified -> PASS")

    # 6. Single-Leaf Screening with Candidates (Diseased Leaf)
    diseased_leaf = "data/processed/PlantVillage/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    with open(diseased_leaf, "rb") as f:
        resp_sl_dis = client.post("/screen-plant", files={"file": ("scab.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp_sl_dis.status_code == 200
    sc_dis = resp_sl_dis.json()["screening"]
    assert sc_dis["candidate_count"] >= 1
    assert sc_dis["screening_state"] == "CANDIDATES_IDENTIFIED"
    assert sc_dis["overlay"].startswith("data:image/jpeg;base64,")
    print(f"6. Single-Leaf Screening (Diseased): Candidates={sc_dis['candidate_count']}, Overlay Present -> PASS")

    # 7. Whole-Plant Screening (Canopy)
    canopy_img = "data/raw/PlantDoc_dataset/test/Squash Powdery mildew leaf/Powdery-Mildew-on-squash.jpg"
    with open(canopy_img, "rb") as f:
        resp_wp = client.post("/screen-plant", files={"file": ("squash.jpg", f.read(), "image/jpeg")}, data={"mode": "whole_plant"})
    assert resp_wp.status_code == 200
    sc_wp = resp_wp.json()["screening"]
    assert sc_wp["mode"] == "whole_plant"
    assert sc_wp["overall_prediction"] is None
    assert sc_wp["regions_checked"] == 24
    assert sc_wp["candidate_count"] >= 1
    print(f"7. Whole-Plant Screening: Mode={sc_wp['mode']}, Overall Pred Decoupled, Crops Checked={sc_wp['regions_checked']}, Candidates={sc_wp['candidate_count']} -> PASS")

    # 8. Uncertain Prediction & Healthy Safety Warning
    ambig_img = r"C:\Users\Mohit Sharma\.gemini\antigravity\brain\ef968360-f136-470a-a159-8b64d2823609\.user_uploaded\media_1794656629400_32844b42.jpg"
    with open(ambig_img, "rb") as f:
        resp_ambig = client.post("/screen-plant", files={"file": ("ambig.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp_ambig.status_code == 200
    ov_ambig = resp_ambig.json()["screening"]["overall_prediction"]
    assert ov_ambig["confidence"] < 0.60
    assert ov_ambig["is_uncertain"] is True
    assert ov_ambig["prediction_state"] == "LOW_CONFIDENCE_UNCERTAIN"
    assert ov_ambig["is_healthy"] is True
    print(f"8. Uncertain Healthy Result: Confidence={ov_ambig['confidence']*100:.1f}%, State={ov_ambig['prediction_state']}, Alert Flag Verified -> PASS")

    # 9. Error Handling Verification
    r_empty = client.post("/screen-plant", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r_empty.status_code == 422
    r_bad_mode = client.post("/screen-plant", files={"file": ("test.jpg", b"123", "image/jpeg")}, data={"mode": "bad"})
    assert r_bad_mode.status_code == 422
    print("9. Error Handling (Empty 422, Invalid Mode 422) -> PASS")

    print("=" * 70)
    print("ALL 9 SMOKE TEST VERIFICATION GATES PASSED (100% SUCCESS)")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()

