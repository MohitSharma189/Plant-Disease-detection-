import os
import sys
import io
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path("backend").resolve()))

from starlette.testclient import TestClient
from app.main import app
from app.services.models import model_registry
from app.config import settings

def run_tests():
    print("=" * 70)
    print("PHASE 6A.4: FULL-STACK APPLICATION INTEGRATION AND WORKFLOW VALIDATION")
    print("=" * 70)

    # 1. Initialize client and ensure model is loaded
    client = TestClient(app)
    model_registry.load()

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(settings.classifier_path),
            "review_threshold": settings.default_review_threshold,
            "calibration_enabled": settings.calibration_enabled,
        },
        "scenarios": {},
        "benchmarks": {},
        "summary": {}
    }

    # Scenario 1: Healthy single-leaf image
    print("\n[Scenario 1] Healthy Single-Leaf Image (Real Inference)...")
    healthy_path = "data/processed/PlantVillage/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"
    with open(healthy_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("healthy.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data1 = resp.json()
    ov1 = data1["screening"]["overall_prediction"]
    sc1 = data1["screening"]
    s1_pass = (
        ov1["predicted_class"] == "Apple___healthy"
        and ov1["prediction_state"] == "HIGH_CONFIDENCE"
        and ov1["confidence"] >= 0.60
        and sc1["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
    )
    print(f"  Result: {ov1['predicted_class']} ({ov1['confidence']*100:.1f}%), State: {ov1['prediction_state']}, Candidates: {sc1['candidate_count']} -> {'PASS' if s1_pass else 'FAIL'}")
    results["scenarios"]["scenario_1_healthy_single_leaf"] = {
        "type": "REAL_MODEL_INFERENCE",
        "file": healthy_path,
        "status_code": resp.status_code,
        "predicted_class": ov1["predicted_class"],
        "confidence": ov1["confidence"],
        "prediction_state": ov1["prediction_state"],
        "screening_state": sc1["screening_state"],
        "candidate_count": sc1["candidate_count"],
        "passed": s1_pass
    }

    # Scenario 2: Diseased single-leaf image
    print("\n[Scenario 2] Diseased Single-Leaf Image (Real Inference)...")
    diseased_path = "data/processed/PlantVillage/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    with open(diseased_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("diseased.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data2 = resp.json()
    ov2 = data2["screening"]["overall_prediction"]
    sc2 = data2["screening"]
    s2_pass = (
        ov2["predicted_class"] == "Apple___Apple_scab"
        and ov2["prediction_state"] == "HIGH_CONFIDENCE"
        and sc2["screening_state"] == "CANDIDATES_IDENTIFIED"
        and sc2["candidate_count"] > 0
    )
    print(f"  Result: {ov2['predicted_class']} ({ov2['confidence']*100:.1f}%), State: {ov2['prediction_state']}, Candidates: {sc2['candidate_count']} -> {'PASS' if s2_pass else 'FAIL'}")
    results["scenarios"]["scenario_2_diseased_single_leaf"] = {
        "type": "REAL_MODEL_INFERENCE",
        "file": diseased_path,
        "status_code": resp.status_code,
        "predicted_class": ov2["predicted_class"],
        "confidence": ov2["confidence"],
        "prediction_state": ov2["prediction_state"],
        "screening_state": sc2["screening_state"],
        "candidate_count": sc2["candidate_count"],
        "passed": s2_pass
    }

    # Scenario 3: Uncertain prediction (<0.60 score)
    print("\n[Scenario 3] Uncertain Prediction Specimen (Real Inference)...")
    uncertain_path = r"C:\Users\Mohit Sharma\.gemini\antigravity\brain\ef968360-f136-470a-a159-8b64d2823609\.user_uploaded\media_1794656629400_32844b42.jpg"
    with open(uncertain_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("uncertain.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data3 = resp.json()
    ov3 = data3["screening"]["overall_prediction"]
    s3_pass = (
        ov3["confidence"] < 0.60
        and ov3["is_uncertain"] == True
        and ov3["prediction_state"] == "LOW_CONFIDENCE_UNCERTAIN"
        and "Review" in ov3["recommendation"] or "clinic" in ov3["recommendation"]
    )
    print(f"  Result: {ov3['predicted_class']} ({ov3['confidence']*100:.1f}%), State: {ov3['prediction_state']}, Is Healthy: {ov3.get('is_healthy')} -> {'PASS' if s3_pass else 'FAIL'}")
    results["scenarios"]["scenario_3_uncertain_prediction"] = {
        "type": "REAL_MODEL_INFERENCE",
        "file": uncertain_path,
        "status_code": resp.status_code,
        "predicted_class": ov3["predicted_class"],
        "confidence": ov3["confidence"],
        "prediction_state": ov3["prediction_state"],
        "is_uncertain": ov3["is_uncertain"],
        "is_healthy": ov3.get("is_healthy"),
        "passed": s3_pass
    }

    # Scenario 4: Single-leaf image with no screening candidates
    print("\n[Scenario 4] Single-Leaf with No Screening Candidates (Real Inference)...")
    with open(healthy_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("healthy.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    sc4 = resp.json()["screening"]
    s4_pass = (
        sc4["candidate_count"] == 0
        and sc4["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
        and "Lack of candidate screening regions" in sc4["safety_disclaimer"]
    )
    print(f"  Candidates: {sc4['candidate_count']}, State: {sc4['screening_state']}, Disclaimer Verified -> {'PASS' if s4_pass else 'FAIL'}")
    results["scenarios"]["scenario_4_single_leaf_no_candidates"] = {
        "type": "REAL_MODEL_INFERENCE",
        "candidate_count": sc4["candidate_count"],
        "screening_state": sc4["screening_state"],
        "safety_disclaimer": sc4["safety_disclaimer"],
        "passed": s4_pass
    }

    # Scenario 5: Single-leaf image with multiple candidates
    print("\n[Scenario 5] Single-Leaf with Multiple Candidates (Real Inference)...")
    with open(diseased_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("diseased.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    sc5 = resp.json()["screening"]
    s5_pass = (
        sc5["candidate_count"] >= 2
        and sc5["screening_state"] == "CANDIDATES_IDENTIFIED"
        and len(sc5["candidate_regions"]) == sc5["candidate_count"]
        and sc5["overlay"].startswith("data:image/jpeg;base64,")
    )
    print(f"  Candidates: {sc5['candidate_count']}, State: {sc5['screening_state']}, Overlay Length: {len(sc5['overlay'])} -> {'PASS' if s5_pass else 'FAIL'}")
    results["scenarios"]["scenario_5_single_leaf_multiple_candidates"] = {
        "type": "REAL_MODEL_INFERENCE",
        "candidate_count": sc5["candidate_count"],
        "screening_state": sc5["screening_state"],
        "overlay_present": bool(sc5["overlay"]),
        "passed": s5_pass
    }

    # Scenario 6: Whole-plant image mode
    print("\n[Scenario 6] Whole-Plant Mode Execution (Real Inference)...")
    wp_path = "data/processed/PlantDoc_Object_Detection/TRAIN/early_20blight_20_283_29.jpg_itok=UMYWI0xr.jpg"
    with open(wp_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("whole_plant.jpg", f.read(), "image/jpeg")}, data={"mode": "whole_plant"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    sc6 = resp.json()["screening"]
    s6_pass = (
        sc6["mode"] == "whole_plant"
        and sc6["overall_prediction"] is None
        and sc6["regions_checked"] == 24
    )
    print(f"  Mode: {sc6['mode']}, Overall Pred Decoupled: {sc6['overall_prediction'] is None}, Crops Checked: {sc6['regions_checked']} -> {'PASS' if s6_pass else 'FAIL'}")
    results["scenarios"]["scenario_6_whole_plant_mode"] = {
        "type": "REAL_MODEL_INFERENCE",
        "mode": sc6["mode"],
        "overall_prediction_present": sc6["overall_prediction"] is not None,
        "regions_checked": sc6["regions_checked"],
        "candidate_count": sc6["candidate_count"],
        "passed": s6_pass
    }

    # Scenario 7: Whole-plant image with no candidates
    print("\n[Scenario 7] Whole-Plant Image with No Candidates (Real Inference)...")
    # Using clean plant foliage
    clean_wp_path = "data/processed/PlantVillage/Apple___healthy/00907d8b-6ae6-4306-bfd7-d54471981a86___RS_HL 5709.JPG"
    with open(clean_wp_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("clean_plant.jpg", f.read(), "image/jpeg")}, data={"mode": "whole_plant"})
    sc7 = resp.json()["screening"]
    s7_pass = (
        sc7["candidate_count"] == 0
        and sc7["screening_state"] == "NO_CANDIDATES_IDENTIFIED"
        and "Lack of candidate screening regions" in sc7["safety_disclaimer"]
    )
    print(f"  Candidates: {sc7['candidate_count']}, State: {sc7['screening_state']}, Safety Disclaimer Checked -> {'PASS' if s7_pass else 'FAIL'}")
    results["scenarios"]["scenario_7_whole_plant_no_candidates"] = {
        "type": "REAL_MODEL_INFERENCE",
        "candidate_count": sc7["candidate_count"],
        "screening_state": sc7["screening_state"],
        "safety_disclaimer": sc7["safety_disclaimer"],
        "passed": s7_pass
    }

    # Scenario 8: Whole-plant image with multiple candidates
    print("\n[Scenario 8] Whole-Plant Image with Multiple Candidates (Real Inference)...")
    squash_path = "data/processed/PlantDoc_Object_Detection/TRAIN/Early_20Blight_20-_20Potatoes.ashx_w=350.jpg"
    with open(squash_path, "rb") as f:
        resp = client.post("/screen-plant", files={"file": ("squash.jpg", f.read(), "image/jpeg")}, data={"mode": "whole_plant"})
    sc8 = resp.json()["screening"]
    s8_pass = (
        sc8["candidate_count"] >= 1
        and sc8["screening_state"] == "CANDIDATES_IDENTIFIED"
        and sc8["regions_checked"] > 0
        and sc8["overlay"].startswith("data:image/jpeg;base64,")
    )
    print(f"  Candidates: {sc8['candidate_count']}, State: {sc8['screening_state']}, Regions Checked: {sc8['regions_checked']}, Top Candidate: {sc8['candidate_regions'][0]['predicted_class']} ({sc8['candidate_regions'][0]['confidence']*100:.1f}%) -> {'PASS' if s8_pass else 'FAIL'}")
    results["scenarios"]["scenario_8_whole_plant_multiple_candidates"] = {
        "type": "REAL_MODEL_INFERENCE",
        "candidate_count": sc8["candidate_count"],
        "regions_checked": sc8["regions_checked"],
        "screening_state": sc8["screening_state"],
        "top_candidate": sc8["candidate_regions"][0]["predicted_class"],
        "passed": s8_pass
    }

    # Scenario 9: Invalid image uploads
    print("\n[Scenario 9] Invalid Image Uploads (Automated ASGI Validation)...")
    # 9a: Empty file
    resp_empty = client.post("/screen-plant", files={"file": ("empty.jpg", b"", "image/jpeg")})
    # 9b: Corrupt file
    resp_corrupt = client.post("/screen-plant", files={"file": ("corrupt.jpg", b"NOT_VALID_IMAGE_DATA_12345", "image/jpeg")})
    # 9c: Sub-128px image (64x64)
    buf_small = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 100)).save(buf_small, format="JPEG")
    resp_small = client.post("/screen-plant", files={"file": ("small.jpg", buf_small.getvalue(), "image/jpeg")})
    # 9d: Oversized image (>10MB)
    dummy_oversized = b"0" * (11 * 1024 * 1024)
    resp_oversized = client.post("/screen-plant", files={"file": ("large.jpg", dummy_oversized, "image/jpeg")})
    # 9e: Invalid format (.txt)
    resp_txt = client.post("/screen-plant", files={"file": ("notes.txt", b"plain text data", "text/plain")})

    s9_pass = (
        resp_empty.status_code == 422
        and resp_corrupt.status_code == 422
        and resp_small.status_code == 422
        and resp_oversized.status_code == 413
        and resp_txt.status_code == 422
    )
    print(f"  Empty: {resp_empty.status_code}, Corrupt: {resp_corrupt.status_code}, Small: {resp_small.status_code}, Oversized: {resp_oversized.status_code}, Text: {resp_txt.status_code} -> {'PASS' if s9_pass else 'FAIL'}")
    results["scenarios"]["scenario_9_invalid_uploads"] = {
        "type": "AUTOMATED_ASGI_VALIDATION",
        "empty_code": resp_empty.status_code,
        "corrupt_code": resp_corrupt.status_code,
        "small_code": resp_small.status_code,
        "oversized_code": resp_oversized.status_code,
        "txt_code": resp_txt.status_code,
        "passed": s9_pass
    }

    # Scenario 10: Backend failure handling
    print("\n[Scenario 10] Backend Failure Handling (Automated ASGI Simulation)...")
    # 10a: Invalid mode
    with open(healthy_path, "rb") as f:
        resp_bad_mode = client.post("/screen-plant", files={"file": ("img.jpg", f.read(), "image/jpeg")}, data={"mode": "invalid_mode_xyz"})
    # 10b: Model unavailable simulation
    saved_classifier = model_registry.classifier
    try:
        model_registry.classifier = None
        with open(healthy_path, "rb") as f:
            resp_no_model = client.post("/screen-plant", files={"file": ("img.jpg", f.read(), "image/jpeg")}, data={"mode": "single_leaf"})
    finally:
        model_registry.classifier = saved_classifier

    s10_pass = (
        resp_bad_mode.status_code == 422
        and resp_no_model.status_code == 503
    )
    print(f"  Bad Mode: {resp_bad_mode.status_code} ({resp_bad_mode.json().get('detail')}), Model Unavailable: {resp_no_model.status_code} ({resp_no_model.json().get('detail')}) -> {'PASS' if s10_pass else 'FAIL'}")
    results["scenarios"]["scenario_10_backend_failures"] = {
        "type": "AUTOMATED_ASGI_SIMULATION",
        "bad_mode_code": resp_bad_mode.status_code,
        "no_model_code": resp_no_model.status_code,
        "passed": s10_pass
    }

    # Scenario 11: Different image dimensions
    print("\n[Scenario 11] Different Image Dimensions (Real Inference)...")
    dimensions_test = [
        ("Square Standard", (256, 256)),
        ("Landscape 4:3", (800, 600)),
        ("Portrait 3:4", (600, 800)),
        ("Widescreen 16:9", (1280, 720)),
    ]
    dim_results = []
    all_dim_pass = True
    for desc, (w, h) in dimensions_test:
        buf = io.BytesIO()
        # Create an image with foliar green gradients
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 1] = 160 # strong green foliar channel
        arr[:, :, 0] = 50
        arr[:, :, 2] = 40
        Image.fromarray(arr).save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
        resp_dim = client.post("/screen-plant", files={"file": ("dim.jpg", raw_bytes, "image/jpeg")}, data={"mode": "single_leaf"})
        d_data = resp_dim.json()
        passed_dim = resp_dim.status_code == 200 and d_data["screening"]["regions_checked"] > 0
        all_dim_pass = all_dim_pass and passed_dim
        dim_results.append({
            "name": desc,
            "dimensions": f"{w}x{h}",
            "status_code": resp_dim.status_code,
            "regions_checked": d_data["screening"]["regions_checked"],
            "candidate_count": d_data["screening"]["candidate_count"],
            "passed": passed_dim
        })
        print(f"  {desc} ({w}x{h}): Status {resp_dim.status_code}, Regions Checked: {d_data['screening']['regions_checked']} -> {'PASS' if passed_dim else 'FAIL'}")

    results["scenarios"]["scenario_11_dimensions"] = {
        "type": "REAL_MODEL_INFERENCE",
        "cases": dim_results,
        "passed": all_dim_pass
    }

    # Scenario 12: Frontend Responsive Layouts & Accessibility Static Audit
    print("\n[Scenario 12] Frontend Responsive Layouts & Accessibility Static Audit...")
    app_jsx = open("frontend/src/App.jsx", "r", encoding="utf-8").read()
    styles_css = open("frontend/src/styles.css", "r", encoding="utf-8").read()

    has_desktop_breakpoint = "@media (max-width: 1050px)" in styles_css
    has_tablet_breakpoint = "@media (min-width: 761px) and (max-width: 900px)" in styles_css
    has_mobile_breakpoint = "@media (max-width: 760px)" in styles_css
    has_small_mobile_breakpoint = "@media (max-width: 520px)" in styles_css
    has_print_styles = "@media print" in styles_css
    has_aria_live = "aria-live=" in app_jsx
    has_role_alert = 'role="alert"' in app_jsx
    has_role_status = 'role="status"' in app_jsx
    has_keyboard_dropzone = "onKeyDown=" in app_jsx

    s12_pass = (
        has_desktop_breakpoint
        and has_tablet_breakpoint
        and has_mobile_breakpoint
        and has_small_mobile_breakpoint
        and has_print_styles
        and has_aria_live
        and has_role_alert
        and has_role_status
        and has_keyboard_dropzone
    )
    print(f"  CSS Breakpoints: Desktop={has_desktop_breakpoint}, Tablet={has_tablet_breakpoint}, Mobile={has_mobile_breakpoint}, Small Mobile={has_small_mobile_breakpoint}, Print={has_print_styles}")
    print(f"  Accessibility: aria-live={has_aria_live}, role='alert'={has_role_alert}, role='status'={has_role_status}, Keyboard Dropzone={has_keyboard_dropzone} -> {'PASS' if s12_pass else 'FAIL'}")

    results["scenarios"]["scenario_12_frontend_responsive_a11y"] = {
        "type": "STATIC_AUDIT",
        "css_breakpoints": {
            "desktop_1050": has_desktop_breakpoint,
            "tablet_900": has_tablet_breakpoint,
            "mobile_760": has_mobile_breakpoint,
            "small_mobile_520": has_small_mobile_breakpoint,
            "print_styles": has_print_styles,
        },
        "accessibility_features": {
            "aria_live": has_aria_live,
            "role_alert": has_role_alert,
            "role_status": has_role_status,
            "keyboard_dropzone": has_keyboard_dropzone,
        },
        "passed": s12_pass
    }

    # Benchmarks: Stage-wise latency measurement over 5 runs
    print("\n[Benchmarking] Stage-wise Latency Measurement (5 runs)...")
    single_leaf_times = []
    whole_plant_times = []
    predict_times = []

    with open(diseased_path, "rb") as f:
        diseased_bytes = f.read()

    # Warmup
    client.post("/screen-plant", files={"file": ("w.jpg", diseased_bytes, "image/jpeg")}, data={"mode": "single_leaf"})

    for i in range(5):
        # /predict endpoint
        t0 = time.perf_counter()
        r_pred = client.post("/predict", files={"file": ("w.jpg", diseased_bytes, "image/jpeg")})
        predict_times.append((time.perf_counter() - t0) * 1000.0)

        # /screen-plant single_leaf
        t0 = time.perf_counter()
        r_sl = client.post("/screen-plant", files={"file": ("w.jpg", diseased_bytes, "image/jpeg")}, data={"mode": "single_leaf"})
        single_leaf_times.append((time.perf_counter() - t0) * 1000.0)

        # /screen-plant whole_plant
        t0 = time.perf_counter()
        r_wp = client.post("/screen-plant", files={"file": ("w.jpg", diseased_bytes, "image/jpeg")}, data={"mode": "whole_plant"})
        whole_plant_times.append((time.perf_counter() - t0) * 1000.0)

    results["benchmarks"] = {
        "predict_endpoint_ms": {
            "mean": float(np.mean(predict_times)),
            "std": float(np.std(predict_times)),
            "min": float(np.min(predict_times)),
            "max": float(np.max(predict_times)),
            "runs": predict_times
        },
        "screen_plant_single_leaf_ms": {
            "mean": float(np.mean(single_leaf_times)),
            "std": float(np.std(single_leaf_times)),
            "min": float(np.min(single_leaf_times)),
            "max": float(np.max(single_leaf_times)),
            "runs": single_leaf_times
        },
        "screen_plant_whole_plant_ms": {
            "mean": float(np.mean(whole_plant_times)),
            "std": float(np.std(whole_plant_times)),
            "min": float(np.min(whole_plant_times)),
            "max": float(np.max(whole_plant_times)),
            "runs": whole_plant_times
        }
    }
    print(f"  /predict Latency: {np.mean(predict_times):.2f} +/- {np.std(predict_times):.2f} ms")
    print(f"  Single-Leaf Screening Total Latency: {np.mean(single_leaf_times):.2f} +/- {np.std(single_leaf_times):.2f} ms")
    print(f"  Whole-Plant Screening Total Latency: {np.mean(whole_plant_times):.2f} +/- {np.std(whole_plant_times):.2f} ms")

    # Overall Summary
    total_scenarios = len(results["scenarios"])
    passed_scenarios = sum(1 for s in results["scenarios"].values() if s.get("passed"))
    results["summary"] = {
        "total_scenarios": total_scenarios,
        "passed_scenarios": passed_scenarios,
        "all_passed": passed_scenarios == total_scenarios
    }
    print("\n" + "=" * 70)
    print(f"OVERALL SUMMARY: {passed_scenarios}/{total_scenarios} SCENARIOS PASSED ({'ALL PASS' if passed_scenarios == total_scenarios else 'FAILURES DETECTED'})")
    print("=" * 70)

    # Save to json
    os.makedirs("reports/phase6a", exist_ok=True)
    out_json = "reports/phase6a/phase6a_4_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_json}")

if __name__ == "__main__":
    run_tests()
