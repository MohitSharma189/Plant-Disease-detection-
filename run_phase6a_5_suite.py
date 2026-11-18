import os
import sys
import io
import time
import json
import glob
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path("backend").resolve()))

from starlette.testclient import TestClient
from app.main import app
from app.services.models import model_registry
from app.config import settings

def find_first(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def run_suite():
    print("=" * 80)
    print("PHASE 6A.5: COMPREHENSIVE APPLICATION VALIDATION, ROBUSTNESS & BENCHMARKING")
    print("=" * 80)

    client = TestClient(app)
    model_registry.load()

    # Hardware & Environment details
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "os": "Windows 11",
                "processor": "AMD Ryzen 7 5800HS with Radeon Graphics (8 Cores, 16 Threads)",
                "ram": "16 GB",
                "python_version": sys.version.split()[0],
                "torch_version": "2.15.0+cpu",
                "fastapi_app": settings.app_name,
            },
            "checkpoint": {
                "path": str(settings.classifier_path.resolve()),
                "sha256": hashlib.sha256(settings.classifier_path.read_bytes()).hexdigest(),
                "expected_sha256": "6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213",
                "verified": hashlib.sha256(settings.classifier_path.read_bytes()).hexdigest() == "6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213",
                "classes_count": len(model_registry.class_names or []),
                "temperature": model_registry.temperature,
                "review_threshold": model_registry.review_threshold,
                "calibration_enabled": settings.calibration_enabled,
            }
        },
        "classification_validation": [],
        "screening_validation": [],
        "robustness_testing": [],
        "backend_reliability": {},
        "concurrency_testing": {},
        "performance_benchmarks": {}
    }

    # =========================================================================
    # PART 1: CLASSIFICATION VALIDATION
    # =========================================================================
    print("\n--- [PART 1] CLASSIFICATION VALIDATION (REPRESENTATIVE SPECIMENS) ---")

    # Locate representative test images
    test_specs = [
        # Healthy
        ("Apple___healthy", find_first("data/processed/PlantVillage/Apple___healthy/*.JPG")),
        ("Tomato___healthy", find_first("data/processed/PlantVillage/Tomato___healthy/*.JPG")),
        ("Grape___healthy", find_first("data/processed/PlantVillage/Grape___healthy/*.JPG")),
        # Diseased
        ("Apple___Apple_scab", find_first("data/processed/PlantVillage/Apple___Apple_scab/*.JPG")),
        ("Grape___Black_rot", find_first("data/processed/PlantVillage/Grape___Black_rot/*.JPG")),
        ("Potato___Late_blight", find_first("data/processed/PlantVillage/Potato___Late_blight/*.JPG")),
        ("Corn_(maize)___Northern_Leaf_Blight", find_first("data/processed/PlantVillage/Corn_(maize)___Northern_Leaf_Blight/*.JPG")),
        ("Tomato___Early_blight", find_first("data/processed/PlantVillage/Tomato___Early_blight/*.JPG")),
        ("Squash___Powdery_mildew", find_first("data/processed/PlantVillage/Squash___Powdery_mildew/*.JPG")),
        # Uncertain healthy specimen (user uploaded)
        ("Ambiguous_Foliage", r"C:\Users\Mohit Sharma\.gemini\antigravity\brain\ef968360-f136-470a-a159-8b64d2823609\.user_uploaded\media_1794656629400_32844b42.jpg"),
        # Challenging in-the-wild specimens from PlantDoc
        ("PlantDoc_Potato_Blight", find_first("data/processed/PlantDoc_Object_Detection/TRAIN/Early_20Blight_20-_20Potatoes*.jpg")),
        ("PlantDoc_Squash_Mildew", find_first("data/raw/PlantDoc_dataset/test/Squash Powdery mildew leaf/Powdery-Mildew-on-squash.jpg")),
    ]

    for label, img_path in test_specs:
        if not img_path or not os.path.exists(img_path):
            print(f"  [SKIP] Image not found for {label}")
            continue
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        resp = client.post("/predict", files={"file": (os.path.basename(img_path), img_bytes, "image/jpeg")})
        assert resp.status_code == 200, f"Error {resp.status_code} on {label}"
        p = resp.json()["model"]
        q = resp.json()["quality"]

        record = {
            "target_label": label,
            "filename": os.path.basename(img_path),
            "file_path": img_path,
            "image_dimensions": f"{q['width']}x{q['height']}",
            "is_blurry": q["is_blurry"],
            "blur_score": q["blur_score"],
            "predicted_class": p["predicted_class"],
            "display_label": p.get("display_label"),
            "confidence": p["confidence"],
            "prediction_state": p["prediction_state"],
            "is_uncertain": p["is_uncertain"],
            "is_healthy": p.get("is_healthy", False),
            "needs_review": p.get("needs_review", False),
            "recommendation": p.get("recommendation", "")
        }
        results["classification_validation"].append(record)
        print(f"  {label:<32} -> {p['predicted_class']:<32} | Score: {p['confidence']*100:5.1f}% | State: {p['prediction_state']:<24} | Review: {p.get('needs_review')}")

    # =========================================================================
    # PART 2: SCREENING VALIDATION (SINGLE-LEAF & WHOLE-PLANT)
    # =========================================================================
    print("\n--- [PART 2] SCREENING VALIDATION ---")

    screening_test_cases = [
        # 1. Single leaf no candidates (Apple healthy)
        ("SingleLeaf_NoCandidates", find_first("data/processed/PlantVillage/Apple___healthy/*.JPG"), "single_leaf"),
        # 2. Single leaf multiple candidates (Apple scab)
        ("SingleLeaf_MultipleCandidates", find_first("data/processed/PlantVillage/Apple___Apple_scab/*.JPG"), "single_leaf"),
        # 3. Whole plant no candidates (Clean foliage)
        ("WholePlant_NoCandidates", find_first("data/processed/PlantVillage/Apple___healthy/*.JPG"), "whole_plant"),
        # 4. Whole plant multiple candidates (PlantDoc Potato Blight)
        ("WholePlant_MultipleCandidates", find_first("data/processed/PlantDoc_Object_Detection/TRAIN/Early_20Blight_20-_20Potatoes*.jpg"), "whole_plant"),
        # 5. Whole plant squash mildew (PlantDoc Squash)
        ("WholePlant_SquashMildew", find_first("data/raw/PlantDoc_dataset/test/Squash Powdery mildew leaf/Powdery-Mildew-on-squash.jpg"), "whole_plant"),
        # 6. Aspect Ratio 1:1 Square (256x256)
        ("AspectRatio_1_1_Square", "synthetic_square_256", "whole_plant"),
        # 7. Aspect Ratio 4:3 Landscape (800x600)
        ("AspectRatio_4_3_Landscape", "synthetic_landscape_800x600", "whole_plant"),
        # 8. Aspect Ratio 3:4 Portrait (600x800)
        ("AspectRatio_3_4_Portrait", "synthetic_portrait_600x800", "whole_plant"),
        # 9. Aspect Ratio 16:9 Widescreen (1280x720)
        ("AspectRatio_16_9_Widescreen", "synthetic_wide_1280x720", "whole_plant"),
    ]

    for name, path_or_type, mode in screening_test_cases:
        if not path_or_type:
            print(f"  [SKIP] Image not found for {name}")
            continue
        if isinstance(path_or_type, str) and path_or_type.startswith("synthetic"):
            parts = path_or_type.split("_")
            dims = parts[-1].split("x")
            if len(dims) == 1:
                w = h = int(dims[0])
            else:
                w, h = int(dims[0]), int(dims[1])
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr[:, :, 1] = 150 # green foliar
            arr[:, :, 0] = 60
            arr[:, :, 2] = 50
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            filename = f"{name}.jpg"
        else:
            if not path_or_type or not os.path.exists(path_or_type):
                print(f"  [SKIP] Image not found for {name}")
                continue
            with open(path_or_type, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(path_or_type)

        t0 = time.perf_counter()
        resp = client.post("/screen-plant", files={"file": (filename, img_bytes, "image/jpeg")}, data={"mode": mode})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200, f"Error {resp.status_code} on {name}"
        sc = resp.json()["screening"]

        top_cand = sc["candidate_regions"][0]["predicted_class"] if sc["candidate_regions"] else "None"
        top_score = sc["candidate_regions"][0]["confidence"] if sc["candidate_regions"] else 0.0

        record = {
            "case_name": name,
            "filename": filename,
            "mode": mode,
            "elapsed_ms": elapsed_ms,
            "regions_checked": sc["regions_checked"],
            "candidate_count": sc["candidate_count"],
            "screening_state": sc["screening_state"],
            "top_candidate": top_cand,
            "top_score": top_score,
            "candidate_boxes": [c["box"] for c in sc["candidate_regions"]],
            "overlay_generated": bool(sc.get("overlay")),
            "safety_disclaimer": sc.get("safety_disclaimer")
        }
        results["screening_validation"].append(record)
        print(f"  {name:<30} | Mode: {mode:<12} | Checked: {sc['regions_checked']:2d} | Cands: {sc['candidate_count']:2d} | Top: {top_cand:<28} ({top_score*100:4.1f}%) | Time: {elapsed_ms:6.1f} ms")

    # =========================================================================
    # PART 3: IMAGE ROBUSTNESS TESTING (PERTURBATIONS)
    # =========================================================================
    print("\n--- [PART 3] IMAGE ROBUSTNESS TESTING (CONTROLLED PERTURBATIONS) ---")

    # Base specimen for perturbation: Apple Scab (standard diseased single leaf)
    base_diseased_path = find_first("data/processed/PlantVillage/Apple___Apple_scab/*.JPG")
    base_healthy_path = find_first("data/processed/PlantVillage/Apple___healthy/*.JPG")
    base_img = Image.open(base_diseased_path).convert("RGB")

    # Baseline prediction
    baseline_pred = model_registry.predict_class(base_img)
    base_class = baseline_pred["predicted_class"]
    base_conf = baseline_pred["confidence"]
    print(f"  Baseline Specimen: Apple Scab | Class: {base_class} | Conf: {base_conf*100:.2f}% | State: {baseline_pred['prediction_state']}")

    perturbations = []

    # 1. Brightness
    for factor, label in [(0.5, "Brightness -50%"), (0.75, "Brightness -25%"), (1.25, "Brightness +25%"), (1.5, "Brightness +50%")]:
        enhancer = ImageEnhance.Brightness(base_img)
        perturbations.append((label, enhancer.enhance(factor)))

    # 2. Contrast
    for factor, label in [(0.5, "Contrast 0.5x (Low)"), (1.5, "Contrast 1.5x (High)")]:
        enhancer = ImageEnhance.Contrast(base_img)
        perturbations.append((label, enhancer.enhance(factor)))

    # 3. Gaussian Blur
    for radius, label in [(1.0, "Gaussian Blur radius 1.0"), (2.0, "Gaussian Blur radius 2.0")]:
        perturbations.append((label, base_img.filter(ImageFilter.GaussianBlur(radius=radius))))

    # 4. Rotation
    for angle, label in [(15, "Rotation +15 deg tilt"), (90, "Rotation 90 deg"), (180, "Rotation 180 deg"), (270, "Rotation 270 deg")]:
        perturbations.append((label, base_img.rotate(angle, expand=True)))

    # 5. Resolution scaling
    for dims, label in [((160, 160), "Downscaled to 160x160"), ((224, 224), "Rescaled to 224x224 (Exact CNN Input)"), ((600, 600), "Upscaled to 600x600")]:
        perturbations.append((label, base_img.resize(dims, Image.BILINEAR)))

    # 6. Aspect Ratio deformation
    for dims, label in [((200, 400), "Aspect Ratio Stretch 1:2"), ((400, 200), "Aspect Ratio Stretch 2:1")]:
        perturbations.append((label, base_img.resize(dims, Image.BILINEAR)))

    # 7. JPEG compression artifacts
    for q, label in [(30, "JPEG Compression Quality 30"), (15, "JPEG Compression Quality 15 (Heavy)")]:
        buf = io.BytesIO()
        base_img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        perturbations.append((label, Image.open(buf).convert("RGB")))

    # 8. Mild Additive Gaussian Noise
    np_base = np.asarray(base_img, dtype=np.float32)
    noise = np.random.normal(0, 15, np_base.shape).astype(np.float32)
    np_noisy = np.clip(np_base + noise, 0, 255).astype(np.uint8)
    perturbations.append(("Gaussian Noise (sigma=15)", Image.fromarray(np_noisy)))

    for pert_name, p_img in perturbations:
        buf = io.BytesIO()
        p_img.save(buf, format="JPEG")
        raw_b = buf.getvalue()

        # Test both /predict and /screen-plant
        resp_pred = client.post("/predict", files={"file": ("pert.jpg", raw_b, "image/jpeg")})
        resp_sc = client.post("/screen-plant", files={"file": ("pert.jpg", raw_b, "image/jpeg")}, data={"mode": "single_leaf"})

        p_data = resp_pred.json()["model"]
        sc_data = resp_sc.json()["screening"]

        class_match = (p_data["predicted_class"] == base_class)
        delta_conf = p_data["confidence"] - base_conf

        record = {
            "perturbation": pert_name,
            "dimensions": f"{p_img.width}x{p_img.height}",
            "predicted_class": p_data["predicted_class"],
            "confidence": p_data["confidence"],
            "confidence_delta": delta_conf,
            "class_preserved": class_match,
            "prediction_state": p_data["prediction_state"],
            "is_uncertain": p_data["is_uncertain"],
            "candidates_found": sc_data["candidate_count"],
            "screening_state": sc_data["screening_state"]
        }
        results["robustness_testing"].append(record)
        print(f"  {pert_name:<36} -> {p_data['predicted_class']:<24} | Conf: {p_data['confidence']*100:5.1f}% (Δ {delta_conf*100:+5.1f}%) | Match: {str(class_match):<5} | Cands: {sc_data['candidate_count']:2d}")

    # =========================================================================
    # PART 4: API & BACKEND RELIABILITY AND ERROR BOUNDARY TESTING
    # =========================================================================
    print("\n--- [PART 4] API & BACKEND RELIABILITY ---")

    err_tests = [
        ("Empty File (0 bytes)", b"", 422, "The uploaded file is empty."),
        ("Corrupt Data Bytes", b"MALFORMED_HEADER_INVALID_JPEG_BITS", 422, "The uploaded file is not a valid image."),
        ("Tiny Sub-128px Image", Image.new("RGB", (64, 64)), 422, "Image dimensions must be at least 128x128 pixels."),
        ("Oversized Upload (>10MB)", b"X" * (10 * 1024 * 1024 + 500), 413, "Image exceeds the 10 MB upload limit."),
        ("Unsupported Text Format", b"Just some plain text content", 422, "The uploaded file is not a valid image."),
    ]

    for name, payload, expected_status, expected_detail in err_tests:
        if isinstance(payload, Image.Image):
            buf = io.BytesIO()
            payload.save(buf, format="JPEG")
            data_bytes = buf.getvalue()
        else:
            data_bytes = payload

        resp = client.post("/screen-plant", files={"file": ("test.jpg", data_bytes, "image/jpeg")})
        passed = (resp.status_code == expected_status) and (expected_detail.lower() in resp.json().get("detail", "").lower())
        results["backend_reliability"][name] = {
            "expected_status": expected_status,
            "observed_status": resp.status_code,
            "detail": resp.json().get("detail", ""),
            "passed": passed
        }
        print(f"  {name:<30} -> Status: {resp.status_code} (Exp: {expected_status}) | Detail: '{resp.json().get('detail')}' -> {'PASS' if passed else 'FAIL'}")

    # Mode validation
    resp_bad_mode = client.post("/screen-plant", files={"file": ("test.jpg", raw_b, "image/jpeg")}, data={"mode": "diagonal_scan"})
    bm_pass = (resp_bad_mode.status_code == 422) and ("Mode must be 'single_leaf' or 'whole_plant'." in resp_bad_mode.json().get("detail", ""))
    results["backend_reliability"]["Invalid Screening Mode"] = {
        "expected_status": 422,
        "observed_status": resp_bad_mode.status_code,
        "detail": resp_bad_mode.json().get("detail"),
        "passed": bm_pass
    }
    print(f"  {'Invalid Screening Mode':<30} -> Status: {resp_bad_mode.status_code} | Detail: '{resp_bad_mode.json().get('detail')}' -> {'PASS' if bm_pass else 'FAIL'}")

    # Model unavailable 503 test
    saved_classifier = model_registry.classifier
    try:
        model_registry.classifier = None
        resp_unavail = client.post("/screen-plant", files={"file": ("test.jpg", raw_b, "image/jpeg")}, data={"mode": "single_leaf"})
        unavail_pass = (resp_unavail.status_code == 503)
        results["backend_reliability"]["Model Unavailable (503)"] = {
            "expected_status": 503,
            "observed_status": resp_unavail.status_code,
            "detail": resp_unavail.json().get("detail"),
            "passed": unavail_pass
        }
        print(f"  {'Model Unavailable (503)':<30} -> Status: {resp_unavail.status_code} | Detail: '{resp_unavail.json().get('detail')}' -> {'PASS' if unavail_pass else 'FAIL'}")
    finally:
        model_registry.classifier = saved_classifier

    # Repeated sequential requests (25 runs stability)
    print("\n  Executing 25 sequential requests for memory/stability check...")
    seq_statuses = []
    for _ in range(25):
        r = client.post("/predict", files={"file": ("test.jpg", raw_b, "image/jpeg")})
        seq_statuses.append(r.status_code)
    all_seq_ok = all(s == 200 for s in seq_statuses)
    results["backend_reliability"]["Sequential 25 Requests"] = {
        "runs": len(seq_statuses),
        "all_200_ok": all_seq_ok,
        "passed": all_seq_ok
    }
    print(f"  Sequential 25 Requests -> 200 OK: {seq_statuses.count(200)}/25 -> {'PASS' if all_seq_ok else 'FAIL'}")

    # Concurrent multithreaded requests (10 parallel clients)
    print("\n  Executing 10 concurrent requests via ThreadPoolExecutor...")
    def send_req(i):
        c = TestClient(app)
        t_start = time.perf_counter()
        res = c.post("/screen-plant", files={"file": (f"test_{i}.jpg", raw_b, "image/jpeg")}, data={"mode": "single_leaf"})
        return res.status_code, (time.perf_counter() - t_start) * 1000.0

    with ThreadPoolExecutor(max_workers=10) as executor:
        conc_results = list(executor.map(send_req, range(10)))

    conc_codes = [c[0] for c in conc_results]
    conc_times = [c[1] for c in conc_results]
    all_conc_ok = all(code == 200 for code in conc_codes)
    results["concurrency_testing"] = {
        "concurrent_clients": 10,
        "all_200_ok": all_conc_ok,
        "latencies_ms": conc_times,
        "mean_latency_ms": float(np.mean(conc_times)),
        "max_latency_ms": float(np.max(conc_times)),
        "passed": all_conc_ok
    }
    print(f"  Concurrent 10 Clients -> 200 OK: {conc_codes.count(200)}/10 | Mean: {np.mean(conc_times):.1f} ms | Max: {np.max(conc_times):.1f} ms -> {'PASS' if all_conc_ok else 'FAIL'}")

    # =========================================================================
    # PART 5: PERFORMANCE BENCHMARKING (50 TIMED RUNS WITH 10 WARMUPS)
    # =========================================================================
    print("\n--- [PART 5] PERFORMANCE BENCHMARKING (50 TIMED RUNS EACH, 10 WARMUPS) ---")

    bench_targets = [
        ("Pure Model Inference (PyTorch forward)", "pure_model"),
        ("Whole-Leaf API Endpoint (/predict)", "predict_api"),
        ("Single-Leaf Screening Endpoint (/screen-plant)", "screen_single_leaf"),
        ("Whole-Plant Screening Endpoint (/screen-plant)", "screen_whole_plant"),
    ]

    benchmark_stats = {}

    # Standard test image: Apple Scab (256x256)
    with open(base_diseased_path, "rb") as f:
        bench_bytes = f.read()
    bench_pil = Image.open(io.BytesIO(bench_bytes)).convert("RGB")

    for title, key in bench_targets:
        print(f"\n  Target: {title}")
        print("    Running 10 warm-up cycles...")
        for _ in range(10):
            if key == "pure_model":
                model_registry.predict_class(bench_pil)
            elif key == "predict_api":
                client.post("/predict", files={"file": ("b.jpg", bench_bytes, "image/jpeg")})
            elif key == "screen_single_leaf":
                client.post("/screen-plant", files={"file": ("b.jpg", bench_bytes, "image/jpeg")}, data={"mode": "single_leaf"})
            elif key == "screen_whole_plant":
                client.post("/screen-plant", files={"file": ("b.jpg", bench_bytes, "image/jpeg")}, data={"mode": "whole_plant"})

        print("    Executing 50 timed benchmark runs...")
        latencies = []
        for _ in range(50):
            t_start = time.perf_counter()
            if key == "pure_model":
                model_registry.predict_class(bench_pil)
            elif key == "predict_api":
                client.post("/predict", files={"file": ("b.jpg", bench_bytes, "image/jpeg")})
            elif key == "screen_single_leaf":
                client.post("/screen-plant", files={"file": ("b.jpg", bench_bytes, "image/jpeg")}, data={"mode": "single_leaf"})
            elif key == "screen_whole_plant":
                client.post("/screen-plant", files={"file": ("b.jpg", bench_bytes, "image/jpeg")}, data={"mode": "whole_plant"})
            latencies.append((time.perf_counter() - t_start) * 1000.0)

        lat_arr = np.array(latencies)
        mean_val = float(np.mean(lat_arr))
        std_val = float(np.std(lat_arr))
        median_val = float(np.median(lat_arr))
        p95_val = float(np.percentile(lat_arr, 95))
        p99_val = float(np.percentile(lat_arr, 99))
        min_val = float(np.min(lat_arr))
        max_val = float(np.max(lat_arr))

        benchmark_stats[key] = {
            "title": title,
            "sample_size": 50,
            "warmup_runs": 10,
            "mean_ms": mean_val,
            "std_ms": std_val,
            "median_ms": median_val,
            "p95_ms": p95_val,
            "p99_ms": p99_val,
            "min_ms": min_val,
            "max_ms": max_val,
            "all_runs": latencies
        }
        print(f"    Mean: {mean_val:6.2f} +/- {std_val:5.2f} ms | Median: {median_val:6.2f} ms | p95: {p95_val:6.2f} ms | p99: {p99_val:6.2f} ms | [Min: {min_val:6.2f}, Max: {max_val:6.2f}]")

    results["performance_benchmarks"] = benchmark_stats

    # =========================================================================
    # SUMMARY & JSON SAVE
    # =========================================================================
    os.makedirs("reports/phase6a", exist_ok=True)
    out_file = "reports/phase6a/phase6a_5_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComprehensive Phase 6A.5 evaluation data saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
