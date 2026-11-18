# Phase 6A.2.1: Checkpoint Deployment Report — Verified Phase 5 Benchmark Deployment

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.2.1  
**Date:** November 14, 2026  
**Auditor / Systems Engineer:** Research & Systems Engineering Agent  
**Host Platform:** Windows Host, AMD Ryzen 7 5800HS (8 Physical Cores / 16 Threads, CPU Execution Mode)  
**Software Stack:** Python 3.12.5, PyTorch 2.15.0+cpu, Torchvision 0.29.2+cpu, FastAPI 0.110+, Uvicorn 0.29+  

---

## 1. Executive Summary

During Phase 6A.2, an audit revealed that the FastAPI production service was still loading an early baseline checkpoint (`saved_models/classifier.pt`, SHA-256: `1aef041f...`, 94.14% validation accuracy) rather than the verified Phase 5 benchmark model (`saved_models/experiments/mobilenet_v3_small_final/classifier.pt`, SHA-256: `6fff52f1...`, 95.14% validation accuracy, equivalent to `best_model.pth` with SHA-256 `6190f54c...`). 

Phase 6A.2.1 successfully remediates this deployment discrepancy by updating the deployment configuration to load the verified Phase 5 benchmark TorchScript checkpoint while strictly adhering to all preservation rules:
1. **Preservation of Artifacts:** The historical baseline checkpoint (`saved_models/classifier.pt`) was preserved untouched and unrenamed. The original Phase 5 benchmark checkpoint (`best_model.pth`) and TorchScript export (`classifier.pt`) remain intact.
2. **Cryptographic Integrity:** Pre- and post-modification SHA-256 digests confirm bit-for-bit identity across all three checkpoints.
3. **Deployment Configuration:** `backend/app/config.py`, `backend/app/services/models.py`, and `backend/app/main.py` were updated to load the verified Phase 5 TorchScript export as the primary classifier, with graceful fallback to test directories when running isolated unit test suites.
4. **Calibration & Review Threshold Integrity:** Temperature scaling remains disabled at $T = 1.0$ (`confidence_calibrated = False`, labeled `"Uncalibrated Model Score"`), and the operational review threshold of $0.60$ is strictly retained as an operational heuristic.
5. **Numerical Parity Verified:** FastAPI inference was tested against offline PyTorch forward evaluation across 10 real PlantVillage specimens, achieving **$\max |\Delta| = 0.000000\text{e}+00$**.
6. **Resolution of Discrepancies:** The Phase 5 model successfully resolved the misclassifications observed in Phase 6A.2 on `Grape___Black_rot` (now correctly predicted at 96.68%) and `Potato___Late_blight` (now correctly predicted at 76.54%).

---

## 2. Checkpoint Provenance & Cryptographic Verification

Prior to modifying configuration files, cryptographic SHA-256 checksums were calculated for all relevant model files:

| Checkpoint Identifier | File Location | File Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- | :--- |
| **Historical Baseline Checkpoint** | `saved_models/classifier.pt` | 6,527,514 bytes | `1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f` | **Preserved untouched**. Initial repository baseline (43,444 train / 10,861 val). |
| **Verified Phase 5 Benchmark (pth)** | `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` | 6,353,709 bytes | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | **Preserved untouched**. Documented Phase 5 benchmark weights (40,151 train / 5,436 val). |
| **Verified Phase 5 TorchScript (pt)** | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | 6,531,482 bytes | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` | **Deployed**. Exact TorchScript export of Phase 5 benchmark model. |
| **Active Class Mapping** | `saved_models/class_mapping.json` | 1,215 bytes | `f8988b3e341acb2e4b47ebcbaf7c756fcf3cc44c15c898953cb7d3a01dfd6945` | **Preserved untouched**. 38 PlantVillage classes sorted alphabetically. |
| **Phase 5 Metadata** | `saved_models/experiments/mobilenet_v3_small_final/classifier_metadata.json` | 4,197 bytes | `611593efc8e27cfa43bf9ebcc4a62bf9f7ca62ea83d47aee4148b871c5a96aa9` | **Preserved untouched**. Comprehensive training configuration & provenance. |

### Verification of Loaded Checkpoint:
* **Loaded Checkpoint Path:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`
* **Computed Checkpoint SHA-256:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`
* **Parity with Benchmark:** Bit-for-bit identical to the verified Phase 5 benchmark.

---

## 3. Configuration & Codebase Modifications

Three files were modified to transition the deployment configuration to the verified Phase 5 checkpoint without modifying model binaries:

### 3.1 `backend/app/config.py`
Explicit deployment paths and calibration flags were added to the Pydantic `Settings` class:
```python
    # Phase 6A.2.1: Deployed checkpoint configured to verified Phase 5 benchmark TorchScript model
    classifier_path: Path = (
        PROJECT_ROOT
        / "saved_models"
        / "experiments"
        / "mobilenet_v3_small_final"
        / "classifier.pt"
    )
    classifier_metadata_path: Path = (
        PROJECT_ROOT
        / "saved_models"
        / "experiments"
        / "mobilenet_v3_small_final"
        / "classifier_metadata.json"
    )
    class_mapping_path: Path = PROJECT_ROOT / "saved_models" / "class_mapping.json"
    calibration_enabled: bool = False  # Keep calibration disabled at T=1.0 until calibration discrepancy is independently resolved
    default_review_threshold: float = 0.60  # Operational engineering heuristic threshold
```

### 3.2 `backend/app/services/models.py`
Updated `ModelRegistry.load()`:
1. **Dynamic Path Resolution:** Inspects `settings.classifier_path`, prioritizing the Phase 5 benchmark checkpoint in production while seamlessly respecting `tmp_path` overrides during isolated unit tests.
2. **Explicit Calibration Enforcement:** Guarantees that `temperature = 1.0`, `calibration_report = None`, and `confidence_calibrated = False` as long as `settings.calibration_enabled` is `False`.
3. **Operational Review Threshold Enforcement:** Ensures that the operational threshold of `0.60` is applied rather than the historical Phase 3 temperature-scaled threshold ($0.2231$).
4. **Checkpoint Provenance Tracking:** Records `self.loaded_checkpoint_path = str(classifier_path)` for complete traceability.

### 3.3 `backend/app/main.py`
Updated `/health` endpoint:
* Ensures `classifier_checkpoint_present` checks the active deployed checkpoint path (`settings.classifier_path`) rather than the static `saved_models/classifier.pt` path.

---

## 4. Real-Specimen Inference Verification

Inference was evaluated on the same 10 real PlantVillage specimens used during Phase 6A.2 across both offline evaluation and deployed FastAPI endpoints:

| Specimen Image Path | Ground Truth Class | Offline Phase 5 Benchmark | Live FastAPI Deployed Endpoint | $\max \|\Delta\|$ | Prediction State | Uncalibrated Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Apple___healthy/0055dd26...JPG` | Apple - Healthy | `Apple___healthy` | `Apple___healthy` | **0.00e+00** | `HIGH_CONFIDENCE` | 98.99% |
| `Apple___Apple_scab/00075aa8...JPG` | Apple - Apple Scab | `Apple___Apple_scab` | `Apple___Apple_scab` | **0.00e+00** | `HIGH_CONFIDENCE` | 95.56% |
| `Cherry...healthy/0008f3d3...JPG` | Cherry (Sour) - Healthy | `Cherry_(including_sour)___healthy` | `Cherry_(including_sour)___healthy` | **0.00e+00** | `HIGH_CONFIDENCE` | 99.90% |
| `Cherry...Pwd.M/00705aa7...JPG` | Cherry (Sour) - Powdery Mildew | `Cherry_(including_sour)___Powdery_mildew` | `Cherry_(including_sour)___Powdery_mildew` | **0.00e+00** | `HIGH_CONFIDENCE` | 99.04% |
| `Corn...healthy/00031d74...jpg` | Corn (Maize) - Healthy | `Corn_(maize)___healthy` | `Corn_(maize)___healthy` | **0.00e+00** | `HIGH_CONFIDENCE` | 89.73% |
| `Corn...Common_rust_/RS_Rust...JPG` | Corn (Maize) - Common Rust | `Corn_(maize)___Common_rust_` | `Corn_(maize)___Common_rust_` | **0.00e+00** | `HIGH_CONFIDENCE` | 98.69% |
| `Grape___Black_rot/00090b0f...JPG` | Grape - Black Rot | `Grape___Black_rot` | `Grape___Black_rot` | **0.00e+00** | `HIGH_CONFIDENCE` | 96.68% |
| `Potato___Late_blight/0051e5e8...JPG` | Potato - Late Blight | `Potato___Late_blight` | `Potato___Late_blight` | **0.00e+00** | `HIGH_CONFIDENCE` | 76.54% |
| `Tomato___Early_blight/0012b9d2...JPG` | Tomato - Early Blight | `Tomato___Early_blight` | `Tomato___Early_blight` | **0.00e+00** | `HIGH_CONFIDENCE` | 60.73% |
| `Tomato___healthy/000146ff...JPG` | Tomato - Healthy | `Tomato___healthy` | `Tomato___healthy` | **0.00e+00** | `HIGH_CONFIDENCE` | 98.66% |

**Overall Maximum Absolute Difference across all 38 classes:** **$\mathbf{0.000000\text{e}+00}$**

### Resolution of Baseline Model Misclassifications:
1. **`Grape___Black_rot`:**
   - *Phase 6A.2 Deployed Baseline:* Misclassified as `Tomato___Bacterial_spot` (96.96%).
   - *Phase 6A.2.1 Deployed Phase 5 Benchmark:* Correctly classified as `Grape___Black_rot` (96.68%).
2. **`Potato___Late_blight`:**
   - *Phase 6A.2 Deployed Baseline:* Misclassified as `Tomato___Late_blight` (45.17%, uncertain).
   - *Phase 6A.2.1 Deployed Phase 5 Benchmark:* Correctly classified as `Potato___Late_blight` (76.54%, confident).

---

## 5. Verification of Prediction States & Error Handling

### 5.1 Category Coverage Validation
* **High-Confidence Healthy:** 4 specimens (`Apple___healthy` 98.99%, `Cherry...healthy` 99.90%, `Corn...healthy` 89.73%, `Tomato...healthy` 98.66%). State: `HIGH_CONFIDENCE`.
* **High-Confidence Diseased:** 6 specimens (`Apple___Apple_scab` 95.56%, `Cherry...Pwd.M` 99.04%, `Corn...Common_rust_` 98.69%, `Grape...Black_rot` 96.68%, `Potato...Late_blight` 76.54%, `Tomato...Early_blight` 60.73%). State: `HIGH_CONFIDENCE`.
* **Low-Confidence Diseased Specimen:** Evaluated `Apple___Apple_scab/2d5265d2...JPG` $\rightarrow$ Predicted `Pepper,_bell___Bacterial_spot` with score $49.32\% < 60.0\%$. State correctly assigned: `LOW_CONFIDENCE_UNCERTAIN` (`needs_review: True`, `is_uncertain: True`).
* **Low-Confidence Healthy Specimen:** Evaluated `Apple___Apple_scab/3652fd23...JPG` $\rightarrow$ Predicted `Apple___healthy` with score $54.53\% < 60.0\%$. State correctly assigned: `LOW_CONFIDENCE_UNCERTAIN` (`needs_review: True`, `is_uncertain: True`).

### 5.2 Screening Pipeline Verification (`/screen-plant`)
* **Single-Leaf Mode:** Processed sample specimen $\rightarrow$ Checked 9 overlapping candidate patches; returned `screening_status = "No Candidate Regions Identified"`, with full visual candidate overlay and clean schema validation.
* **Whole-Plant Mode:** Processed sample specimen $\rightarrow$ Executed $3 \times 3$ grid tiling; returned `screening_status = "No Candidate Regions Identified"`, with proper attribution and disclaimer payloads.

### 5.3 Error Diagnostics
* **Empty Payload (0 bytes):** HTTP 422 (`{"detail": "The uploaded file is empty."}`)
* **Corrupted Image Bytes:** HTTP 422 (`{"detail": "The uploaded file is not a valid image."}`)
* **Sub-Minimum Dimensions ($64 \times 64$):** HTTP 422 (`{"detail": "Image dimensions must be at least 128x128 pixels."}`)
* **Oversized Payload ($>10$ MB):** HTTP 413 (`{"detail": "Image exceeds the 10 MB upload limit."}`)
* **Invalid Screening Mode:** HTTP 422 (`{"detail": "Mode must be 'single_leaf' or 'whole_plant'."}`)

---

## 6. CPU Inference Latency Measurement

Latency was benchmarked anew on the host processor (AMD Ryzen 7 5800HS, 8 physical CPU cores, PyTorch 2.15.0+cpu) using high-resolution monotonic timers:

| Measurement Stage | Iterations | Mean Latency | Std Dev | Median | Min | Max | 95th Percentile ($p95$) | 99th Percentile ($p99$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure TorchScript Forward Pass** | 100 runs | **12.38 ms** | 1.30 ms | 12.58 ms | 9.84 ms | 16.58 ms | **13.98 ms** | 16.00 ms |
| **End-to-End FastAPI `/predict`** | 50 HTTP calls | **22.07 ms** | 1.45 ms | 21.84 ms | 19.38 ms | 25.69 ms | **24.27 ms** | 25.15 ms |

The Phase 5 TorchScript model exhibits execution latency virtually indistinguishable from the baseline model ($12.38\text{ ms}$ vs. $12.69\text{ ms}$ forward pass; $22.07\text{ ms}$ vs. $23.15\text{ ms}$ end-to-end endpoint), demonstrating high runtime efficiency on consumer CPU hardware.

---

## 7. Test Suite Execution

The comprehensive automated regression test suite was executed:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.5, pytest-9.2.0, pluggy-1.7.0
rootdir: E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection
collected 44 items

backend\tests\test_chat.py .........                                     [ 20%]
backend\tests\test_health.py ...                                         [ 27%]
backend\tests\test_learning.py ....                                      [ 36%]
backend\tests\test_measurements.py ...                                   [ 43%]
backend\tests\test_models.py ...                                         [ 50%]
backend\tests\test_prediction_states.py ..........                       [ 72%]
backend\tests\test_quality.py .                                          [ 75%]
backend\tests\test_research.py .....                                     [ 86%]
backend\tests\test_screening.py ......                                   [100%]

======================= 44 passed, 6 warnings in 15.78s =======================
```
All 44 test cases passed without error.

---

## 8. Remaining Limitations

1. **Uncalibrated Model Scores:**
   Scores returned by the deployed endpoint are uncalibrated softmax scores ($T = 1.0$), correctly labeled as `"Uncalibrated Model Score"`. Until temperature scaling is unified and calibrated against the multimodal split holdout partition, calibration remains intentionally disabled.
2. **Operational Review Threshold:**
   The `0.60` threshold functions as an operational engineering cutoff, not an empirically fitted selective accuracy cutoff. Tentative predictions must continue to be reviewed by agricultural professionals.
3. **Out-of-Distribution Sensitivity:**
   The classifier was trained on segmented and close-up PlantVillage leaf images. While the Phase 5 model exhibits superior discriminative power, non-supported crops or natural clutter must not be presumed healthy.

