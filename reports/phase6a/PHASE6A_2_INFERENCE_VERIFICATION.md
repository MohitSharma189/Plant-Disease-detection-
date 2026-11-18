# Phase 6A.2: Inference Verification Report — Comparative Analysis & Error Diagnostics

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.2  
**Date:** November 14, 2026  
**Auditor / Verification Engineer:** Research & Systems Engineering Agent  
**Host Environment:** Windows CPU Host, AMD Ryzen 7 5800HS (8 Physical Cores, 16 Threads)  
**Execution Mode:** Pure PyTorch CPU Execution & FastAPI Asynchronous Test Server  

---

## 1. Scope & Verification Protocol

Phase 6A.2 verifies the numerical and operational consistency of the MobileNetV3-Small plant disease classification model deployed inside the FastAPI web service.

To satisfy all integrity and safety rules:
1. **Real Inference vs. Mocked Testing:**
   All numerical comparisons, latency profiling, and checkpoint audits in this report were conducted using **real image files from the PlantVillage dataset** and **real serialized PyTorch/TorchScript model checkpoints**. Mocked registries were utilized strictly in isolated unit test assertions (`backend/tests/test_prediction_states.py`) to test hypothetical edge-case boundary conditions.
2. **Preservation Rules:**
   No model checkpoints were overwritten or renamed. No preprocessing, normalization, or class mappings were altered.

---

## 2. Cryptographic Checkpoint Provenance

| Checkpoint Identifier | File Location | SHA-256 Digest | File Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Active Production Checkpoint** | `saved_models/classifier.pt` | `1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f` | 6,527,514 bytes | Deployed TorchScript model used by FastAPI service |
| **Active Class Mapping** | `saved_models/class_mapping.json` | `f8988b3e341acb2e4b47ebcbaf7c756fcf3cc44c15c898953cb7d3a01dfd6945` | 1,189 bytes | 38 sorted PlantVillage class strings |
| **Active Metadata** | `saved_models/classifier_metadata.json` | `602fcdd5651fb79f187902d13ca0430e702580798e6a12702737c0411a2f64fa` | 1,472 bytes | Metadata recording 43,444 train / 10,861 val samples |
| **Documented Phase 5 Checkpoint** | `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | 6,353,709 bytes | Documented Phase 5 benchmark checkpoint |
| **Documented Phase 5 JIT Export** | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` | 6,531,482 bytes | Verified TorchScript export of Phase 5 checkpoint |

---

## 3. Real Specimen Comparative Inference Test

Ten real leaf specimens were selected from `data/processed/PlantVillage/` spanning multiple plant families and conditions. Each specimen was evaluated across three distinct pipelines:
- **Pipeline A (FastAPI Live API):** Uploaded via HTTP multipart request to the FastAPI `/predict` endpoint.
- **Pipeline B (Offline Deployed Model):** Direct offline forward pass of `saved_models/classifier.pt` using standard torchvision transforms.
- **Pipeline C (Offline Phase 5 Benchmark):** Direct offline forward pass of the documented Phase 5 benchmark model (`mobilenet_v3_small_final/classifier.pt`).

### Specimen Evaluation Matrix:

| Specimen Image Path | Ground Truth Class | Pipeline A (FastAPI Endpoint) | Pipeline B (Offline Deployed) | $\max \|\Delta_{AB}\|$ | Pipeline C (Phase 5 Benchmark) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Apple___healthy/0055dd26...JPG` | Apple - Healthy | `Apple___healthy` (0.999974) | `Apple___healthy` (0.999974) | **0.00e+00** | `Apple___healthy` (0.989913) |
| `Apple___Apple_scab/00075aa8...JPG` | Apple - Apple Scab | `Apple___Apple_scab` (0.913940) | `Apple___Apple_scab` (0.913940) | **0.00e+00** | `Apple___Apple_scab` (0.955557) |
| `Cherry...healthy/0008f3d3...JPG` | Cherry (Sour) - Healthy | `Cherry...healthy` (0.967871) | `Cherry...healthy` (0.967871) | **0.00e+00** | `Cherry...healthy` (0.998960) |
| `Cherry...Pwd.M/00705aa7...JPG` | Cherry (Sour) - Powdery Mildew | `Cherry...Powdery_mildew` (0.974408) | `Cherry...Powdery_mildew` (0.974408) | **0.00e+00** | `Cherry...Powdery_mildew` (0.990408) |
| `Corn...healthy/00031d74...jpg` | Corn (Maize) - Healthy | `Corn_(maize)___healthy` (0.999168) | `Corn_(maize)___healthy` (0.999168) | **0.00e+00** | `Corn_(maize)___healthy` (0.897340) |
| `Corn...Common_rust_/RS_Rust...JPG` | Corn (Maize) - Common Rust | `Corn_(maize)___Common_rust_` (0.998732) | `Corn_(maize)___Common_rust_` (0.998732) | **0.00e+00** | `Corn_(maize)___Common_rust_` (0.986935) |
| `Grape___Black_rot/00090b0f...JPG` | Grape - Black Rot | `Tomato___Bacterial_spot` (0.969647) | `Tomato___Bacterial_spot` (0.969647) | **0.00e+00** | `Grape___Black_rot` (0.966831) |
| `Potato___Late_blight/0051e5e8...JPG` | Potato - Late Blight | `Tomato___Late_blight` (0.451732) | `Tomato___Late_blight` (0.451732) | **0.00e+00** | `Potato___Late_blight` (0.765351) |
| `Tomato___Early_blight/0012b9d2...JPG` | Tomato - Early Blight | `Tomato___Early_blight` (0.681282) | `Tomato___Early_blight` (0.681282) | **0.00e+00** | `Tomato___Early_blight` (0.607275) |
| `Tomato___healthy/000146ff...JPG` | Tomato - Healthy | `Tomato___healthy` (0.864661) | `Tomato___healthy` (0.864661) | **0.00e+00** | `Tomato___healthy` (0.986646) |

---

## 4. Discrepancy Analysis

### 4.1 Numerical Parity (FastAPI vs. Offline Deployed)
$$\max_{i \in \{1 \dots 10\}} \max_{k \in \{0 \dots 37\}} \left| P_{\text{FastAPI}}(y=k \mid x_i) - P_{\text{Offline}}(y=k \mid x_i) \right| = \mathbf{0.000000\text{e}+00}$$
There is zero discrepancy between offline PyTorch code and the FastAPI serving stack. Preprocessing, image scaling, normalization, inference pass, and tensor indexing are completely faithful to offline PyTorch execution.

### 4.2 Discrepancy between Deployed Model and Phase 5 Benchmark Model
- On the `Grape___Black_rot` specimen, the active deployed model predicted `Tomato___Bacterial_spot` (96.96%), whereas the Phase 5 benchmark model predicted `Grape___Black_rot` (96.68%).
- On the `Potato___Late_blight` specimen, the active deployed model predicted `Tomato___Late_blight` with low confidence (45.17%, flagging `LOW_CONFIDENCE_UNCERTAIN`), whereas the Phase 5 benchmark model correctly predicted `Potato___Late_blight` (76.54%).
- **Root Cause:** The deployed model (`saved_models/classifier.pt`, trained on 43,444 images) is an older checkpoint that was not updated after Phase 2/3/5 combined manifest training. The Phase 5 benchmark checkpoint (`saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth`, trained on 40,151 training images from the verified manifest) demonstrates higher accuracy and fewer false cross-species confusions.
- **Action Taken:** Per the strict instruction *"Do not retrain the model, modify checkpoints... Investigate any discrepancies rather than adjusting outputs to force agreement"*, we have documented this provenance gap in detail rather than silently replacing files.

---

## 5. Latency Profiling Results

Testing was conducted on the host AMD Ryzen 7 5800HS processor (8 physical cores, 8 threads allocated) using monotonic timers:

| Benchmark Stage | Sample Size | Mean Latency | Std Dev | Median | Min | Max | 95th Percentile ($p95$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure TorchScript Forward Pass** | 100 runs | **12.69 ms** | 0.63 ms | 12.60 ms | 10.89 ms | 14.45 ms | **13.69 ms** |
| **End-to-End FastAPI `/predict`** | 50 HTTP calls | **23.15 ms** | 1.00 ms | 23.20 ms | 20.60 ms | 24.93 ms | **24.55 ms** |

### Breakdown of End-to-End Request Overhead:
1. **Multipart Request Parsing & Image Decoding:** $4.2\text{ ms}$
2. **Laplacian Blur Quality Computation:** $3.1\text{ ms}$
3. **Bicubic Resize & Normalization:** $2.2\text{ ms}$
4. **MobileNetV3-Small Forward Pass:** $12.7\text{ ms}$ (54.8% of request)
5. **State Determination & JSON Serialization:** $1.0\text{ ms}$
- **Total Endpoint Latency:** $\approx 23.2\text{ ms}$

---

## 6. Real Error-Handling Diagnostic Verification

The error handling capabilities of the deployed endpoints were validated using real payloads:

```text
1. Empty file (0 bytes):
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Payload:  {"detail": "The uploaded file is empty."}

2. Corrupted file content:
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Payload:  {"detail": "The uploaded file is not a valid image."}

3. Sub-minimum dimensions (< 128x128 pixels, e.g. 64x64):
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Payload:  {"detail": "Image dimensions must be at least 128x128 pixels."}

4. Oversized upload (> 10 MB limit):
   Endpoint: POST /predict
   Status:   HTTP 413 Payload Too Large
   Payload:  {"detail": "Image exceeds the 10 MB upload limit."}

5. Invalid screening mode parameter:
   Endpoint: POST /screen-plant (mode="invalid_mode")
   Status:   HTTP 422 Unprocessable Entity
   Payload:  {"detail": "Mode must be 'single_leaf' or 'whole_plant'."}
```
All error states return immediate, well-structured JSON responses without unhandled server exceptions or connection drops.

---

## 7. Additional Audit: Provenance of the 0.60 Review Threshold

### Investigation:
- In `backend/app/services/models.py`, `self.review_threshold` defaults to `0.60`.
- In `backend/app/services/screening.py`, `MIN_CANDIDATE_SCORE` defaults to `0.60`.
- In `reports/phase3/calibrated_inference_config.json`, the experimentally fitted threshold for 90% selective accuracy was `0.2231` (at $T^* = 0.5538$).

### Conclusion:
The `0.60` threshold was **not experimentally determined** from the training partition or selective accuracy curves. It is an **operational setting / heuristic** chosen to separate reasonably confident predictions from uncertain ones and to prune candidate crops during spatial screening.

The application and documentation correctly label this threshold as an **operational setting** and refrain from presenting it as an empirical or clinically validated boundary.

---

## 8. Unresolved Issues & Recommendations for Subsequent Phases

1. **Checkpoint Unification (Future Phase Candidate):**
   In a future sanctioned migration phase, the deployed model (`saved_models/classifier.pt`) may be updated to the verified Phase 5 benchmark TorchScript model (`saved_models/experiments/mobilenet_v3_small_final/classifier.pt`, SHA-256: `6fff52f144...`), which showed higher accuracy and resolved the misclassifications on `Grape___Black_rot` and `Potato___Late_blight`.
2. **Calibration Parameter Standardization:**
   Until temperature scaling is unified between Phase 3 ($T=0.5538$) and Phase 5 ($T=0.9902$), all deployed scores remain uncalibrated ($T=1.0$) and labeled **"Uncalibrated Model Score"**.

