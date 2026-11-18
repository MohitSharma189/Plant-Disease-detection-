# Phase 6A.2.1: Inference Verification Report — Comparative Parity & Deployment Verification

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.2.1  
**Date:** November 14, 2026  
**Auditor / Verification Engineer:** Research & Systems Engineering Agent  
**Host Environment:** Windows CPU Host, AMD Ryzen 7 5800HS (8 Physical Cores, 16 Threads)  
**Execution Mode:** Pure PyTorch CPU Execution & FastAPI Asynchronous Test Server  

---

## 1. Scope & Verification Protocol

Phase 6A.2.1 verifies that the FastAPI application successfully loads and serves the verified Phase 5 MobileNetV3-Small TorchScript model (`saved_models/experiments/mobilenet_v3_small_final/classifier.pt`, SHA-256: `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`).

### Strict Verification Rules:
1. **Real Specimens & Weights:**
   All comparative evaluations were performed using real leaf specimens from `data/processed/PlantVillage/` and the actual trained TorchScript checkpoint.
2. **Preservation:**
   The historical baseline checkpoint (`saved_models/classifier.pt`, SHA-256: `1aef041f...`) and original Phase 5 benchmark checkpoint (`best_model.pth`, SHA-256: `6190f54c...`) were preserved untouched.
3. **Parity Criterion:**
   Numerical agreement across all 38 class probabilities between the deployed FastAPI endpoint and direct offline PyTorch execution must meet floating-point precision ($\max |\Delta| < 10^{-5}$).

---

## 2. Checkpoint Cryptographic Provenance

| Checkpoint Name | Disk Location | SHA-256 Digest | Size (Bytes) | Role / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Historical Baseline Checkpoint** | `saved_models/classifier.pt` | `1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f` | 6,527,514 | Preserved untouched |
| **Phase 5 Benchmark (pth)** | `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | 6,353,709 | Preserved untouched |
| **Phase 5 TorchScript (pt)** | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` | 6,531,482 | **Active Deployed Model** |

### FastAPI Loaded Checkpoint Verification:
* **Reported Loaded Path:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`
* **Computed Checkpoint SHA-256:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`
* **Loaded Architecture:** `MobileNetV3-Small` (1,556,806 parameters)
* **Calibration State:** $T = 1.0$, `calibration_report = None`, `confidence_calibrated = False`
* **Operational Review Threshold:** $0.60$ (heuristic setting)

---

## 3. Real Specimen Comparative Inference Test

Ten real leaf specimens were evaluated across:
1. **Pipeline A (Offline Phase 5 TorchScript):** Direct offline forward pass of `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` using standard torchvision transforms (`Resize((224, 224))`, `ToTensor()`, `Normalize(...)`, and uncalibrated `softmax`).
2. **Pipeline B (FastAPI Live Endpoint):** Uploaded via HTTP multipart request to the FastAPI `/predict` endpoint.

### Detailed Specimen Results:

```text
================================================================================
Specimen: Apple___healthy (0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG)
  Ground Truth:               Apple___healthy
  Offline Phase 5 Prediction: Apple___healthy                     Score: 0.989913
  FastAPI Live Prediction:    Apple___healthy                     Score: 0.989913
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Apple___Apple_scab (00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG)
  Ground Truth:               Apple___Apple_scab
  Offline Phase 5 Prediction: Apple___Apple_scab                  Score: 0.955557
  FastAPI Live Prediction:    Apple___Apple_scab                  Score: 0.955557
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Cherry_(including_sour)___healthy (0008f3d3-2f85-4973-be9a-1b520b8b59fc___JR_HL 4092.JPG)
  Ground Truth:               Cherry_(including_sour)___healthy
  Offline Phase 5 Prediction: Cherry_(including_sour)___healthy   Score: 0.998960
  FastAPI Live Prediction:    Cherry_(including_sour)___healthy   Score: 0.998960
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Cherry_(including_sour)___Powdery_mildew (00705aa7-5ea2-4419-9440-8ba65e108eb9___FREC_Pwd.M 0267.JPG)
  Ground Truth:               Cherry_(including_sour)___Powdery_mildew
  Offline Phase 5 Prediction: Cherry_(including_sour)___Powdery_mildew Score: 0.990408
  FastAPI Live Prediction:    Cherry_(including_sour)___Powdery_mildew Score: 0.990408
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Corn_(maize)___healthy (00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL 8315 copy 2.jpg)
  Ground Truth:               Corn_(maize)___healthy
  Offline Phase 5 Prediction: Corn_(maize)___healthy              Score: 0.897340
  FastAPI Live Prediction:    Corn_(maize)___healthy              Score: 0.897340
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Corn_(maize)___Common_rust_ (RS_Rust 1563.JPG)
  Ground Truth:               Corn_(maize)___Common_rust_
  Offline Phase 5 Prediction: Corn_(maize)___Common_rust_         Score: 0.986935
  FastAPI Live Prediction:    Corn_(maize)___Common_rust_         Score: 0.986935
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Grape___Black_rot (00090b0f-c140-4e77-8d20-d39f67b75fcc___FAM_B.Rot 0376.JPG)
  Ground Truth:               Grape___Black_rot
  Offline Phase 5 Prediction: Grape___Black_rot                   Score: 0.966831
  FastAPI Live Prediction:    Grape___Black_rot                   Score: 0.966831
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Potato___Late_blight (0051e5e8-d1c4-4a84-bf3a-a426cdad6285___RS_LB 4640.JPG)
  Ground Truth:               Potato___Late_blight
  Offline Phase 5 Prediction: Potato___Late_blight                Score: 0.765351
  FastAPI Live Prediction:    Potato___Late_blight                Score: 0.765351
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Tomato___Early_blight (0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG)
  Ground Truth:               Tomato___Early_blight
  Offline Phase 5 Prediction: Tomato___Early_blight               Score: 0.607275
  FastAPI Live Prediction:    Tomato___Early_blight               Score: 0.607275
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
--------------------------------------------------------------------------------
Specimen: Tomato___healthy (000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG)
  Ground Truth:               Tomato___healthy
  Offline Phase 5 Prediction: Tomato___healthy                    Score: 0.986646
  FastAPI Live Prediction:    Tomato___healthy                    Score: 0.986646
  Prediction State:           HIGH_CONFIDENCE                     Max |Delta|: 0.000000e+00
================================================================================
OVERALL MAXIMUM ABSOLUTE DIFFERENCE: 0.000000e+00
```

### Analytical Verification:
1. **Bit-for-Bit Equivalence:** Across all 10 specimens and all 38 output classes, the maximum difference between offline inference and FastAPI serving was exactly **$0.000000\text{e}+00$**.
2. **Correction of Baseline Errors:**
   - Under the previous baseline checkpoint in Phase 6A.2, `Grape___Black_rot` was misclassified as `Tomato___Bacterial_spot` (96.96%). With the Phase 5 benchmark checkpoint deployed, it is correctly classified as `Grape___Black_rot` with **96.68%** confidence.
   - Under the baseline checkpoint, `Potato___Late_blight` was misclassified as `Tomato___Late_blight` (45.17%, uncertain). With the Phase 5 benchmark checkpoint deployed, it is correctly classified as `Potato___Late_blight` with **76.54%** confidence (`HIGH_CONFIDENCE`).

---

## 4. Verification Across Prediction State Categories

The behavior of the deployed system across the defined prediction state categories was verified:

### 4.1 High-Confidence Healthy Predictions
- Verified on specimens: `Apple___healthy` (98.99%), `Cherry_(including_sour)___healthy` (99.90%), `Corn_(maize)___healthy` (89.73%), `Tomato___healthy` (98.66%).
- Prediction State: `HIGH_CONFIDENCE`
- `needs_review`: `False`
- `is_uncertain`: `False`
- Response Tone: `good`

### 4.2 High-Confidence Diseased Predictions
- Verified on specimens: `Apple___Apple_scab` (95.56%), `Cherry_(including_sour)___Powdery_mildew` (99.04%), `Corn_(maize)___Common_rust_` (98.69%), `Grape___Black_rot` (96.68%), `Potato___Late_blight` (76.54%), `Tomato___Early_blight` (60.73%).
- Prediction State: `HIGH_CONFIDENCE`
- `needs_review`: `False`
- `is_uncertain`: `False`
- Response Tone: `good`

### 4.3 Low-Confidence Diseased Predictions
- Evaluated real specimen: `Apple___Apple_scab/2d5265d2-be99-4e9f-a085-d309cc01b751___FREC_Scab 3023.JPG`.
- Model output: `Pepper,_bell___Bacterial_spot` with uncalibrated score **$49.32\%$**.
- State Decision: Because $49.32\% < 60.0\%$ (the operational review threshold), the backend correctly assigned:
  - `prediction_state`: `"LOW_CONFIDENCE_UNCERTAIN"`
  - `prediction_status`: `"Uncertain Prediction"`
  - `state_tone`: `"amber"`
  - `needs_review`: `True`
  - `is_uncertain`: `True`
  - `explanation`: *"The uncalibrated model score (49.3%) is below the operational review threshold (60%). The model could not produce a sufficiently reliable classification for this specimen..."*

### 4.4 Low-Confidence Healthy Predictions
- Evaluated real specimen: `Apple___Apple_scab/3652fd23-ba4c-4958-8db8-3aff8f614dfd___FREC_Scab 3377.JPG`.
- Model output: `Apple___healthy` with uncalibrated score **$54.53\%$**.
- State Decision: Because $54.53\% < 60.0\%$, the backend correctly assigned:
  - `prediction_state`: `"LOW_CONFIDENCE_UNCERTAIN"`
  - `prediction_status`: `"Uncertain Prediction"`
  - `state_tone`: `"amber"`
  - `needs_review`: `True`
  - `is_uncertain`: `True`
  - Safe Warning: The UI and API explicitly prevent a low-confidence healthy guess from falsely reassuring the user.

---

## 5. Screening Endpoint Consistency (`/screen-plant`)

The screening pipeline was verified on the deployed system:
* **Single-Leaf Screening (`mode="single_leaf"`):**
  - Checked 9 multi-scale overlapping foliar patches.
  - Returned `screening_status = "No Candidate Regions Identified"` (0 candidates above the 0.60 threshold).
  - Clean spatial overlay generated and returned.
* **Whole-Plant Screening (`mode="whole_plant"`):**
  - Executed $3 \times 3$ grid proposal generation.
  - Returned `screening_status = "No Candidate Regions Identified"`.
  - All institutional source attributions, limitations, and safety disclaimers cleanly populated.

---

## 6. Error-Handling Diagnostics

Verified against real API calls using the FastAPI test client:

```text
1. 0-byte Empty File:
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Detail:   {"detail": "The uploaded file is empty."}

2. Corrupt Non-Image Data:
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Detail:   {"detail": "The uploaded file is not a valid image."}

3. Sub-Minimum Image Size (< 128x128 px, e.g. 64x64):
   Endpoint: POST /predict
   Status:   HTTP 422 Unprocessable Entity
   Detail:   {"detail": "Image dimensions must be at least 128x128 pixels."}

4. Oversized File (> 10 MB limit):
   Endpoint: POST /predict
   Status:   HTTP 413 Payload Too Large
   Detail:   {"detail": "Image exceeds the 10 MB upload limit."}

5. Invalid Screening Mode:
   Endpoint: POST /screen-plant (mode="invalid_mode_xyz")
   Status:   HTTP 422 Unprocessable Entity
   Detail:   {"detail": "Mode must be 'single_leaf' or 'whole_plant'."}
```

---

## 7. Fresh CPU Latency Measurements

Measured on AMD Ryzen 7 5800HS (8 physical cores, 8 threads allocated) using high-resolution monotonic clocks:

### 7.1 Single-Image Pure Forward Pass (100 runs)
- **Mean Latency:** **$12.38\text{ ms}$**
- **Standard Deviation:** **$1.30\text{ ms}$**
- **Median Latency:** **$12.58\text{ ms}$**
- **Minimum Latency:** **$9.84\text{ ms}$**
- **Maximum Latency:** **$16.58\text{ ms}$**
- **95th Percentile ($p95$):** **$13.98\text{ ms}$**
- **99th Percentile ($p99$):** **$16.00\text{ ms}$**

### 7.2 End-to-End FastAPI `/predict` Endpoint (50 real HTTP requests)
- **Mean Latency:** **$22.07\text{ ms}$**
- **Standard Deviation:** **$1.45\text{ ms}$**
- **Median Latency:** **$21.84\text{ ms}$**
- **Minimum Latency:** **$19.38\text{ ms}$**
- **Maximum Latency:** **$25.69\text{ ms}$**
- **95th Percentile ($p95$):** **$24.27\text{ ms}$**
- **99th Percentile ($p99$):** **$25.15\text{ ms}$**

---

## 8. Summary of Deployed Status

1. **Active Checkpoint:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` (SHA-256: `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`).
2. **Historical Checkpoints:** `saved_models/classifier.pt` and `mobilenet_v3_small/best_model.pth` remain untouched.
3. **Parity:** $\max |\Delta| = 0.000000\text{e}+00$ between offline Phase 5 evaluation and live FastAPI endpoints.
4. **Accuracy:** Successfully resolved previous baseline misclassifications on `Grape___Black_rot` and `Potato___Late_blight`.
5. **Calibration:** Enforced $T = 1.0$, `confidence_calibrated = False`, labeled `"Uncalibrated Model Score"`.
6. **Threshold:** Operational review threshold maintained at $0.60$.
7. **Test Suite:** 44 tests passed, 0 failures.

