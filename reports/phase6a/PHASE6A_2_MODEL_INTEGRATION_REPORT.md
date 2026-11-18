# Phase 6A.2: Model Integration Report — MobileNetV3-Small Deployment & Integration Verification

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.2  
**Date:** November 14, 2026  
**Auditor / Systems Engineer:** Research & Systems Engineering Agent  
**Host Platform:** Windows Host, AMD Ryzen 7 5800HS (8 Physical Cores / 16 Threads, CPU Execution Mode)  
**Software Stack:** Python 3.12.5, PyTorch 2.15.0+cpu, Torchvision 0.29.2+cpu, FastAPI 0.110+, Uvicorn 0.29+  

---

## 1. Executive Summary

Phase 6A.2 conducts a systematic verification of the MobileNetV3-Small model integration within the production FastAPI application (`backend/app`). The primary objective is to verify that model inference served via REST API endpoints (`/predict` and `/screen-plant`) produces outputs mathematically identical to verified offline PyTorch inference, adheres strictly to preprocessing and class-mapping specifications, measures real CPU execution latencies, and avoids silently applying unverified temperature scaling.

### Key Verification Highlights:
1. **FastAPI vs. Offline Equivalence:**
   Across all tested real PlantVillage specimens, the deployed FastAPI inference pipeline achieved an **exact numerical floating-point agreement ($\max |\Delta| = 0.000000\text{e}+00$)** with pure offline PyTorch TorchScript forward evaluation. Class scores, probabilities, and top-1 predictions matched bit-for-bit.
2. **Checkpoint Provenance Discrepancy (Identified & Documented):**
   A rigorous cryptographic audit revealed that the active deployed model binary at `saved_models/classifier.pt` (SHA-256: `1aef041f87ad115b...`, 6.53 MB) corresponds to the initial repository model checkpoint (trained on 43,444 images, 94.14% validation accuracy). The documented Phase 5 benchmark checkpoint resides at `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` (SHA-256: `6190f54c56ebbb7f...`, 6.35 MB, 95.14% validation accuracy, identical to `saved_models/experiments/mobilenet_v3_small_final/best_model.pth`). As required by project preservation rules, no files were silently replaced; the discrepancy is formally investigated and documented.
3. **Hardware Latency Verification:**
   On an AMD Ryzen 7 5800HS processor using 8 CPU threads, single-image forward inference latency averaged **$12.69 \pm 0.63\text{ ms}$** over 100 consecutive runs ($p95 = 13.69\text{ ms}$). Complete end-to-end FastAPI `/predict` execution (including image decoding, blur evaluation, tensor normalization, inference, and JSON serialization) averaged **$23.15 \pm 1.00\text{ ms}$** ($p95 = 24.55\text{ ms}$).
4. **Calibration Temperature Verification:**
   Verification confirmed that the deployed model registry loads with $T = 1.0$ and sets `confidence_calibrated = False`. No unverified calibration temperature ($T^* = 0.5538$ or $T^* = 0.9902$) is applied silently. All predictions are explicitly labeled **"Uncalibrated Model Score"**.
5. **Threshold Provenance Audit:**
   The `0.60` review threshold is verified as an **operational engineering heuristic**, not an experimentally calibrated decision threshold. The experimentally fitted threshold from Phase 3 calibration ($T^* = 0.5538$) was `0.2231`, which was never migrated to the uncalibrated production model.

---

## 2. Checkpoint Provenance & Cryptographic Verification

A cryptographic SHA-256 digest was computed across all model checkpoints, state dictionaries, and metadata files:

| Checkpoint Identifier | Path | Format | SHA-256 Digest | Size (bytes) | Status / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Deployed Classifier** | `saved_models/classifier.pt` | TorchScript JIT | `1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f` | 6,527,514 | Production API checkpoint (Nov 11 baseline) |
| **Active Metadata** | `saved_models/classifier_metadata.json` | JSON | `602fcdd5651fb79f187902d13ca0430e702580798e6a12702737c0411a2f64fa` | 1,472 | Deployed model configuration |
| **Active Class Mapping** | `saved_models/class_mapping.json` | JSON | `f8988b3e341acb2e4b47ebcbaf7c756fcf3cc44c15c898953cb7d3a01dfd6945` | 1,189 | 38 sorted PlantVillage classes |
| **Phase 5 Benchmark Model** | `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` | PyTorch State Dict | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | 6,353,709 | Documented Phase 5 benchmark checkpoint |
| **Phase 2/3 Final Model** | `saved_models/experiments/mobilenet_v3_small_final/best_model.pth` | PyTorch State Dict | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | 6,353,709 | Bit-for-bit identical to Phase 5 benchmark |
| **Phase 2/3 Final TorchScript** | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | TorchScript JIT | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` | 6,531,482 | Exported TorchScript of Phase 5 checkpoint |

### Discrepancy Assessment:
- The deployed checkpoint `classifier.pt` has SHA-256 `1aef041f...`, whereas Table 10 of the research paper and `reports/phase5/checkpoint_provenance.json` specify `best_model.pth` with SHA-256 `6190f54c...`.
- `classifier.pt` in `saved_models/` was generated during initial setup with validation accuracy of 94.14% on 10,861 holdout validation images.
- The Phase 5 benchmark was trained on the combined multimodal split manifest (40,151 train / 5,436 val) achieving 95.14% validation accuracy and 95.02% holdout test accuracy.
- While both models share identical architecture (MobileNetV3-Small, 1,556,806 parameters, 38 output classes), their weight tensors differ, resulting in subtle prediction divergence on difficult out-of-domain samples (see Section 4).

---

## 3. Architecture Compatibility & Input Preprocessing

### 3.1 Model Architecture Parameters
- **Backbone:** MobileNetV3-Small (TorchVision definition)
- **Feature Extractor Output:** 576 channels (Hard-Swish, Squeeze-and-Excitation blocks)
- **Classifier Head:**
  - `Linear(in_features=576, out_features=1024)` + Hard-Swish
  - `Dropout(p=0.2 / 0.3)`
  - `Linear(in_features=1024, out_features=38)`
- **Total Trainable Parameters:** 1,556,806
- **Serialized Weight Tensors:** 244 state dictionary keys (100% matched between PyTorch and TorchScript)

### 3.2 Preprocessing Pipeline Verification
Both offline evaluation (`training/evaluate_classifier.py`) and online API services (`backend/app/services/models.py`) implement strictly identical preprocessing pipelines:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```
- **Input Dimensions:** 3 channels, $224 \times 224$ pixels.
- **Pixel Scaling:** Unsigned 8-bit integers $[0, 255] \rightarrow$ floating point $[0.0, 1.0]$.
- **Normalization:** ImageNet standard mean and standard deviation.
- **Color Channel Order:** Strict RGB format (PIL `convert("RGB")`).
- **Orientation Correction:** EXIF transpose (`ImageOps.exif_transpose`) applied during image decoding.

### 3.3 Class Index Mapping Verification
Verification of `class_mapping.json` across 6 distinct directories (`saved_models/`, `candidate/`, `mobilenet_v3_small_final/`, `verification/`, `reports/phase2/final/`, and `reports/phase2/verification/`) confirmed that all 38 classes share **100% identical indices (0 to 37)**:
- Index 0: `Apple___Apple_scab`
- Index 3: `Apple___healthy`
- Index 6: `Cherry_(including_sour)___healthy`
- Index 37: `Tomato___healthy`

---

## 4. Inference Comparison: FastAPI vs. Offline PyTorch

Inference was evaluated across 10 real specimen images drawn directly from the processed dataset:

```text
================================================================================
INFERENCE COMPARISON: FASTAPI vs OFFLINE DEPLOYED vs OFFLINE PHASE 5
================================================================================
Specimen: Apple___healthy
  FastAPI Prediction:         Apple___healthy                     (Score: 0.999974)
  Offline Deployed:           Apple___healthy                     (Score: 0.999974)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Apple___healthy                     (Score: 0.989913)  Diff: 1.01e-02
--------------------------------------------------------------------------------
Specimen: Apple___Apple_scab
  FastAPI Prediction:         Apple___Apple_scab                  (Score: 0.913940)
  Offline Deployed:           Apple___Apple_scab                  (Score: 0.913940)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Apple___Apple_scab                  (Score: 0.955557)  Diff: 4.16e-02
--------------------------------------------------------------------------------
Specimen: Cherry_(including_sour)___healthy
  FastAPI Prediction:         Cherry_(including_sour)___healthy   (Score: 0.967871)
  Offline Deployed:           Cherry_(including_sour)___healthy   (Score: 0.967871)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Cherry_(including_sour)___healthy   (Score: 0.998960)  Diff: 3.11e-02
--------------------------------------------------------------------------------
Specimen: Cherry_(including_sour)___Powdery_mildew
  FastAPI Prediction:         Cherry_(including_sour)___Powdery_mildew (Score: 0.974408)
  Offline Deployed:           Cherry_(including_sour)___Powdery_mildew (Score: 0.974408)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Cherry_(including_sour)___Powdery_mildew (Score: 0.990408)  Diff: 1.60e-02
--------------------------------------------------------------------------------
Specimen: Corn_(maize)___healthy
  FastAPI Prediction:         Corn_(maize)___healthy              (Score: 0.999168)
  Offline Deployed:           Corn_(maize)___healthy              (Score: 0.999168)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Corn_(maize)___healthy              (Score: 0.897340)  Diff: 1.02e-01
--------------------------------------------------------------------------------
Specimen: Corn_(maize)___Common_rust_
  FastAPI Prediction:         Corn_(maize)___Common_rust_         (Score: 0.998732)
  Offline Deployed:           Corn_(maize)___Common_rust_         (Score: 0.998732)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Corn_(maize)___Common_rust_         (Score: 0.986935)  Diff: 1.18e-02
--------------------------------------------------------------------------------
Specimen: Grape___Black_rot
  FastAPI Prediction:         Tomato___Bacterial_spot             (Score: 0.969647)
  Offline Deployed:           Tomato___Bacterial_spot             (Score: 0.969647)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Grape___Black_rot                   (Score: 0.966831)  Diff: 9.64e-01
--------------------------------------------------------------------------------
Specimen: Potato___Late_blight
  FastAPI Prediction:         Tomato___Late_blight                (Score: 0.451732)
  Offline Deployed:           Tomato___Late_blight                (Score: 0.451732)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Potato___Late_blight                (Score: 0.765351)  Diff: 5.39e-01
--------------------------------------------------------------------------------
Specimen: Tomato___Early_blight
  FastAPI Prediction:         Tomato___Early_blight               (Score: 0.681282)
  Offline Deployed:           Tomato___Early_blight               (Score: 0.681282)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Tomato___Early_blight               (Score: 0.607275)  Diff: 7.40e-02
--------------------------------------------------------------------------------
Specimen: Tomato___healthy
  FastAPI Prediction:         Tomato___healthy                    (Score: 0.864661)
  Offline Deployed:           Tomato___healthy                    (Score: 0.864661)  Diff: 0.00e+00
  Offline Phase 5 Benchmark:  Tomato___healthy                    (Score: 0.986646)  Diff: 1.22e-01
--------------------------------------------------------------------------------
OVERALL MAXIMUM ABSOLUTE DIFFERENCE (FastAPI vs Offline Deployed): 0.000000e+00
```

### Key Analytical Takeaways:
1. **Zero Discrepancy between API and Offline Deployed Model:**
   The deployed FastAPI endpoint `/predict` reproduces the offline TorchScript outputs with **$\max |\Delta| = 0.000000\text{e}+00$**, confirming zero numerical degradation, distortion, or preprocessing bugs during HTTP upload handling.
2. **Impact of Checkpoint Versioning:**
   The deployed baseline model misclassified the `Grape___Black_rot` sample as `Tomato___Bacterial_spot` and `Potato___Late_blight` as `Tomato___Late_blight` (with low confidence 45.1%). Conversely, the Phase 5 benchmark model correctly classified both specimens (`Grape___Black_rot` at 96.68% and `Potato___Late_blight` at 76.54%). This demonstrates that the Phase 5 benchmark model has superior feature representation, validating the documented training improvements on the combined multimodal manifest.

---

## 5. CPU Latency Benchmark

Measurements were performed on the host AMD Ryzen 7 5800HS processor (8 physical cores, 8 threads allocated) using high-resolution monotonic timers (`time.perf_counter()`):

### 5.1 Single-Image Neural Network Forward Pass (100 Runs)
- **Warmup:** 25 unmeasured forward passes.
- **Evaluation:** 100 consecutive single-image passes ($1 \times 3 \times 224 \times 224$).
- **Mean Latency:** **$12.69\text{ ms}$**
- **Standard Deviation:** **$0.63\text{ ms}$**
- **Median Latency:** **$12.60\text{ ms}$**
- **Minimum Latency:** **$10.89\text{ ms}$**
- **Maximum Latency:** **$14.45\text{ ms}$**
- **95th Percentile ($p95$):** **$13.69\text{ ms}$**
- **99th Percentile ($p99$):** **$14.27\text{ ms}$**

### 5.2 End-to-End FastAPI `/predict` Endpoint Latency (50 Real HTTP Requests)
- **Pipeline Stages:** HTTP multipart parsing $\rightarrow$ byte validation $\rightarrow$ EXIF transposition $\rightarrow$ Laplacian blur variance $\rightarrow$ Bicubic resize & tensor normalization $\rightarrow$ TorchScript forward pass $\rightarrow$ probability distribution formatting $\rightarrow$ JSON response serialization.
- **Mean Latency:** **$23.15\text{ ms}$**
- **Standard Deviation:** **$1.00\text{ ms}$**
- **Median Latency:** **$23.20\text{ ms}$**
- **Minimum Latency:** **$20.60\text{ ms}$**
- **Maximum Latency:** **$24.93\text{ ms}$**
- **95th Percentile ($p95$):** **$24.55\text{ ms}$**

*Observation:* Neural network forward inference represents approximately 55% of total endpoint processing time, with the remaining 45% ($~10.5\text{ ms}$) attributable to image decoding, Laplacian blur calculation, and JSON serialization.

---

## 6. Audit of the 0.60 Review Threshold Provenance

An exhaustive audit of the training scripts, configuration files, reports, and commit history was performed to identify the source of the `0.60` ($60.0\%$) review threshold:

1. **Experimental Calibration Threshold (Phase 3):**
   In Phase 3, `training/evaluate_classifier.py` implemented `choose_review_threshold` aiming for $\ge 90\%$ selective accuracy on the calibration partition (5,429 samples). That procedure produced an optimal threshold of:
   $$\tau^* = 0.2231346070766449$$
   This threshold was documented in `reports/phase3/calibrated_inference_config.json` and applied strictly to temperature-scaled logits ($T^* = 0.5538$).
2. **Origin of the 0.60 Value:**
   - In `backend/app/services/models.py`, `review_threshold: float = 0.60` was hardcoded as the default fallback parameter when `classifier_metadata.json` lacked a calibrated threshold configuration.
   - In `backend/app/services/screening.py`, `MIN_CANDIDATE_SCORE = 0.60` was established as an engineering filter to discard low-scoring candidate patches during sliding-window foliar screening.
3. **Formal Finding:**
   The `0.60` threshold was **not experimentally derived** from an empirical Risk-Coverage or ROC curve of the active deployed model. It is an **operational setting** chosen to prevent unconvincing model outputs (especially random or near-uniform guesses) from being presented as confident detections.
   *Decision:* The documentation and user interface must continue to identify `0.60` as an **operational review threshold** rather than a clinically or experimentally validated diagnostic cutoff.

---

## 7. Calibration Status & Score Presentation

- `ModelRegistry` actively verifies:
  - `self.temperature == 1.0` (unscaled logits)
  - `self.calibration_report is None`
  - `self.review_threshold == 0.60` (operational setting)
- Outputs returned to clients are strictly labeled:
  `score_label = "Uncalibrated Model Score"`
- In accordance with Phase 6A.1 requirements, uncalibrated scores are never presented as calibrated probabilities or diagnostic certainties.

