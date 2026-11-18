# Phase 6A.1: Test Results Report — Comprehensive Prediction-State Validation

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.1  
**Date:** November 14, 2026  
**Test Suite:** `backend/tests/test_prediction_states.py` + full regression suite (`backend/tests/`)  
**Overall Outcome:** 44 Passed, 0 Failed, 0 Regressions (100% Pass Rate)  

---

## 1. Test Suite Summary

The Phase 6A.1 test execution evaluated the 10 mandatory operational scenarios required by the project specification, alongside the full regression test suite covering health checks, quality checks, measurements, models, research, chat, and learning services.

```text
============================= test session starts =============================
platform win32 -- Python 3.12.5, pytest-9.2.0, pluggy-1.7.0
rootdir: E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection
collected 44 items

backend/tests/test_chat.py .........                                     [ 20%]
backend/tests/test_health.py ...                                         [ 27%]
backend/tests/test_learning.py ....                                      [ 36%]
backend/tests/test_measurements.py ...                                   [ 43%]
backend/tests/test_models.py ...                                         [ 50%]
backend/tests/test_prediction_states.py ..........                       [ 72%]
backend/tests/test_quality.py .                                          [ 75%]
backend/tests/test_research.py .....                                     [ 86%]
backend/tests/test_screening.py ......                                   [100%]

======================= 44 passed, 6 warnings in 9.08s ========================
```

---

## 2. Detailed Results across 10 Mandatory Scenarios

### Scenario 1: High-Confidence Healthy Prediction
- **Input Specimen:** `Apple___healthy` forward inference with uncalibrated score = 92.0% ($\ge 0.60$).
- **Expected Outcome:** State = `HIGH_CONFIDENCE`, Status = `Model Prediction`, Tone = `good`, Display Label = `Apple - Healthy`, Score Label = `Uncalibrated Model Score`, Uncertainty flag = `False`.
- **Observed Result:** 
  - `prediction_state`: `HIGH_CONFIDENCE`
  - `prediction_status`: `Model Prediction`
  - `state_tone`: `good`
  - `display_label`: `Apple - Healthy`
  - `score_label`: `Uncalibrated Model Score`
  - `is_uncertain`: `False`
  - `is_healthy`: `True`
- **Status:** **PASS**

---

### Scenario 2: High-Confidence Diseased Prediction
- **Input Specimen:** `Tomato___Early_blight` forward inference with uncalibrated score = 88.0% ($\ge 0.60$).
- **Expected Outcome:** State = `HIGH_CONFIDENCE`, Status = `Model Prediction`, Tone = `good`, Display Label = `Tomato - Early Blight`, Score Label = `Uncalibrated Model Score`, Uncertainty flag = `False`.
- **Observed Result:**
  - `prediction_state`: `HIGH_CONFIDENCE`
  - `prediction_status`: `Model Prediction`
  - `state_tone`: `good`
  - `display_label`: `Tomato - Early Blight`
  - `score_label`: `Uncalibrated Model Score`
  - `is_uncertain`: `False`
  - `is_healthy`: `False`
- **Status:** **PASS**

---

### Scenario 3: Low-Confidence Healthy Prediction (User Issue Reproduction)
- **Input Specimen:** `Cherry_(including_sour)___healthy` forward inference with uncalibrated score = 28.0% ($< 0.60$), matching user-reported screenshot.
- **Expected Outcome:** State = `LOW_CONFIDENCE_UNCERTAIN`, Status = `Uncertain Prediction`, Tone = `amber`, Display Label = `Cherry (Sour) - Healthy`, Headline = `Uncertain Prediction · Tentative match: Cherry (Sour) - Healthy`, Uncertainty flag = `True`, Needs review = `True`. Explicit healthy caution alert triggered.
- **Observed Result:**
  - `prediction_state`: `LOW_CONFIDENCE_UNCERTAIN`
  - `prediction_status`: `Uncertain Prediction`
  - `state_tone`: `amber`
  - `display_label`: `Cherry (Sour) - Healthy`
  - `score_label`: `Uncalibrated Model Score`
  - `is_uncertain`: `True`
  - `is_healthy`: `True`
  - `headline`: `Uncertain Prediction · Tentative match: Cherry (Sour) - Healthy`
  - `explanation`: *"The uncalibrated model score (28.0%) is below the operational review threshold (60%). The model could not produce a sufficiently reliable classification for this specimen. Tentative predictions must not be treated as a confirmed diagnosis or confirmation of plant health."*
- **Status:** **PASS**

---

### Scenario 4: Low-Confidence Diseased Prediction
- **Input Specimen:** `Apple___Apple_scab` forward inference with uncalibrated score = 45.0% ($< 0.60$).
- **Expected Outcome:** State = `LOW_CONFIDENCE_UNCERTAIN`, Status = `Uncertain Prediction`, Tone = `amber`, Display Label = `Apple - Apple Scab`, Headline = `Uncertain Prediction · Tentative match: Apple - Apple Scab`, Uncertainty flag = `True`.
- **Observed Result:**
  - `prediction_state`: `LOW_CONFIDENCE_UNCERTAIN`
  - `prediction_status`: `Uncertain Prediction`
  - `state_tone`: `amber`
  - `display_label`: `Apple - Apple Scab`
  - `score_label`: `Uncalibrated Model Score`
  - `is_uncertain`: `True`
  - `is_healthy`: `False`
- **Status:** **PASS**

---

### Scenario 5: No Candidate Screening Regions
- **Input Specimen:** Image scan where all sliding-window patches produce healthy top-1 predictions or scores $< 0.60$.
- **Expected Outcome:** Screening State = `NO_CANDIDATES_IDENTIFIED`, Status = `No Candidate Regions Identified`, Tone = `neutral`, Candidate count = 0, Safety disclaimer explicitly warns absence of candidate crops does not mean leaf is healthy.
- **Observed Result:**
  - `screening_state`: `NO_CANDIDATES_IDENTIFIED`
  - `screening_status`: `No Candidate Regions Identified`
  - `screening_tone`: `neutral`
  - `candidate_count`: `0`
  - `regions_checked`: `9` (or `12` depending on image dimensions)
  - `safety_disclaimer`: *"Important Safety Notice: Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free. Candidate screening only detects conspicuous foliar patches matching trained disease categories above the threshold."*
- **Status:** **PASS**

---

### Scenario 6: Multiple Candidate Screening Regions
- **Input Specimen:** Specimen with multiple diseased crops crossing the 0.60 threshold (`Tomato___Early_blight` 85%, `Apple___Apple_scab` 78%).
- **Expected Outcome:** Screening State = `CANDIDATES_IDENTIFIED`, Status = `X Candidate Region(s) Identified`, Tone = `info`, Candidate count $\ge 1$, candidate bounding boxes, scores, and safe action guidance populated.
- **Observed Result:**
  - `screening_state`: `CANDIDATES_IDENTIFIED`
  - `screening_status`: `2 Candidate Region(s) Identified`
  - `screening_tone`: `info`
  - `candidate_count`: `2`
  - `candidate_regions[0].display_label`: `Tomato - Early Blight`
  - `candidate_regions[0].score_label`: `Uncalibrated Model Score`
  - `candidate_regions[0].box`: valid `{left, top, right, bottom}` integers
  - `candidate_regions[0].image`: valid base64 data URL
- **Status:** **PASS**

---

### Scenario 7: Invalid Image Upload
- **Input Specimen:** 
  1. Empty payload (0 bytes).
  2. Corrupted file content (`NOT AN IMAGE FILE CONTENT`).
- **Expected Outcome:** HTTP 422 Unprocessable Entity with clear, actionable validation error detail.
- **Observed Result:**
  - Empty file: HTTP 422, `detail: "The uploaded file is empty."`
  - Corrupt file: HTTP 422, `detail: "The uploaded file is not a valid image."`
- **Status:** **PASS**

---

### Scenario 8: Unsupported Image Format
- **Input Specimen:** Text file upload (`text/plain` MIME type with non-image content).
- **Expected Outcome:** HTTP 422 Unprocessable Entity rejecting unparseable format.
- **Observed Result:** HTTP 422, `detail: "The uploaded file is not a valid image."`
- **Status:** **PASS**

---

### Scenario 9: Backend Inference Failure
- **Input Specimen:** Neural network execution throwing simulated runtime error (`ModelUnavailableError`).
- **Expected Outcome:** Server captures exception and raises HTTP 503 Service Unavailable without crashing worker process.
- **Observed Result:** HTTP 503, `detail: "Simulated neural network runtime inference failure."`
- **Status:** **PASS**

---

### Scenario 10: Multi-Aspect Image Dimensions
- **Input Specimen:** Testing sliding window generation across diverse aspect ratios:
  - Landscape: $640 \times 360$ px
  - Portrait: $360 \times 640$ px
  - High-resolution: $800 \times 600$ px
  - Tall strip: $256 \times 512$ px
- **Expected Outcome:** Windows are properly distributed without out-of-bounds crops; overlay image matches input dimensions; candidate count and regions checked correctly computed.
- **Observed Result:** All dimensions processed cleanly; `overlay` generated as valid JPEG data URL; sliding window coordinates strictly within $[0, \text{width}] \times [0, \text{height}]$.
- **Status:** **PASS**

---

## 3. Frontend Build & Regression Validation

- **Vite Production Build:**
  ```text
  vite v6.4.5 building for production...
  transforming...
  ✓ 27 modules transformed.
  rendering chunks...
  computing gzip size...
  dist/index.html                   0.51 kB │ gzip:  0.31 kB
  dist/assets/index-B9O5wCAr.css   29.48 kB │ gzip:  7.19 kB
  dist/assets/index-B1Lx3FGb.js   202.52 kB │ gzip: 63.23 kB
  ✓ built in 867ms
  ```
- **Live Server Regression Check:**
  - Live backend daemon (FastAPI on `http://127.0.0.1:8000`) verified via `/health`.
  - Live frontend dev server (Vite on `http://127.0.0.1:5173`) running and hot-reloading components.

---

## 4. UI Comparison Before and After Phase 6A.1

| Aspect | Prior Presentation (Defective) | Improved Phase 6A.1 Presentation |
| :--- | :--- | :--- |
| **Section Layout** | Single unorganized vertical column mixing patch-screening with whole-leaf prediction. | **Two clearly demarcated sections:** Section A (Whole-Leaf Classification) and Section B (Candidate-Region Screening). |
| **28% Low-Confidence Result** | Displayed in a green-bordered card with raw string `Cherry_(including_sour)___healthy`, easily mistaken as verified healthy. | Prominently designated **"Uncertain Prediction"** in an amber card with title `Tentative match: Cherry (Sour) - Healthy`, accompanied by an explicit warning alert. |
| **No Candidates Found** | Displayed as "No regions met the candidate threshold" at the top of the screen with no context. | Displayed in Section B with status pill **"No Candidate Regions Identified"**, metrics breakdown, and mandatory safety notice: *"Lack of candidate regions does not establish that the leaf or plant is healthy or disease-free."* |
| **Class Names** | Raw code strings: `Pepper,_bell___Bacterial_spot`, `Corn_(maize)___Common_rust_`. | Clean academic labels: `Bell Pepper - Bacterial Spot`, `Corn (Maize) - Common Rust`, with internal codes preserved in muted subtitle. |
| **Score Label** | Ambiguous generic text: `28% uncalibrated score`. | Explicit scientific label: `28% Uncalibrated Model Score`. |

---

## 5. Conclusion

All 10 required test scenarios passed without failure. The application is now in a substantially more reliable, professional, and scientifically cautious state, with zero ambiguity surrounding uncertain model predictions and candidate-region screening.

