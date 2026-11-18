# Phase 6A.1: Project Audit and Application Assessment Report

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Date of Audit:** November 14, 2026  
**Auditor:** Research & Systems Engineering Agent  
**Environment:** Windows CPU Host, Python 3.12.5, PyTorch 2.15.0+cpu, FastAPI 0.110+, React 18, Vite 5  

---

## 1. Executive Summary

Phase 6A.1 initiates the production and user-interface hardening cycle for the Plant Disease Screening System. The objective of this phase is to conduct an exhaustive technical audit of the existing full-stack system, identify critical communication and presentation gaps, formalize distinct model prediction states, and eliminate safety risks where low-confidence classifications (specifically healthy predictions) could be misinterpreted as verified disease-free diagnoses.

### Key Audit Findings:
1. **Safety Communication Vulnerability (Primary Issue):**
   When an uploaded specimen yields an uncalibrated low confidence score (e.g., 28% for `Cherry_(including_sour)___healthy`), and no candidate crops cross the 0.60 sliding-window threshold, the UI previously rendered a green-bordered result card for the whole-leaf classification alongside "No regions met the candidate threshold". This juxtaposition creates a critical risk that users (farmers, agronomists, or researchers) interpret the low-confidence guess and lack of candidate disease boxes as proof that the plant is healthy.
2. **Conflated Pipeline Stages:**
   Whole-leaf classification and candidate-region screening were presented in a single unstructured vertical stack without clear visual separation, obscuring the distinction between whole-image image categorization and patch-level foliar screening.
3. **Threshold and Calibration Provenance:**
   The active production model (`saved_models/classifier.pt`) operates with uncalibrated logits ($T = 1.0$), as `classifier_metadata.json` does not embed an evaluation report or temperature parameter. While Phase 3 reported an experimental $T^* = 0.5538$ (L-BFGS optimization on `best_model.pth`) and Phase 5 reported $T^* = 0.9902$ across comparative architectures, the deployed model has consistently reported uncalibrated scores. The review threshold in `backend/app/services/models.py` defaults to `0.60` ($60\%$), matching `MIN_CANDIDATE_SCORE = 0.60` in `backend/app/services/screening.py`.
4. **UI Design & Raw Labels:**
   Internal class identifiers (e.g., `Cherry_(including_sour)___healthy`, `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`) were exposed directly to end-users without human-readable display names, reducing legibility and academic polish.

---

## 2. Existing Project Structure

```text
E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection
├── backend/
│   ├── app/
│   │   ├── config.py              # Application settings and directory paths
│   │   ├── main.py                # FastAPI endpoints (/health, /predict, /screen-plant, etc.)
│   │   ├── preprocessing/
│   │   │   └── quality.py         # Laplacian blur detection and PIL image decoding
│   │   ├── schemas/
│   │   │   └── analysis.py        # Pydantic schemas (Health, Quality, Chat, Research, etc.)
│   │   └── services/
│   │       ├── chat.py            # Multilingual agricultural Q&A assistant
│   │       ├── learning.py        # Response-style preference recording
│   │       ├── measurements.py    # Metric tracking
│   │       ├── models.py          # ModelRegistry, TorchScript loading, single-image inference
│   │       ├── research.py        # Europe PMC scholarly search integration
│   │       └── screening.py       # Multi-scale sliding window & foliar screening pipeline
│   └── tests/                     # 34 pytest unit tests (all passing)
├── data/
│   ├── processed/
│   │   ├── PlantVillage/          # 38 isolated classes (54,274 images)
│   │   └── PlantDoc_Object_Detection/ # External benchmark (236 images, 452 annotations)
│   └── raw/                       # Source downloads
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Single-page application controlling image screening & UI
│   │   ├── main.jsx               # React DOM entry point
│   │   └── styles.css             # Architectural styling, color variables, layout rules
│   ├── package.json               # Vite + React configuration
│   └── vite.config.js             # Local dev server config
├── reports/
│   ├── phase1/ - phase5/          # Milestone reports and benchmark logs
│   ├── research_paper/            # Springer Nature LaTeX package and compilation assets
│   └── phase6a/                   # Phase 6A documentation (Audit, Implementation, Tests)
├── saved_models/
│   ├── classifier.pt              # TorchScript MobileNetV3-Small (1.56M params, 6.06 MB)
│   ├── class_mapping.json         # 38 sorted class labels
│   └── classifier_metadata.json   # Training metadata (image_size: 224, seed: 42)
└── training/                      # Benchmark training and evaluation scripts
```

---

## 3. Frontend & Backend Status

### 3.1 Backend Service Status
- **Framework:** FastAPI running on Uvicorn (`http://127.0.0.1:8000`).
- **Health Verification:** `/health` reports `status: ok`, `inference_ready: True`, `classifier_loaded: True`, `segmentation_loaded: False`.
- **Inference Pipeline:** Single forward pass on 224×224 normalized RGB tensor (`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`).
- **Screening Execution:** Multi-scale grid generation (36 windows max), candidate filtering at `score >= 0.60`, suppression of `healthy` class predictions from candidate crops, non-maximum suppression (IoU threshold 0.55), Grad-CAM model attribution.
- **Image Validation:** Rejects payloads $> 10\text{ MB}$, decodes PNG/JPEG/WEBP via PIL, verifies RGB conversion, computes Laplacian blur variance.

### 3.2 Frontend Application Status
- **Framework:** React 18 with Vite dev server (`http://127.0.0.1:5173`).
- **State Management:** Local React hooks (`useState`, `useRef`, `useMemo`) combined with client-side IndexedDB for offline history retention.
- **Visual Design:** Warm paper palette (`--paper: #f3f1e8`, `--ink: #17231d`, `--pine: #173a2b`, `--amber: #d9903c`), restrained typography with Aptos / Georgia / Cascadia Mono.

---

## 4. Model & Checkpoint Provenance

| Component | Verified Asset | Digest / Configuration | Operational Role |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | MobileNetV3-Small | 1.56M parameters, 6.06 MB TorchScript | Production classification and patch screening |
| **Active Checkpoint** | `saved_models/classifier.pt` | TorchScript JIT serialization | Deployed inference binary |
| **Class Mapping** | `saved_models/class_mapping.json` | 38 unique PlantVillage classes | Label index mapping |
| **Metadata** | `saved_models/classifier_metadata.json` | `image_size: 224`, `seed: 42`, `device: cpu` | Operational hyperparameter loading |
| **Review Threshold** | `backend/app/services/models.py` | `0.60` (60.0%) | Flagging uncertain predictions (`needs_review`) |
| **Screening Threshold** | `backend/app/services/screening.py` | `0.60` (60.0%) | Candidate crop selection (`MIN_CANDIDATE_SCORE`) |

---

## 5. Screening Pipeline Implementation

1. **Sliding Windows:**
   - In `single_leaf` mode: Window fraction = 0.52 of the shortest dimension, stride = 0.62 of the window width, yielding up to 36 candidate crops.
   - In `whole_plant` mode: Window fraction = 0.38, stride = 0.62.
2. **Foliar Candidate Suppression:**
   - Lines 214–215 of `screening.py`: Crops whose predicted label contains `healthy` or whose top-1 score is $< 0.60$ are skipped.
   - **Reason:** The screening pipeline is explicitly designed to propose *disease candidate regions* requiring inspection.
3. **PlantDoc Ground-Truth Distinction:**
   - Annotations in the external PlantDoc benchmark represent whole-leaf bounding boxes (236 images, 452 annotations), **not** lesion segmentation masks. The application must never characterize sliding-window bounding boxes as lesion segmentation.

---

## 6. Uncertainty Calibration Discrepancy Analysis

An explicit calibration discrepancy exists across project phases:
- **Phase 3 Report:** Documented $T^* = 0.5538$ obtained via 50-iteration L-BFGS optimization on `best_model.pth` over the 5,429-sample calibration partition, reducing ECE from 10.25% to 0.58%.
- **Phase 5 Report:** Documented $T^* = 0.9902$ for MobileNetV3-Small when re-evaluated in the multi-architecture comparative benchmark.
- **Active Deployment:** The production `saved_models/classifier.pt` and `classifier_metadata.json` do not configure `calibration_temperature` or an `evaluation_report`. As a consequence, `models.py` evaluates with $T = 1.0$ and sets `confidence_calibrated = False`.

**Decision per Specification:** Until the calibration provenance is unified across retraining splits, the system must strictly report all scores as **"Uncalibrated Model Score"** and never present scores as calibrated posterior probabilities or diagnostic certainties.

---

## 7. Prediction-State Issues Identified

| Issue ID | Description | Severity | Root Cause |
| :--- | :--- | :--- | :--- |
| **ISS-01** | Low-confidence healthy prediction (e.g., 28%) displayed with green card styling alongside "No regions met candidate threshold", misleading users that specimen is confirmed healthy. | **Critical** | Frontend rendered `overall_prediction` using the default positive green card style regardless of whether `needs_review` was true, and placed screening summary above classification without context. |
| **ISS-02** | Absence of backend-defined state enumeration. Backend only provided boolean `needs_review`, forcing frontend to guess how to present uncertainty. | **High** | Lack of formal `prediction_state` (`HIGH_CONFIDENCE` vs `LOW_CONFIDENCE_UNCERTAIN`) and `screening_state` in API schemas. |
| **ISS-03** | Conflation of Whole-Leaf Classification and Candidate-Region Screening into a single undifferentiated card sequence. | **High** | Single vertical container in `App.jsx` rendered whole-leaf and patch-screening results sequentially without distinct sectional boundaries. |
| **ISS-04** | Raw class identifiers displayed directly to users (e.g., `Cherry_(including_sour)___healthy`). | **Medium** | Missing user-facing display label transformation in API responses and frontend views. |
| **ISS-05** | Labeling of scores as generic percentage or ambiguous "uncalibrated score". | **Medium** | Missing mandatory explicit label `"Uncalibrated Model Score"` on whole-leaf prediction cards. |

---

## 8. Exact Files Requiring Modification

1. `backend/app/schemas/analysis.py`:
   - Define formal response schemas for `PredictionResult` and `ScreeningResult`.
   - Add fields: `prediction_state`, `prediction_status`, `state_tone`, `is_uncertain`, `display_label`, `crop_name`, `condition_name`, `score_label`, `guidance_text`, `screening_state`, `screening_status`, `screening_tone`, `screening_message`, `safety_disclaimer`.
2. `backend/app/services/models.py`:
   - Add label formatting utility converting raw PlantVillage class strings into clean, user-friendly labels while preserving raw identifiers.
   - Populate `prediction_state` (`HIGH_CONFIDENCE` vs `LOW_CONFIDENCE_UNCERTAIN`), `display_label`, and explicit uncertainty guidance in `predict_class`.
3. `backend/app/services/screening.py`:
   - Incorporate `screening_state` (`NO_CANDIDATES_IDENTIFIED` vs `CANDIDATES_IDENTIFIED`), state titles, summary explanations, and safety disclaimers into `screen_plant_image`.
   - Attach display labels and score labels to all candidate regions.
4. `backend/app/main.py`:
   - Ensure `/predict` and `/screen-plant` endpoints emit the enriched, structured state objects.
5. `frontend/src/App.jsx`:
   - Restructure result presentation into two clearly segregated sections:
     - **Section A: Whole-Leaf Classification**
     - **Section B: Candidate-Region Screening**
   - Implement dynamic state badges (Green for High-Confidence Prediction, Amber for Uncertain Prediction, Neutral for No Candidates).
   - Render readable display labels alongside raw class names.
   - Display explicit "Uncalibrated Model Score" labels.
   - Ensure low-confidence healthy predictions clearly state unreliability and do not appear as confirmed healthy.
6. `frontend/src/styles.css`:
   - Add styling for Section A and Section B headers, state-specific badges (`badge-good`, `badge-amber`, `badge-neutral`), amber result cards for uncertain predictions, and clear boundary containers.
7. `backend/tests/`:
   - Add automated test coverage validating backend prediction-state logic and frontend API response structures across all required scenarios.

