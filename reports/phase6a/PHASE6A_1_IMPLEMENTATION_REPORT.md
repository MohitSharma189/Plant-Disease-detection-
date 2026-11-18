# Phase 6A.1: Implementation Report — Result Presentation & Prediction-State Improvement

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Phase:** 6A.1  
**Date:** November 14, 2026  
**Status:** Completed and Verified  

---

## 1. Executive Overview

Phase 6A.1 resolves a critical usability and diagnostic communication problem in the Plant Disease Screening System. In previous builds, low-confidence predictions (such as a 28% uncalibrated score for `Cherry_(including_sour)___healthy`) were presented alongside "No regions met the candidate threshold" in a green-bordered card, creating a false impression that the model had certified the leaf as healthy.

To prevent this misinterpretation and ensure rigorous scientific communication, we implemented:
1. **Backend-Governed Prediction States:** Fully enumerated states (`HIGH_CONFIDENCE` vs `LOW_CONFIDENCE_UNCERTAIN`) determined solely by the backend using the verified 0.60 review threshold.
2. **Prominent Uncertainty State:** Any classification score $< 0.60$ is explicitly designated as an **"Uncertain Prediction"** with an amber indicator, a tentative label prefix (`Tentative match:`), and explicit guidance that low-confidence results must never be treated as confirmed health or diagnosis.
3. **Formal Screening States:** Candidate-region screening emits formal states (`NO_CANDIDATES_IDENTIFIED` vs `CANDIDATES_IDENTIFIED`) accompanied by a mandatory safety notice explaining that absence of candidate regions does not prove the specimen is disease-free.
4. **Clean Information Hierarchy:** Complete visual and structural separation of **Section A: Whole-Leaf Classification** and **Section B: Candidate-Region Screening**.
5. **Human-Readable Plant & Condition Nomenclature:** Automatic transformation of raw internal identifiers (e.g., `Cherry_(including_sour)___healthy`, `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`) into clear, professional labels (`Cherry (Sour) - Healthy`, `Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot`), while preserving raw identifiers for reproducibility.
6. **Explicit Score Labeling:** All scores are unambiguously labeled as **"Uncalibrated Model Score"**.

---

## 2. Files Modified

| File Path | Type | Key Changes |
| :--- | :--- | :--- |
| `backend/app/schemas/analysis.py` | Backend | Added formal Pydantic schemas: `PredictionResult`, `CandidateRegion`, `ScreeningResult`, `PredictResponse`, `ScreenPlantResponse` defining enumerated states, display labels, and disclaimers. |
| `backend/app/services/models.py` | Backend | Added `format_class_metadata` helper covering all 38 classes; updated `predict_class` to evaluate the 0.60 review threshold and return `prediction_state`, `prediction_status`, `state_tone`, `headline`, `explanation`, `recommendation`, `display_label`, `crop_name`, `condition_name`, and `score_label`. |
| `backend/app/services/screening.py` | Backend | Updated `Candidate` dataclass with `display_label`, `crop_name`, `condition_name`, and `score_label`; enriched `screen_plant_image` to evaluate `screening_state` (`NO_CANDIDATES_IDENTIFIED` vs `CANDIDATES_IDENTIFIED`), `screening_status`, `screening_tone`, and explicit safety disclaimers. |
| `backend/tests/test_prediction_states.py` | Testing | New comprehensive unit test suite covering all 10 required validation scenarios (high/low confidence healthy/diseased, no/multiple candidates, invalid/unsupported uploads, backend errors, multi-aspect dimensions). |
| `frontend/src/App.jsx` | Frontend | Added client-side fallback `formatDisplayLabel`; structured results into **Section A (Whole-Leaf Classification)** and **Section B (Candidate-Region Screening)**; implemented dynamic state pills (Green vs Amber vs Neutral); added explicit healthy prediction caution alert; updated printable report. |
| `frontend/src/styles.css` | Styling | Added CSS classes: `.results-section`, `.section-classification`, `.section-screening`, `.status-pill` (`.pill-good`, `.pill-amber`, `.pill-neutral`, `.pill-info`), `.classification-card` (`.card-confident`, `.card-uncertain`), `.healthy-warning-alert`, `.screening-metrics-bar`, and `.screening-status-card`. |

---

## 3. Backend Prediction-State Logic

The backend strictly determines the operational prediction state; the frontend merely displays the state, tone, and guidance returned by the backend.

### 3.1 Whole-Leaf Classification States
```python
review_threshold = 0.60  # Default verified threshold in models.py
confidence = probabilities[index]
needs_review = confidence < review_threshold

if needs_review:
    prediction_state = "LOW_CONFIDENCE_UNCERTAIN"
    prediction_status = "Uncertain Prediction"
    state_tone = "amber"
    headline = f"Uncertain Prediction · Tentative match: {display_label}"
    explanation = (
        f"The uncalibrated model score ({confidence:.1%}) is below the operational review threshold ({review_threshold:.0%}). "
        "The model could not produce a sufficiently reliable classification for this specimen. "
        "Tentative predictions must not be treated as a confirmed diagnosis or confirmation of plant health."
    )
    recommendation = (
        "Treat this result as tentative and seek confirmation from a qualified plant diagnostic clinic "
        "or local agricultural extension service. Consider retaking a sharper, well-lit close-up."
    )
else:
    prediction_state = "HIGH_CONFIDENCE"
    prediction_status = "Model Prediction"
    state_tone = "good"
    headline = f"Model Prediction · {display_label}"
    explanation = (
        f"The uncalibrated model score ({confidence:.1%}) meets the operational review threshold ({review_threshold:.0%}). "
        "This output represents an automated machine-learning screening prediction, not a certified clinical or laboratory diagnosis."
    )
    recommendation = (
        "Qualified agronomic verification is recommended before taking field-management or chemical actions."
    )
```

### 3.2 Candidate-Region Screening States
```python
candidate_score_threshold = 0.60  # MIN_CANDIDATE_SCORE in screening.py

if len(deduplicated) == 0:
    screening_state = "NO_CANDIDATES_IDENTIFIED"
    screening_status = "No Candidate Regions Identified"
    screening_tone = "neutral"
    screening_message = f"No candidate regions crossed the screening threshold ({candidate_score_threshold:.0%}) across {checked} examined image crops."
    safety_disclaimer = (
        "Important Safety Notice: Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free. "
        "Candidate screening only detects conspicuous foliar patches matching trained disease categories above the threshold."
    )
else:
    screening_state = "CANDIDATES_IDENTIFIED"
    screening_status = f"{len(deduplicated)} Candidate Region(s) Identified"
    screening_tone = "info"
    screening_message = f"{len(deduplicated)} candidate region(s) met the foliar screening threshold ({candidate_score_threshold:.0%}) across {checked} examined crops."
    safety_disclaimer = "Bounding boxes represent exploratory candidate crops requiring inspection, not confirmed lesion boundaries or clinical diagnoses."
```

---

## 4. Frontend Result Presentation Improvements

### 4.1 Two Distinct Sections
1. **Section A: Whole-Leaf Classification:**
   - Dedicated header row with `Section A` tag and `Whole-Leaf Classification` title.
   - Status pill in the upper-right corner:
     - High confidence: Green pill with `Model Prediction`.
     - Low confidence: Amber pill with `Uncertain Prediction`.
   - Card appearance:
     - Confident: Neutral-green card (`#edf2e6` background, `#b8cba9` border).
     - Uncertain: Warm amber card (`#fff8e8` background, `#e3caa0` border).
   - Display items:
     - Prominent human-readable title (e.g. `Cherry (Sour) - Healthy`).
     - Dataset label code in muted font (`Cherry_(including_sour)___healthy`).
     - Large percentage score with explicit `Uncalibrated Model Score` label.
     - Guidance box providing context on unreliability and extension clinic referrals.
     - **Specific Caution on Low-Confidence Healthy Predictions:** An alert banner explicitly warning that low confidence healthy scores must never be used to confirm plant health.
2. **Section B: Candidate-Region Screening:**
   - Dedicated header row with `Section B` tag and `Candidate-Region Screening` title.
   - Status pill: `No Candidate Regions Identified` (Neutral) or `X Candidate Region(s) Identified` (Info).
   - Metrics grid: Displays Screening Mode, Windows Examined, Candidates Above Threshold, and Threshold (60%).
   - Safety Callout: Prominently states that no candidates does not mean the plant is healthy.
   - Overlay visualization and candidate crop cards.

---

## 5. Threshold Verification & Provenance

- **Review Threshold:** Verified at `0.60` ($60.0\%$) in `backend/app/services/models.py`.
- **Screening Candidate Threshold:** Verified at `0.60` ($60.0\%$) in `backend/app/services/screening.py`.
- **Consistency:** Classification and candidate screening thresholds are unified at 0.60.
- **Calibration Status:** Production `classifier_metadata.json` does not configure temperature scaling or an evaluation report. Consequently, all scores are reported as **"Uncalibrated Model Score"** with $T = 1.0$.

---

## 6. Unresolved Technical Issues & Ongoing Work

1. **Temperature Calibration Discrepancy:**
   - Phase 3 converged to $T^* = 0.5538$ (L-BFGS on `best_model.pth`).
   - Phase 5 converged to $T^* = 0.9902$ across the comparative architecture benchmark.
   - *Resolution for Phase 6A.1:* The deployed model strictly serves uncalibrated scores and identifies them transparently. Resolving the optimization difference across split manifests remains documented for future retuning cycles.
2. **Segmentation Capability:**
   - Semantic segmentation checkpoint (`saved_models/segmentation.pt`) is absent by design; the system correctly reports bounding boxes as exploratory candidate crops and Grad-CAM as foliar attribution, never claiming lesion mask segmentation.

