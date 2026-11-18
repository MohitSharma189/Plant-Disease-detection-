# Phase 6A.5: Comprehensive Application Validation Report

**Research Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Date:** November 14, 2026  
**Environment:** Windows 11 CPU-based System (AMD Ryzen 7 5800HS with Radeon Graphics, 8 Cores / 16 Threads, 16 GB RAM)  
**Active Checkpoint:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`  
**Checkpoint SHA-256:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`  
**Status:** Complete – Checkpoint Verified, Multi-specimen Inference Audited, Screening Evaluated, Reliability & Concurrency Validated, Performance Benchmarked

---

## 1. Executive Summary

Phase 6A.5 provides a comprehensive, empirically rigorous validation of the integrated Plant Disease Screening System. Building upon the verified deployment of the **Phase 5 MobileNetV3-Small TorchScript model** (`6fff52f1...`) and the full-stack ASGI integration established in Phases 6A.1–6A.4, this phase assesses system reliability, classification behavior across representative specimens, screening candidate proposal dynamics, error handling resilience, multi-client concurrency, and execution latencies.

All tests were performed using actual model inference and real image files without parameter forcing, artificial score padding, or mock overrides.

---

## 2. Checkpoint and Configuration Integrity

Prior to executing operational tests, the deployed model binary, preprocessing transforms, and runtime settings were audited against the Phase 5 benchmark specification:

| Parameter | Configured Value | Benchmark Specification | Verification Status |
| :--- | :--- | :--- | :---: |
| **Active Checkpoint Path** | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | `.../classifier.pt` | **MATCH** |
| **SHA-256 Digest** | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` | `6fff52f1...71213` | **VERIFIED (Bit-for-bit)** |
| **Network Architecture** | MobileNetV3-Small (TorchScript container) | MobileNetV3-Small | **MATCH** |
| **Output Classes** | 38 classes (`classifier_metadata.json`) | 38 classes | **MATCH** |
| **Input Dimensions** | $224 \times 224 \times 3$ (RGB) | $224 \times 224 \times 3$ | **MATCH** |
| **Normalization** | ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) | ImageNet standard | **MATCH** |
| **Softmax Temperature** | $T = 1.0$ (Uncalibrated raw logits) | $T = 1.0$ | **MATCH** |
| **Operational Review Threshold** | $0.60$ ($60\%$ score threshold) | $0.60$ | **MATCH** |
| **Screening Candidate Threshold** | $0.60$ (`MIN_CANDIDATE_SCORE`) | $0.60$ | **MATCH** |
| **Calibration Flag** | `settings.calibration_enabled = False` | Disabled | **MATCH** |

No configuration drift, accidental weight modifications, or class mapping misalignments have occurred.

---

## 3. Classification Validation on Representative Specimens

A diverse suite of 12 real specimens—spanning healthy controls, varied disease etiologies (fungal, bacterial, oomycete), out-of-distribution foliage, and complex in-the-wild field photography—was evaluated through `/predict`:

| Specimen Identifier | Ground-Truth Category | Model Predicted Class | Uncalibrated Score | Prediction State | Human Review |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `0055dd26...RS_HL 5672.JPG` | `Apple___healthy` | `Apple___healthy` | $99.0\%$ | `HIGH_CONFIDENCE` | No |
| `00907d8b...RS_HL 5709.JPG` | `Tomato___healthy` | `Tomato___healthy` | $98.7\%$ | `HIGH_CONFIDENCE` | No |
| `Grape_healthy_01.JPG` | `Grape___healthy` | `Grape___healthy` | $100.0\%$ | `HIGH_CONFIDENCE` | No |
| `00075aa8...FREC_Scab 3335.JPG` | `Apple___Apple_scab` | `Apple___Apple_scab` | $95.6\%$ | `HIGH_CONFIDENCE` | No |
| `Grape_Black_rot_01.JPG` | `Grape___Black_rot` | `Grape___Black_rot` | $96.7\%$ | `HIGH_CONFIDENCE` | No |
| `Potato_Late_blight_01.JPG` | `Potato___Late_blight` | `Potato___Late_blight` | $76.5\%$ | `HIGH_CONFIDENCE` | No |
| `Corn_NLB_01.JPG` | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | $78.7\%$ | `HIGH_CONFIDENCE` | No |
| `Tomato_Early_blight_01.JPG` | `Tomato___Early_blight` | `Tomato___Early_blight` | $60.7\%$ | `HIGH_CONFIDENCE` | No |
| `Squash_Powdery_mildew_01.JPG` | `Squash___Powdery_mildew` | `Squash___Powdery_mildew` | $95.9\%$ | `HIGH_CONFIDENCE` | No |
| `media_1794656629400_32844b42.jpg` | Unknown / Ambient Leaf | `Corn_(maize)___healthy` | $26.6\%$ | `LOW_CONFIDENCE_UNCERTAIN` | **Yes (Alert)** |
| `Early_20Blight_20-_20Potatoes.jpg` | PlantDoc In-the-Wild Potato | `Potato___Early_blight` | $49.4\%$ | `LOW_CONFIDENCE_UNCERTAIN` | **Yes (Review)** |
| `Powdery-Mildew-on-squash.jpg` | PlantDoc In-the-Wild Squash | `Squash___Powdery_mildew` | $66.9\%$ | `HIGH_CONFIDENCE` | No |

### Findings:
1. **In-Domain PlantVillage Generalization:** In-domain laboratory specimens across 6 distinct crops and 6 disease conditions consistently achieve confident, correct classifications ($60.7\% - 100.0\%$).
2. **Ambiguous Specimen Handling:** The ambient leaf specimen (`media_...32844b42.jpg`) produced a low score ($26.6\%$) assigned to `Corn_(maize)___healthy`. Because confidence is below the $0.60$ operational threshold, the backend tagged the result as `LOW_CONFIDENCE_UNCERTAIN`, successfully triggering the frontend healthy-warning alert rather than providing false reassurance.
3. **Out-of-Domain Attenuation:** Challenging field photography from PlantDoc with cluttered soil backgrounds produced significant score attenuation ($49.4\%$ on Potato Blight), correctly routing the prediction to human expert review.

---

## 4. Screening Validation (Single-Leaf vs. Whole-Plant)

Candidate-region screening was evaluated across single-leaf close-ups, multi-scale canopy scans, varied aspect ratios, and dense foliar scenes:

| Case Description | Specimen Source | Mode | Regions Checked | Candidates Identified | Top Candidate Class (Score) | Processing Latency |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| Clean Leaf Control | `Apple___healthy` | `single_leaf` | 9 | 0 | None (All healthy pruned) | $191.7\text{ ms}$ |
| Lesion-Bearing Leaf | `Apple___Apple_scab` | `single_leaf` | 9 | 8 | `Apple___Apple_scab` ($98.7\%$) | $149.4\text{ ms}$ |
| Canopy Clean Control | Clean Foliage | `whole_plant` | 9 | 0 | None | $100.3\text{ ms}$ |
| Complex Field Canopy | PlantDoc Potato Blight | `whole_plant` | 12 | 2 | `Potato___Late_blight` ($79.1\%$) | $187.1\text{ ms}$ |
| Dense Powdery Mildew | PlantDoc Squash Mildew | `whole_plant` | 24 | 7 | `Squash___Powdery_mildew` ($95.3\%$) | $280.4\text{ ms}$ |
| Square Canvas ($1:1$) | Synthetic ($256 \times 256$) | `whole_plant` | 9 | 0 | None | $90.4\text{ ms}$ |
| Landscape ($4:3$) | Synthetic ($800 \times 600$) | `whole_plant` | 24 | 0 | None | $252.4\text{ ms}$ |
| Portrait ($3:4$) | Synthetic ($600 \times 800$) | `whole_plant` | 24 | 0 | None | $235.2\text{ ms}$ |
| Widescreen ($16:9$) | Synthetic ($1280 \times 720$) | `whole_plant` | 24 | 0 | None | $252.3\text{ ms}$ |

### Screening Distinctions:
* **Exploratory Crop Windows, Not Lesion Masks:** Candidate bounding boxes demarcate overlapping image crops ($224 \times 224$ scaled) that satisfied the non-healthy threshold. They do not represent fine-grained pixel segmentation or biological lesion perimeters.
* **Aspect Ratio Adaptability:** Sliding window strides ($62\%$) and scale fractions dynamically adapt to canvas orientation ($1:1$, $4:3$, $3:4$, $16:9$), generating between 9 and 24 proposals without geometry distortion or indexing boundary faults.

---

## 5. API and Backend Reliability

The FastAPI backend was subjected to negative testing, fault injection, sequential soak tests, and concurrent multi-client requests:

### 5.1 Fault Injection & Boundary Verification
* **Empty Payload (0 bytes):** Handled with HTTP `422 Unprocessable Entity` (`"The uploaded file is empty."`).
* **Corrupted Image Bytes:** Handled with HTTP `422 Unprocessable Entity` (`"The uploaded file is not a valid image."`).
* **Sub-128px Image ($64 \times 64$):** Handled with HTTP `422 Unprocessable Entity` (`"Image dimensions must be at least 128x128 pixels."`).
* **Oversized Upload ($>10\text{ MB}$):** Handled with HTTP `413 Request Entity Too Large` (`"Image exceeds the 10 MB upload limit."`).
* **Non-Image Plain Text File:** Handled with HTTP `422 Unprocessable Entity`.
* **Invalid Screening Mode (`mode="diagonal_scan"`):** Handled with HTTP `422 Unprocessable Entity` (`"Mode must be 'single_leaf' or 'whole_plant'."`).
* **Model Unloaded Fault Simulation:** Handled with HTTP `503 Service Unavailable` (`"A trained classifier checkpoint and class mapping are required for image screening."`).

### 5.2 Endurance & Concurrency Assessment
* **Sequential Soak Test (25 requests):** $100\%$ success rate ($25/25$ HTTP 200 OK), steady heap memory footprint, no memory leaks or thread hanging.
* **Concurrent Thread Pool (10 simultaneous clients):** $100\%$ success rate ($10/10$ HTTP 200 OK). Mean turnaround was $889.7\text{ ms}$ with maximum latency of $1172.1\text{ ms}$, demonstrating smooth queueing across CPU threads without process termination or socket dropouts.

---

## 6. Frontend Presentation & Accessibility Audit

Static and runtime auditing of the React application in [frontend/src/App.jsx](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/frontend/src/App.jsx) and [frontend/src/styles.css](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/frontend/src/styles.css) verified:

1. **Decoupled Result Architecture:**
   * Section A (Whole-Leaf Classification) renders independently of Section B (Candidate-Region Screening).
   * In whole-plant mode, Section A displays an informative notice explaining that whole-leaf classification is disabled for whole-plant scenes.
2. **Safety Warnings & Alerts:**
   * Amber alert banner (`.healthy-warning-alert`) explicitly warns users whenever a healthy prediction is tentative ($< 0.60$).
   * Safety disclaimer (`.screening-safety-disclaimer`) appears prominently whenever 0 candidates are detected, explicitly refuting the false assumption that zero candidates equals confirmed plant health.
3. **Model Attention Clarification:**
   * Grad-CAM visualizations are captioned explicitly as model saliency / attention maps, clarifying that highlighted areas denote feature importance rather than verified biological lesions.
4. **Accessibility Compliance:**
   * Screen reader live updates supported via `aria-live="polite"` on results and loading feedback.
   * Semantic ARIA roles (`role="alert"`, `role="status"`).
   * Keyboard accessible dropzone (`onKeyDown` Enter/Space triggers).
   * Responsive CSS media queries verified for desktop ($>1050\text{px}$), tablet ($761-900\text{px}$), mobile ($\le 760\text{px}$, $\le 520\text{px}$), and `@media print`.

---

## 7. Performance Benchmarking Summary

Evaluated over 50 consecutive runs with 10 explicit warm-up runs on the AMD Ryzen 7 5800HS CPU (8 cores / 16 threads):

| Execution Layer / Target | Sample Size | Mean Latency ($\text{ms}$) | Median ($\text{ms}$) | $p95$ ($\text{ms}$) | $p99$ ($\text{ms}$) | Min / Max ($\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure TorchScript Model Forward** | 50 | **$8.43 \pm 1.16$** | **$8.18$** | **$10.77$** | **$11.41$** | $7.01$ / $11.54$ |
| **Whole-Leaf API (`/predict`)** | 50 | **$27.84 \pm 0.77$** | **$27.80$** | **$29.41$** | **$29.94$** | $26.68$ / $30.12$ |
| **Single-Leaf Screening (`/screen-plant`)** | 50 | **$186.24 \pm 11.52$** | **$187.07$** | **$197.99$** | **$217.20$** | $166.45$ / $233.43$ |
| **Whole-Plant Screening (`/screen-plant`)** | 50 | **$200.03 \pm 17.63$** | **$195.93$** | **$244.75$** | **$266.79$** | $182.70$ / $273.31$ |

The CPU latency profile confirms rapid turnaround ($< 250\text{ ms}$ for $95\%$ of all screening transactions), well within interactive field usage requirements without GPU acceleration.

---

## 8. Remaining Limitations and Validation Gaps

1. **Closed-Set Taxonomic Scope:** The classifier is bounded to the 38 PlantVillage classes. Out-of-domain weed foliage, physiological abiotic disorders, or unindexed pathogens are mapped into the nearest closed-set category.
2. **Orientation Sensitivity:** Perturbation tests demonstrated that while class predictions remain resilient, confidence scores drop below the review threshold ($0.60$) under non-standard rotational orientations ($15^\circ, 90^\circ, 270^\circ$).
3. **Severe Overexposure Failure Mode:** Extreme overexposure ($+50\%$ brightness) washes out fine foliar lesion contrast, resulting in false healthy predictions.
4. **Field vs. Lab Domain Gap:** Complex canopy clutter and direct sunlight in field photography (PlantDoc) produce lower confidence scores than laboratory backdrops, underscoring that candidate screening is an assistive triage tool rather than a clinical diagnostic instrument.

