# Verdant Field Laboratory: User Guide

**Application:** Plant Disease Screening System with MobileNetV3-Small  
**Version:** 0.1.0 (Phase 6A Release)  
**Primary Audience:** Agricultural extension agents, researchers, agronomists, and growers.

---

## 1. Introduction

The Verdant Field Laboratory is an interactive screening application designed to assist in preliminary foliar symptom assessment. It combines:
1. **Whole-Leaf Classification:** An evaluation of the overall leaf specimen against 38 plant and disease categories.
2. **Candidate-Region Screening:** An automated multi-scale sliding window scan that isolates foliar patches displaying conspicuous disease symptoms above an operational threshold.

> [!CAUTION]
> **Advisory Scope:** This application is an **exploratory screening aid**, not a clinical plant diagnostic laboratory or legal phyto-sanitary authority. It generates algorithmic hypotheses based on image similarity to trained datasets.

---

## 2. Preparing and Uploading an Image Specimen

### 2.1 Specimen Capture Recommendations
For reliable algorithmic screening, follow these capture standards:
* **Diffuse Lighting:** Capture leaves under diffuse daylight or overcast conditions. **Avoid harsh midday direct sun, camera flash, or heavy shadows**, which can wash out dark fungal lesions and produce false healthy predictions.
* **Vertical Orientation:** Orient the leaf naturally with the blade roughly upright (petiole toward the bottom). Non-standard rotations can depress confidence scores.
* **Sharp Focus:** Ensure the camera is sharply focused on foliar textures. Blurry images reduce feature contrast and trigger quality warnings.
* **Aspect Ratio:** Keep the natural $4:3$ or $1:1$ frame aspect ratio. Do not stretch or compress the image digitally.

### 2.2 Upload Methods
The application supports three input mechanisms:
1. **Drag-and-Drop:** Drag any JPEG, PNG, or WEBP image onto the dropzone canvas.
2. **File Browser:** Click the dropzone to open your device's file picker.
3. **Webcam Capture:** Click **"Use webcam to capture"** to stream from an attached USB or laptop camera, then press **"Capture photo"**.

### 2.3 Upload Constraints
* **Formats Supported:** JPEG (`.jpg`, `.jpeg`), PNG (`.png`), WEBP (`.webp`).
* **Maximum File Size:** $10\text{ MB}$ (files exceeding this are immediately blocked).
* **Minimum Dimensions:** $128 \times 128\text{ pixels}$ (files below this size fail quality verification).

---

## 3. Selecting the Screening Mode

Before screening, select the appropriate mode:

### A. Single Leaf Mode (`Single leaf`)
* **When to Use:** When your photograph is a close-up focused on an isolated individual leaf blade against a neutral background.
* **Pipeline Behavior:** 
  * Executes full-image classification (Section A).
  * Automatically screens 9 overlapping foliar crops ($52\%$ scale fraction, $62\%$ stride) to detect localized symptomatic lesions (Section B).
  * Generates a Grad-CAM model saliency attribution map.

### B. Whole Plant Mode (`Whole plant`)
* **When to Use:** When your photograph depicts an entire plant canopy, multiple foliage branches, or leaves against soil/trellis backgrounds.
* **Pipeline Behavior:**
  * Bypasses whole-leaf classification to prevent applying single-leaf classification assumptions to complex multi-leaf canopies.
  * Evaluates a sliding canopy grid (12–24 crops depending on aspect ratio).
  * Reports all localized candidate patches that exceed the $0.60$ threshold.

---

## 4. Interpreting Results

The results view is partitioned into two clearly separated sections:

### Section A: Whole-Leaf Classification

1. **Prediction Status Pill:**
   * **`Model Prediction` (Green):** Confidence meets or exceeds the $0.60$ operational review threshold.
   * **`Uncertain Prediction` (Amber):** Confidence is below $0.60$. Professional human review is strongly recommended.
2. **Predicted Class & Dataset Label:**
   * Displays the human-readable plant and condition (e.g., `Apple - Scab`) alongside the internal classification key (`Apple___Apple_scab`).
3. **Uncalibrated Model Score:**
   * Displays the raw softmax probability percentage (e.g., `96%`).
   * Explicitly labeled as `"Uncalibrated Model Score"`, indicating raw neural network output.
4. **Agronomic Guidance & Recommendations:**
   * Practical, conservative actions and suggestions for confirmation before applying chemical or biological interventions.
5. **Special Safety Alert for Healthy Predictions:**
   * If the model predicts a `healthy` class with score $< 60\%$, an amber caution banner appears:
     > *"Caution on healthy prediction: Because this confidence score is below the operational review threshold (60%), this specimen must never be assumed healthy or disease-free. A low-confidence healthy prediction is an inconclusive tentative match. Professional diagnostic inspection is strongly recommended."*

---

### Section B: Candidate-Region Screening

1. **Screening Metrics Bar:**
   * **Screening Mode:** Single Leaf vs Whole Plant.
   * **Windows Examined:** Number of crops analyzed (typically 9 or 24).
   * **Candidates Above Threshold:** Number of localized crops meeting the $0.60$ threshold.
   * **Candidate Threshold:** Fixed at $60\%$.
2. **Screening Status Card & Safety Notice:**
   * Summarizes candidate detections.
   * When **0 candidates** are identified, the notice explicitly states:
     > *"Important Safety Notice: Lack of candidate screening regions does not establish that the leaf or plant is healthy or disease-free. Candidate screening identifies conspicuous foliar patches matching trained disease categories above the threshold and does not detect subtle, systemic, or unindexed conditions."*
3. **Annotated Image Overlay:**
   * Original specimen displayed with numbered red rectangular bounding boxes marking detected candidate crops.
4. **Grad-CAM Saliency Attribution:**
   * Heatmap showing which pixel clusters most strongly influenced the neural network's top prediction.
   * *Clarification:* Highlighted pixels denote **feature importance**, not verified biological lesion perimeters.
5. **Candidate Region Cards:**
   * Each candidate card displays the cropped image patch, predicted class, uncalibrated score %, observed visible evidence, severity rating, practical field actions, biological/chemical treatment considerations, and institutional extension citations.

---

## 5. What the System CANNOT Establish

Users must never rely on the application for the following determinations:
1. **Proof of Health:** Low confidence healthy matches and zero-candidate screening results do **not** confirm that a crop is healthy or pathogen-free.
2. **Out-of-Distribution Diagnosis:** Unknown weed leaves or unindexed crop diseases are mapped to the nearest trained category.
3. **Chemical Prescriptions:** Suggested management actions are general institutional guidelines, not certified agronomic prescriptions. Consult a local university extension specialist or certified crop advisor before purchasing or applying treatments.
