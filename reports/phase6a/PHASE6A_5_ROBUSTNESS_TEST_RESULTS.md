# Phase 6A.5: Image Robustness Test Results

**Research Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Date:** November 14, 2026  
**Evaluation Platform:** Windows 11 CPU-based System (AMD Ryzen 7 5800HS, 8 Cores / 16 Threads)  
**Active Checkpoint:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`  
**Checkpoint SHA-256:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`  
**Base Test Specimen:** `data/processed/PlantVillage/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG`  
**Baseline Performance:** Class: `Apple___Apple_scab` | Score: $95.56\%$ | State: `HIGH_CONFIDENCE` | Candidates: $8$

---

## 1. Methodology and Scope

To evaluate model resilience under realistic image capture degradation, controlled synthetic transformations were applied to a verified disease specimen (`Apple___Apple_scab`). 

> [!IMPORTANT]
> **Experimental Disclaimer:** All transformed images evaluated in this section are algorithmically derived synthetic test inputs. These tests evaluate numerical and perceptual stability under controlled perturbations; they do **not** constitute independent real-world agricultural field validation. Model preprocessing was strictly preserved without parameter adjustments to artificially enhance robustness.

---

## 2. Robustness Test Summary Matrix

| Perturbation Category | Transformation Parameters | Predicted Class | Uncalibrated Score | Score Shift ($\Delta$) | Class Preserved? | Prediction State | Screening Candidates |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Specimen** | None (Original Image, $256 \times 256$) | `Apple___Apple_scab` | **$95.56\%$** | **$0.0\%$** | **Yes** | `HIGH_CONFIDENCE` | **8** |
| **Brightness Reduction** | Brightness factor $0.75$ ($-25\%$) | `Apple___Apple_scab` | $93.36\%$ | $-2.20\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Severe Darkening** | Brightness factor $0.50$ ($-50\%$) | `Apple___Apple_scab` | $87.49\%$ | $-8.07\%$ | Yes | `HIGH_CONFIDENCE` | 5 |
| **Brightness Increase** | Brightness factor $1.25$ ($+25\%$) | `Apple___Apple_scab` | $87.16\%$ | $-8.40\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Severe Overexposure** | Brightness factor $1.50$ ($+50\%$) | **`Apple___healthy`** | **$63.79\%$** | **$-31.77\%$** | **NO (Flipped)** | `HIGH_CONFIDENCE` | **4** |
| **Low Contrast** | Contrast factor $0.50$ ($50\%$ reduction) | `Apple___Apple_scab` | $94.75\%$ | $-0.81\%$ | Yes | `HIGH_CONFIDENCE` | 7 |
| **High Contrast** | Contrast factor $1.50$ ($50\%$ increase) | `Apple___Apple_scab` | $94.88\%$ | $-0.68\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Mild Gaussian Blur** | Radius $\sigma = 1.0\text{ px}$ | `Apple___Apple_scab` | $97.94\%$ | $+2.38\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Moderate Blur** | Radius $\sigma = 2.0\text{ px}$ | `Apple___Apple_scab` | $64.88\%$ | $-30.68\%$ | Yes | `HIGH_CONFIDENCE` | 4 |
| **Slight Rotation Tilt** | $+15^\circ$ planar rotation | `Apple___Apple_scab` | $49.77\%$ | $-45.79\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 7 |
| **Right-Angle Rotation** | $90^\circ$ clockwise rotation | `Apple___Apple_scab` | $43.95\%$ | $-51.61\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 3 |
| **Inverted Rotation** | $180^\circ$ inverted rotation | `Apple___Apple_scab` | $84.36\%$ | $-11.20\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Counter-Clockwise** | $270^\circ$ rotation | `Apple___Apple_scab` | $33.86\%$ | $-61.70\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 2 |
| **Severe Downscaling** | Downscaled to $160 \times 160\text{ px}$ | `Apple___Apple_scab` | $97.35\%$ | $+1.79\%$ | Yes | `HIGH_CONFIDENCE` | 2 |
| **Direct CNN Rescaling** | Rescaled to $224 \times 224\text{ px}$ | `Apple___Apple_scab` | $93.99\%$ | $-1.57\%$ | Yes | `HIGH_CONFIDENCE` | 4 |
| **Upscaling** | Bicubic upscaled to $600 \times 600\text{ px}$ | `Apple___Apple_scab` | $95.77\%$ | $+0.21\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Vertical Stretch** | Aspect ratio $1:2$ ($200 \times 400\text{ px}$) | `Apple___Apple_scab` | $96.79\%$ | $+1.23\%$ | Yes | `HIGH_CONFIDENCE` | 6 |
| **Horizontal Stretch** | Aspect ratio $2:1$ ($400 \times 200\text{ px}$) | `Apple___Apple_scab` | $97.23\%$ | $+1.67\%$ | Yes | `HIGH_CONFIDENCE` | **0** |
| **JPEG Compression** | Quality factor $Q = 30$ | `Apple___Apple_scab` | $94.02\%$ | $-1.54\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **Heavy JPEG Artifacts**| Quality factor $Q = 15$ | `Apple___Apple_scab` | $85.74\%$ | $-9.82\%$ | Yes | `HIGH_CONFIDENCE` | 2 |
| **Additive Sensor Noise**| Gaussian noise ($\sigma = 15$) | `Apple___Apple_scab` | $98.44\%$ | $+2.88\%$ | Yes | `HIGH_CONFIDENCE` | 3 |

---

## 3. Analysis of Vulnerabilities and Failure Modes

### 3.1 Failure Mode 1: Severe Overexposure Causes Healthy Misclassification
* **Observed Phenomenon:** When brightness was increased by $+50\%$, the predicted class flipped from `Apple___Apple_scab` to `Apple___healthy` with $63.79\%$ confidence.
* **Mechanism:** MobileNetV3-Small’s inverted bottleneck features rely heavily on edge gradient contrasts and localized dark pigmentation to identify fungal scab lesions. Extreme overexposure saturates pixels toward the upper clipping boundary ($255$), washing out dark lesion textures and rendering the leaf surface uniformly light green.
* **Operational Implication:** Strong direct sunlight or flash reflection is a primary failure mode that can generate false-negative health predictions.

### 3.2 Failure Mode 2: In-Plane Rotational Sensitivity
* **Observed Phenomenon:**
  * While the argmax class remained `Apple___Apple_scab` across all rotations ($15^\circ, 90^\circ, 180^\circ, 270^\circ$), the uncalibrated confidence score plummeted from $95.56\%$ down to $49.77\%$ at $15^\circ$, $43.95\%$ at $90^\circ$, and $33.86\%$ at $270^\circ$.
  * For $15^\circ, 90^\circ,$ and $270^\circ$, confidence fell below the $0.60$ operational threshold, triggering `LOW_CONFIDENCE_UNCERTAIN` and flagging the specimen for human review.
  * Inverted rotation ($180^\circ$) remained relatively resilient ($84.36\%$).
* **Mechanism:** PlantVillage training imagery is predominantly oriented vertically with petioles at the bottom. The model learned slight orientation-dependent priors in its convolutional feature hierarchy.
* **Operational Implication:** The review threshold mechanism successfully caught these orientation drops, demonstrating appropriate uncertainty escalation.

### 3.3 Failure Mode 3: Blur-Induced Attenuation
* **Observed Phenomenon:** Mild blur ($\sigma = 1.0$) was well-tolerated ($97.94\%$), but moderate blur ($\sigma = 2.0$) depressed confidence by $-30.68\%$ ($64.88\%$) and pruned candidate regions from 8 down to 4.
* **Mechanism:** Gaussian smoothing destroys high-frequency textural cues characteristic of foliar fungal mycelia and necrotic spotting.
* **Operational Implication:** Validates the inclusion of OpenCV Laplacian blur scoring in the application ingestion pipeline (threshold $25.0$) to warn users before analysis.

### 3.4 Failure Mode 4: Aspect Ratio Distortion Prunes Screening Proposals
* **Observed Phenomenon:** Horizontal canvas stretching ($2:1$, $400 \times 200$) maintained whole-leaf classification ($97.23\%$), but candidate regions fell to $0$.
* **Mechanism:** Whole-leaf inference resizes the entire image to $224 \times 224$ prior to classification, whereas candidate-region proposal extracts rectangular sliding patches. Non-square patches extracted from stretched canvases fail scale checks and ExG foliar saliency thresholds.
* **Operational Implication:** Users should capture specimens with standard near-square or $4:3$ frame geometries.

---

## 4. Practical Capture Recommendations for Field Deployments

Based on empirical perturbation findings, operational guidelines for users and extension agents should mandate:
1. **Avoid Saturated Lighting:** Do not photograph leaves in harsh direct overhead midday sunlight; capture under diffuse shade or overcast skies to prevent lesion contrast washout.
2. **Upright Leaf Orientation:** Align the leaf blade approximately vertically (tip pointing upward or downward) rather than sideways.
3. **Sharp Focus:** Discard blurry images before submission; use tap-to-focus on visible spots.
4. **Natural Aspect Ratio:** Maintain the native camera aspect ratio ($4:3$ or $1:1$); avoid post-capture digital distortion.

