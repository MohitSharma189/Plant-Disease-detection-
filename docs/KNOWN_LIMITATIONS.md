# Plant Disease Screening System: Known Limitations and Validation Gaps

**Research Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Documentation Version:** Phase 6A Release Specification  

---

## 1. Overview

This document provides a transparent, scientifically honest account of the algorithmic boundaries, empirical limitations, and validation gaps inherent in the current application. Users, researchers, and developers must understand these constraints before interpreting outputs or considering deployment.

---

## 2. Dataset and Domain Biases

### 2.1 Laboratory vs. Field Discrepancy
* **PlantVillage In-Domain Bias:** The primary training and validation corpus (PlantVillage) consists of excised leaves photographed under controlled indoor lighting against uniform neutral or white backgrounds.
* **Lack of Environmental Complexity:** The training distribution lacks authentic field factors, such as:
  * Variable solar irradiance, direct sun glint, and dappled canopy shadows.
  * Soil textures, mulch, drip lines, and trellis supports.
  * Co-occurring nutritional deficiencies (nitrogen chlorosis, iron deficiency) and insect herbivory.
  * Dust, dew, pesticide spray residue, and wind motion blur.

### 2.2 Cross-Domain Generalization Gap (PlantDoc)
* As established in Phase 4 and Phase 5 empirical evaluations, classification accuracy drops substantially when the model is evaluated on in-the-wild field imagery (PlantDoc):
  * **PlantVillage Holdout Test Accuracy:** $\sim 95.02\%$
  * **PlantDoc External Zero-Shot Accuracy:** $\sim 38.98\%$ top-1 ($64.83\%$ top-3)
* The model cannot reliably distinguish complex field symptoms without substantial false-positive activations on background foliage.

---

## 3. Probability Calibration and Uncertainty Limitations

### 3.1 Uncalibrated Score Reporting
* Temperature scaling calibration remains **disabled** ($T = 1.0$), and scores are explicitly presented as `"Uncalibrated Model Score"`.
* In Phase 5 and Phase 6A.2 audits, temperature parameters ($T^* \approx 0.9902$) fitted on isolated validation splits failed to maintain calibration on out-of-distribution and cross-domain specimens.
* The reported percentage scores reflect **relative neural network logit ranking**, not true frequentist posterior probabilities. A score of $95\%$ does not mean there is a $95\%$ probability that the leaf has the disease in a real field setting.

### 3.2 Operational Review Threshold Provenance
* The $0.60$ review threshold is an **operational heuristic**, not an analytically proven Bayesian boundary. While it effectively caught low-confidence ambiguous specimens in validation testing, it does not guarantee error-free triage.

---

## 4. Screening Pipeline & Spatial Localization Constraints

### 4.1 Coarse Bounding Proposals vs. Lesion Segmentation
* The candidate-region screening pipeline generates **rectangular sliding-window image proposals** ($224 \times 224$ scaled).
* These bounding boxes demarcate overlapping crops that met the foliar screening criteria; they are **not pixel-level segmentation masks** and do not compute exact lesion area percentages or biological damage scores.

### 4.2 Precision and Recall Trade-Offs
* As measured in Phase 4.1 benchmark evaluations, the multi-scale sliding grid achieves moderate recall on conspicuous foliar lesions, but suffers from low precision ($\sim 10.6\%$ at $\text{IoU} \ge 0.25$).
* Approximately 8 to 9 out of every 10 proposed candidate crops in cluttered field foliage are spurious background activations that satisfied the foliar color filter.

---

## 5. Capture Sensitivity & Failure Modes

Empirical perturbation testing in Phase 6A.5 revealed specific capture-dependent failure modes:

1. **Severe Overexposure (+50% Brightness):**
   * High lighting saturation clips RGB values to 255, obliterating dark fungal mycelial contrast. In testing, this flipped an `Apple___Apple_scab` specimen into a false `Apple___healthy` prediction ($63.79\%$).
2. **In-Plane Rotational Sensitivity:**
   * Rotations of $15^\circ, 90^\circ,$ and $270^\circ$ dropped confidence scores by $45\% - 61\%$, falling below the $0.60$ threshold. The training distribution's upright orientation prior makes the network sensitive to tilted leaf blades.
3. **Focal Blur Degradation:**
   * Gaussian blur ($\sigma \ge 2.0$) destroys high-frequency lesion textures, severely degrading both classification score and localized candidate proposals.
4. **Canvas Geometry Distortions:**
   * Non-uniform horizontal stretching ($2:1$) eliminated candidate region proposals entirely.

---

## 6. Hardware & Concurrency Boundaries

* **Single-User Interactive Design:** The application is optimized for single-client local CPU execution on extension laptops or field workstations.
* **Throughput Limits:** Concurrent testing demonstrated stable operation up to 10 simultaneous clients ($\sim 889.7\text{ ms}$ turnaround). Under high multi-user web traffic, requests queue synchronously on CPU threads; deployment at scale would require asynchronous worker pools (e.g. Celery / Redis) and dedicated inference workers.

---

## 7. Absence of Clinical Regulatory Certification

* This software has **not been evaluated or approved by any agricultural regulatory board, plant diagnostic laboratory, or government department of agriculture**.
* It must not be marketed or deployed as an autonomous diagnostic device, crop insurance verification tool, or certified pesticide application prescriber.

