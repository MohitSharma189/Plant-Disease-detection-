# PHASE 7.1 – ROBUSTNESS CLAIM AND PERTURBATION AUDIT REPORT

**Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma, Aniket Kumar, Raghav Kumar  
**Date:** November 14, 2026  
**Auditor:** Antigravity Research Verification System  
**Audit Scope:** 20-perturbation synthetic stress tests, failure mode characterization, manuscript claim alignment, and safety boundaries  
**Status:** PASS – SCIENTIFIC CLAIMS ALIGNED WITH EMPIRICAL EVIDENCE  

---

## 1. Executive Summary

Phase 7.1 conducted an independent audit of the robustness evaluations reported across Phase 6A.5 and Section 4 of the Springer Nature manuscript. The audit examined all 20 synthetic perturbation stress tests applied to the verified baseline disease specimen (`Apple___Apple_scab`, original confidence $95.56\%$, $8$ screening candidates).

**Critical Conclusions:**
1. **Critical Failure Mode Identified:** Severe overexposure ($+50\%$ brightness, factor $1.50$) induced a catastrophic silent false negative. The model flipped prediction from `Apple___Apple_scab` to `Apple___healthy` with **$63.79\%$ confidence**, which sits **above** the $0.60$ operational review threshold. This prediction silently bypassed the uncertainty filter as `HIGH_CONFIDENCE`, presenting a diseased leaf as healthy to the user.
2. **Manuscript Claim Correction:** Earlier draft claims asserting that the operational threshold "reliably protects against all degraded inputs" or provides "robust field safety" were scientifically inaccurate. The manuscript has been revised to explicitly document the severe overexposure failure mode, classify it as a critical safety hazard, and acknowledge the inability of scalar confidence thresholds alone to capture contrast-driven lesion erasure.
3. **Synthetic vs. Field Scope:** All 20 tests evaluate numerical and perceptual sensitivity under synthetic digital transformations. They do **not** establish real-world agricultural field readiness.

---

## 2. Complete 20-Perturbation Stress Test Matrix

The table below compiles the empirical results of all 20 perturbation conditions evaluated against the deployed MobileNetV3-Small TorchScript classifier (`saved_models/experiments/mobilenet_v3_small_final/classifier.pt`, SHA-256: `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`).

| ID | Perturbation Category | Transformation Parameters | Predicted Class | Uncalibrated Score | Score Shift ($\Delta$) | Class Preserved? | Prediction State ($\tau = 0.60$) | Screening Candidates |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Baseline Specimen** | Original image ($256 \times 256$) | `Apple___Apple_scab` | **$95.56\%$** | **$0.00\%$** | **Yes** | `HIGH_CONFIDENCE` | **8** |
| 1 | Brightness Reduction | Factor $0.75$ ($-25\%$) | `Apple___Apple_scab` | $93.36\%$ | $-2.20\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 2 | Severe Darkening | Factor $0.50$ ($-50\%$) | `Apple___Apple_scab` | $87.49\%$ | $-8.07\%$ | Yes | `HIGH_CONFIDENCE` | 5 |
| 3 | Brightness Increase | Factor $1.25$ ($+25\%$) | `Apple___Apple_scab` | $87.16\%$ | $-8.40\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| **4** | **Severe Overexposure** | Factor $1.50$ ($+50\%$) | **`Apple___healthy`** | **$63.79\%$** | **$-31.77\%$** | **NO (Flipped)** | **`HIGH_CONFIDENCE`** | **4** |
| 5 | Low Contrast | Factor $0.50$ ($-50\%$) | `Apple___Apple_scab` | $94.75\%$ | $-0.81\%$ | Yes | `HIGH_CONFIDENCE` | 7 |
| 6 | High Contrast | Factor $1.50$ ($+50\%$) | `Apple___Apple_scab` | $94.88\%$ | $-0.68\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 7 | Mild Gaussian Blur | Radius $\sigma = 1.0\text{ px}$ | `Apple___Apple_scab` | $97.94\%$ | $+2.38\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 8 | Moderate Blur | Radius $\sigma = 2.0\text{ px}$ | `Apple___Apple_scab` | $64.88\%$ | $-30.68\%$ | Yes | `HIGH_CONFIDENCE` | 4 |
| 9 | Slight Rotation Tilt | $+15^\circ$ planar rotation | `Apple___Apple_scab` | $49.77\%$ | $-45.79\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 7 |
| 10 | Right-Angle Rotation | $+90^\circ$ clockwise rotation | `Apple___Apple_scab` | $43.95\%$ | $-51.61\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 3 |
| 11 | Inverted Rotation | $+180^\circ$ inverted rotation | `Apple___Apple_scab` | $84.36\%$ | $-11.20\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 12 | Counter-Clockwise | $+270^\circ$ rotation | `Apple___Apple_scab` | $33.86\%$ | $-61.70\%$ | Yes | **`LOW_CONF_UNCERTAIN`** | 2 |
| 13 | Severe Downscaling | Downscaled to $160 \times 160\text{ px}$ | `Apple___Apple_scab` | $97.35\%$ | $+1.79\%$ | Yes | `HIGH_CONFIDENCE` | 2 |
| 14 | Direct CNN Rescaling | Rescaled to $224 \times 224\text{ px}$ | `Apple___Apple_scab` | $93.99\%$ | $-1.57\%$ | Yes | `HIGH_CONFIDENCE` | 4 |
| 15 | Bicubic Upscaling | Upscaled to $600 \times 600\text{ px}$ | `Apple___Apple_scab` | $95.77\%$ | $+0.21\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 16 | Vertical Stretch | Aspect ratio $1:2$ ($200 \times 400$) | `Apple___Apple_scab` | $96.79\%$ | $+1.23\%$ | Yes | `HIGH_CONFIDENCE` | 6 |
| 17 | Horizontal Stretch | Aspect ratio $2:1$ ($400 \times 200$) | `Apple___Apple_scab` | $97.23\%$ | $+1.67\%$ | Yes | `HIGH_CONFIDENCE` | **0** |
| 18 | Mild JPEG Compression | Quality factor $Q = 30$ | `Apple___Apple_scab` | $94.02\%$ | $-1.54\%$ | Yes | `HIGH_CONFIDENCE` | 8 |
| 19 | Heavy JPEG Artifacts | Quality factor $Q = 15$ | `Apple___Apple_scab` | $85.74\%$ | $-9.82\%$ | Yes | `HIGH_CONFIDENCE` | 2 |
| 20 | Additive Sensor Noise | Gaussian noise ($\sigma = 15$) | `Apple___Apple_scab` | $98.44\%$ | $+2.88\%$ | Yes | `HIGH_CONFIDENCE` | 3 |

---

## 3. Systematic 5-Category Robustness Taxonomy

The 20 perturbations fall into five mutually distinct operational outcome categories:

### Category 1: Class Preservation with High Confidence ($\Delta \text{Score} > -15\%$)
- **Perturbations:** Mild Gaussian Blur ($+2.38\%$), Additive Sensor Noise ($+2.88\%$), Severe Downscaling ($+1.79\%$), Horizontal Stretch ($+1.67\%$), Vertical Stretch ($+1.23\%$), Bicubic Upscaling ($+0.21\%$), High Contrast ($-0.68\%$), Low Contrast ($-0.81\%$), Mild JPEG $Q=30$ ($-1.54\%$), Direct CNN Rescale ($-1.57\%$), Brightness $-25\%$ ($-2.20\%$).
- **Behavior:** The model's primary classification remains correct and the top-1 softmax score remains comfortably above $90\%$. The inverted residual blocks tolerate minor high-frequency noise, linear contrast variation, and uniform scaling without feature degradation.

### Category 2: Confidence Reduction Without Class Change
- **Perturbations:**
  - Severe Darkening ($-50\%$ brightness): $95.56\% \rightarrow 87.49\%$ ($\Delta = -8.07\%$).
  - Brightness Increase ($+25\%$ brightness): $95.56\% \rightarrow 87.16\%$ ($\Delta = -8.40\%$).
  - Heavy JPEG Artifacts ($Q=15$): $95.56\% \rightarrow 85.74\%$ ($\Delta = -9.82\%$).
  - Inverted Rotation ($180^\circ$): $95.56\% \rightarrow 84.36\%$ ($\Delta = -11.20\%$).
  - Moderate Gaussian Blur ($\sigma = 2.0$): $95.56\% \rightarrow 64.88\%$ ($\Delta = -30.68\%$).
- **Behavior:** The correct disease class (`Apple___Apple_scab`) is maintained, but the softmax probability mass disperses. Moderate blur approaches the operational boundary ($64.88\%$ vs. $0.60$ threshold), demonstrating that texture attenuation rapidly erodes discriminative signal.

### Category 3: Prediction-State Transitions to `LOW_CONFIDENCE_UNCERTAIN`
- **Perturbations:**
  - Slight Planar Tilt ($+15^\circ$): Score dropped to **$49.77\%$** ($\Delta = -45.79\%$).
  - Right-Angle Clockwise ($+90^\circ$): Score dropped to **$43.95\%$** ($\Delta = -51.61\%$).
  - Counter-Clockwise ($+270^\circ$): Score dropped to **$33.86\%$** ($\Delta = -61.70\%$).
- **Behavior:** While the argmax class remained `Apple___Apple_scab`, the confidence collapsed below the $0.60$ review threshold. The application successfully intercepted these outputs, transitioning them to `LOW_CONFIDENCE_UNCERTAIN` and warning the user to seek expert verification.
- **Root Cause:** PlantVillage images exhibit strong canonical vertical orientations (leaf apex pointing up or petiole pointing down). The learned convolutional weights possess an orientation bias that penalizes tilted geometries.

### Category 4: Candidate-Count Discrepancies in Screening Pipeline
- **Perturbations:**
  - Horizontal Stretch ($2:1$, $400 \times 200$): Whole-leaf score remained $97.23\%$, but screening proposals dropped from **$8$ to $0$**.
  - Severe Downscaling ($160 \times 160$): Candidates dropped from **$8$ to $2$**.
  - Moderate Blur ($\sigma = 2.0$): Candidates dropped from **$8$ to $4$**.
  - Heavy JPEG Artifacts ($Q=15$): Candidates dropped from **$8$ to $2$**.
  - Additive Noise ($\sigma = 15$): Candidates dropped from **$8$ to $3$**.
  - Counter-Clockwise ($270^\circ$): Candidates dropped from **$8$ to $2$**.
- **Behavior:** Candidate proposal depends on sliding window crops and ExG (Excess Green) foliar saliency thresholds. Geometric canvas distortion or excessive block compression disrupts the patch aspect ratio and background-leaf thresholding, suppressing candidate extraction even when the whole-leaf classification pipeline (which resizes the entire image to $224 \times 224$) succeeds.

### Category 5: Critical Silent Failure Mode (Overexposure Lesion Erasure)
- **Perturbation:** Severe Overexposure ($+50\%$ brightness, factor $1.50$).
- **Output:** **`Apple___healthy`** with **$63.79\%$ score** ($4$ candidates proposed).
- **Failure Classification:** **Catastrophic Silent False Negative.**
- **Detailed Mechanism:**
  1. Apple scab (*Venturia inaequalis*) manifests as dark olive-green to black velvety necrotic lesions on the adaxial leaf surface.
  2. MobileNetV3-Small detects scab primarily through spatial gradient contrast between dark lesion boundaries and the lighter green leaf lamina.
  3. Increasing image brightness by $50\%$ saturates high-luminance pixel channels to the $255$ clipping boundary. This erases the darker textural contrast of the scab lesions, shifting their pixel histograms into the normal green foliage range.
  4. The model misinterprets the washed-out surface as an unblemished leaf, outputting `Apple___healthy`.
  5. Because the uncalibrated score is **$63.79\%$**, it exceeds the operational review threshold of $0.60$.
  6. The system therefore outputs a `HIGH_CONFIDENCE` state, displaying a green affirmative badge to the user.
  7. **Agronomic Impact:** In a field setting, this failure mode would certify an actively infected, contagious plant as completely healthy, preventing fungicide application and risking crop-wide pathogen spread.

---

## 4. Manuscript Claim Verification and Reconciliation

The auditor verified that all overreaching or ambiguous robustness claims in the manuscript have been revised:

| Manuscript Section | Original Draft Wording / Risk | Revised Verified Wording | Audit Status |
| :--- | :--- | :--- | :--- |
| **Abstract** | Implied uncalibrated threshold guarantees user safety across field variations. | Clarified that the $0.60$ threshold filters moderate rotational uncertainty, but does not prevent contrast-driven misclassifications like severe overexposure. | **ALIGNED** |
| **Section 4.4 (Robustness)** | Generic statement that model was "robust to 19 of 20 perturbations". | Explicitly details the severe overexposure failure mode ($63.79\%$ healthy prediction) as a silent false negative bypassing the $0.60$ review filter. | **ALIGNED** |
| **Section 5.1 (Validation Gaps)** | Failed to highlight physical camera lighting as a core safety risk. | Explains that scalar confidence thresholds cannot detect lesion erasure caused by solar glint or overexposure, mandating hardware/exposure pre-checks. | **ALIGNED** |
| **Section 6 (Conclusion)** | Overly optimistic field readiness tone. | Replaced with clear scientific caution: the system is an exploratory screening prototype with documented validation gaps, not an autonomous diagnostic agent. | **ALIGNED** |

---

## 5. Audit Conclusion

The robustness analysis of the Plant Disease Screening System is scientifically rigorous and fully reconciled with empirical evidence. The documentation of the severe overexposure failure mode is honest, accurate, and essential for agricultural deployment ethics.

