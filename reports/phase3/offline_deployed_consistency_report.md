# Offline versus Deployed Inference Consistency Report

**Date:** 2026-11-12T09:35:39Z  
**Execution Environment:** Python 3.12.5 (Windows, CPU-only PyTorch)  
**Model Architecture:** MobileNetV3-Small (38-class plant disease classification)  
**Checkpoint Path:** `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\saved_models\experiments\mobilenet_v3_small_final\best_model.pth`  
**Checkpoint SHA-256:** `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009`  
**Combined Manifest SHA-256:** `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868`  

---

## 1. Executive Summary

This report audits and verifies numerical, categorical, and logical consistency between the **offline research evaluation pipeline** and the **deployed FastAPI inference backend** (`backend.app.services.models.ModelRegistry`).

### Audit Verdict: **PASSED (100% BIT-FOR-BIT CONSISTENT)**

- **Predicted Class Equality:** 100% agreement across all test samples.
- **Calibrated Confidence Discrepancy:** Maximum absolute difference $\le 1.19 	imes 10^{-7}$ (within single-precision floating-point epsilon).
- **Class Scores Discrepancy (38 Classes):** Maximum absolute score difference $\le 1.19 	imes 10^{-7}$.
- **Review Decision Flag (`needs_review`):** 100% identical logical outcomes.
- **Confidence Calibrated Flag (`confidence_calibrated`):** Verified `True` in deployed service metadata.

---

## 2. Configuration & Parameter Reconciliation

| Parameter | Legacy Backend Setting | Calibrated Definitive Value | Parameter Source |
| :--- | :--- | :--- | :--- |
| **Model Checkpoint** | `saved_models/classifier.pt` | `saved_models/experiments/mobilenet_v3_small_final/best_model.pth` | Phase 2 Definitive Baseline |
| **Calibration Temperature ($T$)** | `1.0` (uncalibrated) | `0.553813` | L-BFGS optimization on 5,429 Calibration images |
| **Review Threshold ($\tau$)** | `0.60` (arbitrary heuristic) | `0.223135` | Calibration partition empirical threshold policy |
| **Target Accuracy Policy** | None | $\ge 90.0\%$ | Empirical calibration partition selective accuracy |
| **Input Image Resolution** | $224 \times 224$ | $224 \times 224$ | Standard MobileNetV3-Small input geometry |
| **Attribution / Grad-CAM** | MobileNetV3-Small | MobileNetV3-Small state dict load | `ModelRegistry.gradcam` verified |

---

## 3. Sample-by-Sample Consistency Audit

The following table details the comparison for representative holdout test images from both the **PlantVillage holdout test partition** and the **official PlantDoc external test set**:

| # | Dataset Origin | Image File | True Label | Offline Predicted | Deployed Predicted | Offline Conf. | Deployed Conf. | Max Score Diff | Needs Review (Off/Dep) | Verdict |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :---: |
| 1 | PlantVillage Holdout Test | `0208f4eb-45a4-4399-904e-989ac2c6257c___FREC_Scab 3037.JPG` | `Apple___Apple_scab` | `Apple___Apple_scab` | `Apple___Apple_scab` | 0.4884 | 0.4884 | 0.0e+00 | False/False | **PASS** |
| 2 | PlantVillage Holdout Test | `94ef739e-57a7-4f97-be8e-6ba2cfd16780___JR_FrgE.S 8664.JPG` | `Apple___Black_rot` | `Apple___Black_rot` | `Apple___Black_rot` | 1.0000 | 1.0000 | 0.0e+00 | False/False | **PASS** |
| 3 | PlantVillage Holdout Test | `5ebfa4f5-0bcb-4055-a44b-612cbbfdeb96___JR_HL 3987.JPG` | `Cherry_(including_sour)___healthy` | `Cherry_(including_sour)___healthy` | `Cherry_(including_sour)___healthy` | 0.9999 | 0.9999 | 0.0e+00 | False/False | **PASS** |
| 4 | PlantVillage Holdout Test | `99ee381f-c18c-4cf4-b62a-af315af69d5d___RS_NLB 3493.JPG` | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Northern_Leaf_Blight` | 0.9094 | 0.9094 | 0.0e+00 | False/False | **PASS** |
| 5 | PlantVillage Holdout Test | `057d2526-1752-4d6c-902d-85f85ec93d41___Rut._Bact.S 0806.JPG` | `Peach___Bacterial_spot` | `Peach___Bacterial_spot` | `Peach___Bacterial_spot` | 0.9978 | 0.9978 | 0.0e+00 | False/False | **PASS** |
| 6 | PlantDoc External Test | `052609_20Hartman_20Crabapple_20scab_20single_20leaf.JPG.jpg` | `Apple___Apple_scab` | `Apple___Apple_scab` | `Apple___Apple_scab` | 0.5966 | 0.5966 | 0.0e+00 | False/False | **PASS** |
| 7 | PlantDoc External Test | `0605_Rust-induced_leafspot.jpg` | `Apple___Cedar_apple_rust` | `Apple___Cedar_apple_rust` | `Apple___Cedar_apple_rust` | 0.4446 | 0.4446 | 0.0e+00 | False/False | **PASS** |
| 8 | PlantDoc External Test | `leaf-blueberry-15281271.jpg` | `Blueberry___healthy` | `Peach___healthy` | `Peach___healthy` | 0.4971 | 0.4971 | 0.0e+00 | False/False | **PASS** |
| 9 | PlantDoc External Test | `stock-photo-peach-leaf-isolated-on-white-background-281097503.jpg` | `Peach___healthy` | `Peach___healthy` | `Peach___healthy` | 0.9617 | 0.9617 | 0.0e+00 | False/False | **PASS** |
| 10 | PlantDoc External Test | `Shoemaker_7068.JPG.jpg` | `Tomato___Early_blight` | `Raspberry___healthy` | `Raspberry___healthy` | 0.3792 | 0.3792 | 0.0e+00 | False/False | **PASS** |

---

## 4. Architectural & Preprocessing Parity

1. **Preprocessing Pipeline:**
   - Both offline and deployed pipelines apply torchvision transforms:
     ```python
     transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])
     ```
2. **Inference Mode & Determinism:**
   - Both pipelines invoke `torch.inference_mode()` with `model.eval()`.
   - Dropout is completely deactivated in both execution paths.
3. **Probability Calculation:**
   - Offline formulation: `torch.softmax(logits / 0.553813, dim=1)`
   - Deployed formulation: `torch.softmax(logits / self.temperature, dim=1)`
   - Both yield mathematically identical probability distributions over all 38 disease categories.

---

## 5. Deployment Recommendation

1. Production deployments should point `settings.model_dir` to `saved_models/experiments/mobilenet_v3_small_final/`.
2. The legacy production checkpoint `saved_models/classifier.pt` and candidate `saved_models/candidate/best_state.pt` remain completely untouched to support rollback if required.
3. The temperature $T = 0.5538$ and review threshold $\tau = 0.2231$ provide calibrated risk screening for end users.
