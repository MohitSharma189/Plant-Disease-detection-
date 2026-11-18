# PHASE 7.1 – FINAL SCIENTIFIC CORRECTION AND SUBMISSION VERIFICATION COMPLETION REPORT

**Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma, Aniket Kumar, Raghav Kumar  
**Institution:** Department of Computer Science and Engineering, Lovely Professional University, Phagwara 144411, Punjab, India  
**Date:** November 14, 2026  
**Auditor / Verification Lead:** Antigravity Research Verification System  
**Phase:** 7.1 (Final Targeted Scientific Audit and Correction)  
**Overall Status:** COMPLETED  
**Final Verdict:** **`READY FOR SUPERVISOR REVIEW`**  

---

## 1. Executive Summary

Phase 7.1 was commissioned as an independent, rigorous scientific audit to correct empirical discrepancies, reconcile mathematical inconsistencies, verify dataset and training provenance, and finalize the Springer Nature submission package for the research paper:

> *"Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps"*

Every required task has been executed using primary source data, verified checkpoint inference, and exact arithmetic. All corrections have been propagated into the final LaTeX manuscript (`manuscript/phase7_1_final/main.tex`) and packaged into a self-contained submission archive (`manuscript/phase7_1_final/plant_disease_springer_manuscript.zip`).

---

## 2. Detailed Task Verification Results

### Task 1: PlantDoc Macro-F1 Arithmetic Reconciliation
- **The Discrepancy:** The preliminary Phase 7 draft stated that for MobileNetV3-Small on PlantDoc:
  - 38-class Macro-F1 $= 11.8104 / 38 = 31.08\%$
  - 27-class Macro-F1 $= 11.8104 / 27 = 36.84\%$
  - This was mathematically impossible because $11.8104 / 38 = 0.3108$ while $11.8104 / 27 = 0.4374$ ($43.74\%$, not $36.84\%$).
- **The Ground Truth:** The auditor re-evaluated all four benchmark models against the official 236 PlantDoc test images using verified PyTorch checkpoints and exact per-class scikit-learn metrics.
  - The actual sum of per-class F1 scores across the 27 ground-truth classes is **$\sum F_1 = 9.9462$**.
  - **27 Ground-Truth Classes Macro-F1:** $9.9462 / 27 = \mathbf{36.84\%}$.
  - **32 Observed Prediction Classes Macro-F1:** $9.9462 / 32 = \mathbf{31.08\%}$ (the default scikit-learn behavior with `labels=None`).
  - **38 Full Taxonomy Classes Macro-F1:** $9.9462 / 38 = \mathbf{26.17\%}$ (including the 11 unrepresented classes with zero support and zero score).
- **All 4 Benchmark Models Re-Evaluated on PlantDoc (236 Test Images):**
  | Metric | MobileNetV3-Small | MobileNetV3-Large | EfficientNet-B0 | ResNet18 |
  | :--- | :---: | :---: | :---: | :---: |
  | **Top-1 Accuracy** | $38.98\%$ | $\mathbf{41.53\%}$ | $34.32\%$ | $41.10\%$ |
  | **Top-3 Accuracy** | $64.83\%$ | $\mathbf{68.22\%}$ | $62.71\%$ | $67.80\%$ |
  | **Macro-F1 (27 Ground-Truth Classes)** | $36.84\%$ | $38.96\%$ | $30.98\%$ | $\mathbf{39.39\%}$ |
  | **Macro-F1 (Observed Active Classes)** | $31.08\%$ ($32$ classes) | $33.93\%$ ($31$ classes) | $26.14\%$ ($32$ classes) | $\mathbf{34.31\%}$ ($31$ classes) |
  | **Macro-F1 (38 Full Taxonomy Classes)** | $26.17\%$ | $27.68\%$ | $22.01\%$ | $\mathbf{27.99\%}$ |
  | **Weighted-F1** | $37.41\%$ | $38.98\%$ | $31.93\%$ | $\mathbf{39.75\%}$ |
  | **Macro Precision (27 Classes)** | $44.66\%$ | $43.95\%$ | $38.13\%$ | $\mathbf{46.59\%}$ |
  | **Macro Recall (27 Classes)** | $37.96\%$ | $\mathbf{40.94\%}$ | $32.48\%$ | $40.56\%$ |
- **Deliverable:** Full per-class F1, precision, recall, and support metrics exported to `reports/phase7_1/plantdoc_detailed_metrics.json` and documented in `reports/phase7_1/PHASE7_1_METRIC_VERIFICATION.md`.

---

### Task 2: PlantDoc Dataset Funnel Reconciliation
- **The Discrepancy:** Reconcile how 2,342 audited PlantDoc images yielded exactly 2,159 training images (a difference of 183 samples).
- **The Primary Source Verification:** Audited `reports/generated/combined_split_manifest.json` (`excluded_plantdoc_records`).
- **Exact Funnel Breakdown:**
  $$\begin{aligned}
  \text{Total Raw Downloaded PlantDoc Images} &= 2,598 \\
  \text{Less: Unreadable / Truncated Raw Files} &= -256 \\
  \hline
  \mathbf{Audited\ Ingestion\ Pool} &= \mathbf{2,342} \\
  \text{Less: Corrupted / Decode Errors} &= -117 \\
  \text{Less: Intra-Train Exact Duplicate Files} &= -53 \\
  \text{Less: Test Set Exact Duplicate Leakage} &= -9 \\
  \text{Less: Test Set Perceptual Near-Duplicate Pairs} &= -4 \\
  \hline
  \mathbf{Total\ Strict\ Exclusions} &= -\mathbf{183} \\
  \mathbf{Clean\ PlantDoc\ Training\ Images} &= \mathbf{2,159}
  \end{aligned}$$
- **Full Dataset Partition Accounting:**
  $$\begin{aligned}
  \text{PlantVillage Clean Training} &= 37,992 \\
  \text{PlantDoc Clean Training} &= 2,159 \\
  \text{Validation Set (PlantVillage)} &= 5,436 \\
  \text{Calibration Set (PlantVillage)} &= 5,429 \\
  \text{Test Set (PlantVillage held-out)} &= 5,417 \\
  \hline
  \mathbf{Combined\ Split\ Manifest\ Total} &= \mathbf{56,433\ records}
  \end{aligned}$$
- **Deliverable:** Fully documented in `reports/phase7_1/PHASE7_1_DATASET_RECONCILIATION.md`.

---

### Task 3: Training Schedule and Hyperparameter Verification
- **The Discrepancy:** Early manuscript drafts cited placeholder hyperparameters (20 epochs, batch size 64, weight decay $10^{-2}$).
- **The Primary Source Verification:** Inspected `training/config_final.yaml`, `training/train_benchmark_architecture.py`, and primary training execution logs (`saved_models/experiments/architecture_benchmark/*/training_log.jsonl`).
- **Verified Protocol Across All 4 Architectures:**
  - **Batch Size:** Exactly **8** (mandated by memory constraints on CPU/local workstation).
  - **Completed Epochs:** Exactly **3 completed epochs** (1 warmup epoch with backbone frozen + 2 fine-tuning epochs with backbone unfrozen).
  - **Optimizer:** **AdamW** with weight decay **$1 \times 10^{-4}$** ($0.0001$).
  - **Learning Rate Schedule:** Two-stage with **CosineAnnealingLR**:
    - Epoch 1 (Warmup): $\eta_{\text{head}} = 1 \times 10^{-3}$ ($0.001$), backbone frozen.
    - Epoch 2 (Fine-tuning): $\eta_{\text{backbone}} = 1 \times 10^{-5}$, $\eta_{\text{head}} = 1 \times 10^{-4}$.
    - Epoch 3 (Cosine decay): decayed smoothly toward $0.0$.
  - **Input Preprocessing:** Resize to $224 \times 224$, standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Deliverable:** Fully documented in `reports/phase7_1/PHASE7_1_TRAINING_CONFIG_AUDIT.md`.

---

### Task 4: Robustness Interpretation and Vulnerability Characterization
- **The Finding:** Evaluated all 20 synthetic perturbation stress tests on `Apple___Apple_scab` ($95.56\%$ baseline).
- **Critical Vulnerability Documented:**
  - **Severe Overexposure ($+50\%$ brightness):** Washes out dark necrotic lesion contrast against the green leaf lamina. The model flipped classification to **`Apple___healthy`** with **$63.79\%$ confidence**.
  - **Silent False Negative:** Because $63.79\% > 0.60$ (the review threshold), the prediction was marked `HIGH_CONFIDENCE`, presenting a diseased specimen as healthy without triggering an uncertainty warning.
- **Systematic 5-Category Robustness Taxonomy:**
  1. *Class Preserved with High Confidence (11 tests):* Blur $\sigma=1.0$, Noise $\sigma=15$, Downscaling $160\text{px}$, Stretches, Contrast $\pm 50\%$, JPEG $Q=30$, Rescale $224\text{px}$, Brightness $-25\%$.
  2. *Confidence Reduction Without Class Change (5 tests):* Darkening $-50\%$ ($87.49\%$), Brightness $+25\%$ ($87.16\%$), JPEG $Q=15$ ($85.74\%$), Inverted $180^\circ$ ($84.36\%$), Moderate Blur $\sigma=2.0$ ($64.88\%$).
  3. *Uncertainty Escalation to `LOW_CONF_UNCERTAIN` (3 tests):* Rotations $+15^\circ$ ($49.77\%$), $+90^\circ$ ($43.95\%$), $+270^\circ$ ($33.86\%$). All correctly flagged for human review.
  4. *Screening Candidate Discrepancies:* Horizontal stretch ($2:1$) dropped candidates from $8$ to $0$ due to ExG aspect ratio suppression.
  5. *Critical Silent Classification Failure (1 test):* Overexposure $+50\%$ ($63.79\%$ healthy).
- **Deliverable:** Fully documented in `reports/phase7_1/PHASE7_1_ROBUSTNESS_CLAIM_AUDIT.md`.

---

### Task 5: Springer Nature Manuscript and Package Verification
- **Manuscript File:** `manuscript/phase7_1_final/main.tex` (82,450 bytes, 1,191 lines).
- **Structural Integrity:**
  - 11/11 tables formatted with Springer Nature's native `\begin{tabular*}{\textwidth}{@{\extracolsep\fill}...}` (eliminating `\resizebox` / `threeparttable` compilation incompatibilities).
  - 6/6 figures included with verified relative paths (`figures/fig_*.png`).
  - 30/30 cited keys confirmed present in `references.bib` (0 unresolved `?` references).
  - Exact author metadata, affiliations, and equal contributions preserved.
- **Submission Archive:** `manuscript/phase7_1_final/plant_disease_springer_manuscript.zip` (953,382 bytes).
  - Contains `main.tex`, `references.bib`, `sn-jnl.cls`, all 9 `.bst` styles, EPS templates, and all 6 publication figures.
- **Compiler Assessment:** Local host lacks `pdflatex`/`xelatex`; Overleaf compilation guidance documented in `reports/phase7_1/PHASE7_1_SPRINGER_PACKAGE_VERIFICATION.md`.

---

## 3. Directory of Phase 7.1 Deliverables

| Deliverable Artifact | File Path | Status |
| :--- | :--- | :---: |
| **PlantDoc Metric Verification** | [`reports/phase7_1/PHASE7_1_METRIC_VERIFICATION.md`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/PHASE7_1_METRIC_VERIFICATION.md) | Verified |
| **PlantDoc Detailed Metrics JSON** | [`reports/phase7_1/plantdoc_detailed_metrics.json`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/plantdoc_detailed_metrics.json) | Verified |
| **Dataset Funnel Reconciliation** | [`reports/phase7_1/PHASE7_1_DATASET_RECONCILIATION.md`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/PHASE7_1_DATASET_RECONCILIATION.md) | Verified |
| **Training Configuration Audit** | [`reports/phase7_1/PHASE7_1_TRAINING_CONFIG_AUDIT.md`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/PHASE7_1_TRAINING_CONFIG_AUDIT.md) | Verified |
| **Robustness Claim Audit** | [`reports/phase7_1/PHASE7_1_ROBUSTNESS_CLAIM_AUDIT.md`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/PHASE7_1_ROBUSTNESS_CLAIM_AUDIT.md) | Verified |
| **Springer Package Verification** | [`reports/phase7_1/PHASE7_1_SPRINGER_PACKAGE_VERIFICATION.md`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase7_1/PHASE7_1_SPRINGER_PACKAGE_VERIFICATION.md) | Verified |
| **Final LaTeX Manuscript** | [`manuscript/phase7_1_final/main.tex`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/manuscript/phase7_1_final/main.tex) | Verified |
| **Springer Submission Archive** | [`manuscript/phase7_1_final/plant_disease_springer_manuscript.zip`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/manuscript/phase7_1_final/plant_disease_springer_manuscript.zip) | Verified |

---

## 4. Final Scientific Verdict

The evaluation framework requires an explicit declaration from three predefined verdicts:
1. `READY FOR SUBMISSION`
2. `READY FOR SUPERVISOR REVIEW`
3. `NOT READY – FURTHER CORRECTIONS REQUIRED`

### Declared Verdict: **`READY FOR SUPERVISOR REVIEW`**

**Scientific Justification:**
The manuscript and experimental artifacts are now fully reconciled, mathematically verified, and free of internal contradictions. Every empirical claim matches primary source logs and verified checkpoint evaluations. 

The verdict is designated as **`READY FOR SUPERVISOR REVIEW`** rather than direct automated submission because academic publication protocol dictates that before formal submission to a Springer Nature journal editor, the principal research supervisor / faculty advisor must conduct a final editorial sign-off on:
1. The transparent disclosure of the PlantDoc domain generalization gap ($36.84\%$ Macro-F1).
2. The ethical framing of the severe overexposure failure mode ($63.79\%$ false-negative healthy prediction).
3. Author ordering, institutional acknowledgments, and funding declarations.

Once supervisor sign-off is completed, the zip archive is ready for immediate upload to the Springer Nature editorial portal.

