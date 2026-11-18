# Phase 6 Final Scientific Audit & Correction Verification Report

**Manuscript Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma$^{1*}$, Aniket Kumar$^{1\dagger}$, Raghav Kumar$^{1\dagger}$ ($^\dagger$equal contribution)  
**Affiliation:** $^{1}$Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, 144411, India  
**Target Publication Venue:** Springer Nature Applied Intelligence / Computers and Electronics in Agriculture  
**Manuscript Output:** `reports/research_paper/final_correction/plant_disease_research_paper.pdf`  
**Page Count:** Strictly 18 pages (Meeting the 18–22 page requirement)  
**Date of Audit:** November 13, 2026  

---

## Executive Summary

This report documents the exhaustive scientific audit and numerical reconciliation of the research paper in response to the faculty review guidelines. In accordance with the prompt's instructions:
1. No changes were made to paper identity: title, authors, affiliations, email addresses, Springer Nature typography, page layout, and overall section structure remain strictly preserved.
2. Every numerical metric, dataset split count, latency measurement, and execution parameter was cross-examined against the underlying experimental artifacts (Phase 1–5 reports, CSVs, JSON manifests, and execution scripts).
3. All discrepancies identified during the audit were either rigorously reconciled with mathematical and empirical proof or explicitly designated as unresolved methodological variations.
4. All prescriptive agronomic recommendations (chemical fungicides, pesticides, quarantine mandates) were removed from Appendix E and Table E1, replaced by responsible human-in-the-loop screening protocols.

---

## 1. Item-by-Item Numerical & Methodological Audit Table

| Audit Item | Original Value in Manuscript | Verified Artifact Value | Corrected Value | Source Artifact / Experiment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined Manifest Total** | 56,879 images | 56,433 records | **56,433 images** | `reports/generated/combined_split_manifest.json` (SHA-256: `f9ef8b9d...15868`) | **RESOLVED** |
| **Table 1 Partition Sum** | 56,669 images | 56,433 manifest + 236 external | **56,433 manifest total**; **236 external benchmark** | `combined_split_manifest.json` & PlantDoc test split | **RESOLVED** |
| **PlantVillage Source Total** | 54,305 images | 54,305 images across 38 classes | **54,305 images** | `reports/phase1/dataset_integrity_report.md` | **RESOLVED** |
| **Duplicate Exclusions** | 21 images | 31 images total (10 cross-label + 21 intra-class) | **31 duplicate images excluded** (10 cross-label conflicting + 21 intra-class exact) | `training/prepare_splits.py` & `reports/generated/split_manifest.json` | **RESOLVED** |
| **Partitioned PlantVillage Images** | 54,284 images | 54,305 − 31 = 54,274 images | **54,274 eligible images** (37,992 train, 5,436 val, 5,429 calib, 5,417 test) | `reports/generated/split_manifest.json` | **RESOLVED** |
| **Added PlantDoc Training Images** | 2,159 images | 2,159 images added to train split only | **2,159 images** (Combined train = 40,151) | `reports/generated/combined_split_manifest.json` | **RESOLVED** |
| **Phase 3 Calibration (MVS)** | $T^* = 0.5538$, ECE 0.58% | $T^* = 0.5538$, ECE 0.58% (calib), 0.74% (holdout) | **$T^* = 0.5538$, ECE 0.58%** | `reports/phase3/calibration_metrics.json` | **RESOLVED** |
| **Phase 5 Calibration (MVS)** | $T^* = 0.9902$, ECE 9.94% | $T^* = 0.9902$, post-calib test ECE 9.94% | **$T^* = 0.9902$, test ECE 9.94%** | `reports/phase5/calibration_comparison.csv` | **RESOLVED** |
| **Calibration Discrepancy Attribution** | Attributed to 50 vs. 10 L-BFGS iterations | Script inspection reveals: log-T vs raw-T, lr 0.1 vs 0.01, Wolfe line search vs no line search, calib vs test split | **Unresolved Methodological Variation** (documented transparently without speculative assertions) | `training/evaluate_classifier.py` vs `training/run_phase5_evaluation.py` | **UNRESOLVED (Reported Transparently)** |
| **PlantDoc Selective Coverage** | 93.22% | 220 / 236 accepted images ($93.2203\%$) | **93.22%** (16 rejected, 220 retained) | `reports/phase3/plantdoc_external_metrics.csv` | **RESOLVED** |
| **PlantDoc Selective Accuracy** | 40.91% | 90 / 220 correct predictions ($40.9091\%$) | **40.91%** (vs 38.98% non-selective) | `reports/phase3/plantdoc_external_metrics.csv` | **RESOLVED** |
| **Review Threshold Rejection Count** | Threshold 0.2231 rejected 0 calib samples | Calibration split has min confidence 0.2231 $\implies$ 0 rejected; PlantDoc test has 16 samples $< 0.2231$ | **0 calib rejected (100% coverage)**; **16 PlantDoc rejected (6.78%)** | `reports/phase3/calibration_metrics.json` & `plantdoc_external_metrics.csv` | **RESOLVED** |
| **Phase 4 Baseline Latency** | 167.02 ms vs 117.16 ms | 117.16 ms was model forward pass alone; 167.02 ms was total end-to-end pipeline latency in harness | **117.16 ms forward inference**; **167.02 ms total end-to-end latency** | `reports/phase4/screening_metrics.csv` vs `reports/phase4_1/latency_benchmark.csv` | **RESOLVED** |
| **Phase 4.1 Dev Recall vs Test Recall** | Mixed 32.71%, 34.57%, 38.05% | Dev set (100 imgs): Config E = 32.71%, NMS Max 10 = 34.57%; Test benchmark (236 imgs, 452 leaves): **38.05%** | **Distinguished Dev Set (34.57%) from Final Test Benchmark (38.05%)** | `reports/phase4_1/candidate_generation_comparison.csv` vs `localization_metrics.csv` | **RESOLVED** |
| **Localization Precision & F1 @ 0.25** | Precision 10.60%, F1 16.58% | TP = 172, FP = 1,451 $\implies$ Prec = 10.60%, F1 = 16.58% | **Precision = 10.60%**, **F1 = 16.58%** | `reports/phase4_1/localization_metrics.csv` | **RESOLVED** |
| **False Positives Per Image @ 0.25** | 6.15 FP/img | 1,451 FP / 236 images = 6.1483 FP/img | **6.15 FP/image** | `reports/phase4_1/baseline_vs_optimized.csv` | **RESOLVED** |
| **Localization Matching Formulation** | Unclear class dependence | Recall 38.05% is class-agnostic leaf localization; 66 of 172 TPs had correct pathology class (38.37% class acc) | **Class-agnostic leaf localization (38.05%)**; **38.37% class accuracy on localized candidates** | `reports/phase4_1/localization_metrics.csv` | **RESOLVED** |
| **CPU Latency Stage Breakdown Sum** | Stages summed to 486.06 ms (Total: 844.25 ms) | Missing 358.19 ms is candidate crop extraction, 12% padding, PIL bicubic resize, & tensor normalization | **Added Crop Extraction & Tensor Prep row (358.19 ms)**; sum strictly matches **844.25 ms** | `scratch/run_phase4_1_final_test_evaluation.py` lines 231–241 | **RESOLVED** |
| **Training Epochs & Batch Size** | Appendix reported 15 epochs, batch size 16 | Definitive run used 3 epochs (1 head warmup + 2 fine-tune), batch size 8 | **Batch Size = 8**, **Epochs = 3**, script `training/train_definitive.py` | `training/config_final.yaml` & `reports/phase2/final/PHASE2_COMPLETION_REPORT.md` | **RESOLVED** |
| **Software Package Versions** | Python 3.10.11, PyTorch 2.1.2, OpenCV 4.8.1 | Actual environment: Python 3.12.5, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, OpenCV 5.0.0, NumPy 2.5.4 | **Updated to exact verified runtime versions** | Python `.venv` metadata inspection | **RESOLVED** |
| **Citation Years & BibTeX Keys** | 10 key-year mismatches (e.g. `chen2021` with year 2020) | Actual publication years verified against DOIs | **Aligned all 10 keys and in-text APA citations to true publication years** | Cross-checked against authoritative publishers | **RESOLVED** |
| **Agronomic Prescriptive Directives** | Contained fungicide names (chlorothalonil) & quarantine orders | Research prototype cannot provide validated medical/chemical prescriptions | **Removed all chemical/quarantine directives; replaced with human-in-the-loop screening protocols** | Appendix E and Table E1 | **RESOLVED** |

---

## 2. In-Depth Analysis of Critical Corrections

### 1. Dataset Accounting Reconciliation
The confusion between 56,879, 56,669, and 56,433 arose from mixing the training manifest with the external test set and initial duplicate tallies:
* **PlantVillage Source:** 54,305 total readable images.
* **Deduplication Audit:** Exact SHA-256 and perceptual dHash ($\le 3$ bits) identified **31 duplicate images**:
  * 10 cross-label conflicting duplicates (e.g., identical leaves annotated with contradictory disease labels across classes) were completely purged to avoid label noise.
  * 21 intra-class redundant exact duplicates were pruned.
* **Eligible PlantVillage Partition:** $54,305 - 31 = 54,274$ images.
* **Partition Allocation (Seed 42):**
  * Train: 37,992 images (70.00%)
  * Validation: 5,436 images (10.02%)
  * Calibration: 5,429 images (10.00%)
  * Holdout Test: 5,417 images (9.98%)
* **PlantDoc Training Images Added:** 2,159 deduplicated images were integrated exclusively into the training partition.
* **Final Combined Split Manifest Total:** $37,992 + 2,159 + 5,436 + 5,429 + 5,417 = \mathbf{56,433\text{ images}}$ (SHA-256: `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868`).
* **Official External PlantDoc Test Set:** 236 images (452 ground-truth leaves across 13 cross-mapped classes) remained untouched and strictly external. The old figure 56,669 was simply $56,433 + 236$. Both are now clearly delineated.

### 2. Probability Calibration Discrepancy (Phase 3 vs. Phase 5)
A thorough audit of the actual Python scripts resolved the empirical differences:
* **Phase 3 Script (`training/evaluate_classifier.py`):**
  * Parameterization: $\log T$ with $\log T_0 = 0 \implies T_0 = 1.0$.
  * Optimizer: L-BFGS with `lr = 0.1`, `max_iter = 80`, `line_search_fn = "strong_wolfe"`.
  * Objective: NLL minimized on the 5,429-sample calibration split.
  * Result: $T^* = 0.5538$, reducing calibration ECE from $10.25\% \to 0.58\%$ ($0.74\%$ on holdout test).
* **Phase 5 Script (`training/run_phase5_evaluation.py`):**
  * Parameterization: Direct temperature $T$ initialized at $T_0 = 1.5$.
  * Optimizer: L-BFGS with `lr = 0.01` (10x smaller learning rate), `max_iter = 50`, without line search.
  * Metric Target: Evaluated on the 5,417-sample test split.
  * Result: $T^* = 0.9902$, post-calibration test ECE reported as $9.94\%$.
* **Audit Resolution:** Both scripts evaluated checkpoint `6190f54c...` on partition `f9ef8b9d...`. However, because controlled loss surface ablation has not isolated the relative contributions of the learning rate, line search, and initialization, this discrepancy is **classified as an unresolved methodological variation** between independent evaluation pipelines rather than asserting optimizer failure as a proven fact.

### 3. PlantDoc Selective Prediction Verification
* **Threshold Selection:** Review threshold $\tau = 0.2231$ was chosen on the 5,429 PlantVillage calibration samples as the minimum confidence threshold achieving $\ge 90\%$ selective accuracy. Because in-domain calibration confidences were uniformly above 0.2231, **zero samples were rejected on the calibration partition** (coverage = 100.0%, accuracy = 95.23%).
* **PlantDoc Test Evaluation:** When the identical threshold $\tau = 0.2231$ was applied to the 236 out-of-domain PlantDoc images:
  * Denominator: 236 total images.
  * Rejected: Exactly **16 images** ($6.78\%$) fell below 0.2231 and were referred for expert inspection.
  * Retained: Exactly **220 images** ($93.22\%$ selective coverage).
  * Accuracy: Among the 220 accepted images, exactly **90** were correct, yielding a selective top-1 accuracy of **40.91%** ($90 / 220 = 0.40909$) compared to the non-selective accuracy of 38.98% ($92 / 236$).

### 4. Phase 4 vs. Phase 4.1 Screening Discrepancies
The apparent contradictions in recall across tables were traced to partition scopes:
* **Development-Set Ablation (100 images):**
  * Config E composite pipeline: Recall@0.25 = **32.71%**, latency = 863.4 ms (`candidate_generation_comparison.csv`).
  * Hybrid NMS Max 10 (Selected): Recall@0.25 = **34.57%**, FP/img = 6.86 (`nms_comparison.csv`).
  * Hybrid NMS Max 12: Recall@0.25 = **39.03%**, but FP/img rose to 8.08.
* **Final Independent Test Benchmark (236 images, 452 leaves):**
  * Evaluated the selected composite pipeline (Config E + Hybrid NMS Max 10 + 20% context padding).
  * True Positives = 172 leaves $\implies$ **Recall@0.25 = 172 / 452 = 38.05%** (+109.76% relative improvement over Phase 4 baseline of 18.14%).
  * False Positives = 1,451 proposals $\implies$ **Precision@0.25 = 172 / (172 + 1,451) = 10.60%**.
  * F1-Score = **16.58%**; FP/image = **6.15**; Latency = **844.25 ms**.
* **Localization Matching:** Matching is **class-agnostic spatial leaf localization** (measuring candidate bounding-box overlap $\text{IoU} \ge 0.25$ against ground-truth leaves). When joint localization and disease classification is enforced, exactly **66** of the 172 localized leaves have the correct pathology label (**38.37%** crop classification accuracy, consistent with PlantDoc's overall 38.98% classification performance).
* **Baseline Latency:** 117.16 ms was Phase 4 forward model inference alone; 167.02 ms was total end-to-end latency in the Phase 4.1 comparative harness. Both are reported with their respective scopes.

### 5. CPU Latency Stage Breakdown Correction
Inspection of `scratch/run_phase4_1_final_test_evaluation.py` lines 231–241 identified the previously un-attributed 358.19 ms:
1. Window Generation (Multi-scale & Aspect-aware): **0.10 ms**
2. Foliar Saliency Pre-filtering (ExG Index): **24.16 ms**
3. **Candidate Crop Extraction, 12% Dynamic Context Padding, PIL Sub-image Slicing, Bicubic Resampling, & Tensor Normalization (`EVAL_TRANSFORM`)**: **358.19 ms** (averaging ~5.5 ms per extracted crop across 50–80 proposed windows)
4. MobileNetV3-Small Batch Crop Forward Pass: **461.62 ms**
5. Hybrid Two-Stage NMS Post-Processing: **0.18 ms**
* **Sum:** $0.10 + 24.16 + 358.19 + 461.62 + 0.18 = \mathbf{844.25\text{ ms}}$ (exact match with total measured end-to-end latency).
* For Baseline: $0.04 + 0.00 + 67.07 + 99.87 + 0.04 = \mathbf{167.02\text{ ms}}$.

### 6. Reproducibility & Software Environment
* **Training Configuration:** Corrected commands to invoke `training/train_definitive.py --config training/config_final.yaml` with **3 completed epochs** (1 head warmup + 2 fine-tuning), **batch size 8**, initial head lr 0.001, backbone fine-tune lr 1e-5, label smoothing 0.10, seed 42.
* **Environment Metadata:** Updated from erroneous values to exact runtime versions: **Python 3.12.5**, **PyTorch 2.15.0+cpu**, **torchvision 0.29.2+cpu**, **OpenCV 5.0.0**, **NumPy 2.5.4**, **Pillow 13.0.0**, **ReportLab 5.1.0**, **pypdf 7.0.1** on **AMD Ryzen 7 5800HS (8 cores / 16 threads)** running **Windows 10 Home (64-bit)**.

### 7. References & Agronomic Safety
* **Reference Audit:** Fixed 10 mismatched BibTeX keys and in-text APA citations to match true publication years (`Chen et al., 2020`, `Zhang et al., 2019`, `Barbedo, 2016`, `Lu et al., 2017`, `Jiang et al., 2019`, `D'Souza et al., 2023`, `He et al., 2021`, `Barbedo, 2018`, `Zhang et al., 2019`, `Kwak et al., 2023`). All 40 citations are verified peer-reviewed publications.
* **Agronomic Protocol:** Removed all prescriptive chemical recommendations (e.g. chlorothalonil, chemical sprays) and quarantine directives from Appendix E and Table E1. Replaced with responsible human-in-the-loop screening protocols (foliar margin physical inspection, hand lens verification, sampling for extension pathology clinic referral, soil/tissue test recommendation).

---

## 3. Production Deliverables Directory

All corrected assets are preserved in `reports/research_paper/final_correction/`:

1. **`plant_disease_research_paper.pdf`**  
   - Compiled PDF strictly **18 pages** long.
   - Preserves exact Springer Nature typography, geometry (54 pt margins), running headers, and footers.
   - Zero table overflow, all 14 tables and 6 publication figures rendered cleanly.
2. **`main.tex`**  
   - Production-ready Springer Nature LaTeX source (`sn-jnl` class) incorporating all reconciled data, tables, equations, and appendices.
3. **`references.bib`**  
   - Cleaned APA BibTeX library containing 40 genuine peer-reviewed entries with matching publication years and verified DOIs.
4. **`figures/`**  
   - All 6 publication figure PNGs (`fig_confmat_mvs.png`, `fig_confmat_mvl.png`, `fig_calibration.png`, `fig_arch_comparison.png`, `fig_localization.png`, `fig_prcurves.png`).
5. **`plant_disease_research_paper_package.zip`**  
   - Complete package archive containing the PDF, LaTeX source, BibTeX library, figures, and this verification report.

---
*Certified by Antigravity AI Research Verification Harness — November 13, 2026.*

