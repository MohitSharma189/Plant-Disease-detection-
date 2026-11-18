# PHASE 7.1 – COMPLETE PLANTDOC DATASET FUNNEL RECONCILIATION REPORT

**Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma, Aniket Kumar, Raghav Kumar  
**Date:** November 14, 2026  
**Auditor:** Antigravity Research Verification System  
**Artifact Verified:** `reports/generated/combined_split_manifest.json` (SHA-256: `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868`)  
**Status:** PASS – FULLY RECONCILED (ZERO DISCREPANCIES)  

---

## 1. Executive Summary

Phase 7.1 independently investigated the numerical gap between the **2,342 audited source classification images** and the **2,159 selected training images** ($2{,}342 - 2{,}159 = 183$ images). 

Every single one of the **183 excluded records** was verified in the primary cryptographic manifest `reports/generated/combined_split_manifest.json` under `excluded_plantdoc_records`. The dataset ingestion funnel is completely accounted for by exact arithmetic:

$$\text{Audited Images (2,342)} - \text{Corrupt/Missing (117)} - \text{Train Duplicates (53)} - \text{Test Exact Duplicates (9)} - \text{Test Near Duplicates (4)} = \mathbf{2,159}$$

---

## 2. Complete Step-by-Step Dataset Reconciliation Table

| Funnel Stage | Image Count | Delta ($\Delta$) | Exclusion Category & Operational Reason | Primary Source Artifact |
| :--- | :---: | :---: | :--- | :--- |
| **Stage 1: Raw GitHub Repository Download** | **2,598** | — | Initial cloned PlantDoc classification repository (Singh et al., commit `5467f60`). | `reports/phase1/plantdoc_dataset_location_report.md` |
| **Stage 2: File Readability & PIL Verification** | **2,342** | $-256$ | 256 unreadable, empty (0-byte), or corrupt images discarded during initial filesystem scan. | `reports/phase1/plantdoc_dataset_location_report.md` |
| **Stage 3: Training Manifest Deduplication** | | | | |
| $\bullet$ Corrupt or Missing Records | — | $-117$ | Image paths that failed secondary hash resolution or had truncated byte streams (`plantdoc_corrupt_or_missing`). | `combined_split_manifest.json` (117 records) |
| $\bullet$ Intra-Training Redundancy | — | $-53$ | Exact duplicate image copies within the training partition (`plantdoc_duplicate_within_training`). | `combined_split_manifest.json` (53 records) |
| $\bullet$ Exact Test Set Duplicates | — | $-9$ | Images in training partition sharing identical SHA-256 hashes with images in the official 236-image test split (`plantdoc_duplicate_of_plantdoc_test`). | `combined_split_manifest.json` (9 records) |
| $\bullet$ Near-Duplicate Test Leakage | — | $-4$ | Images in training partition identified via perceptual hashing as near-duplicates ($d_H \le 4$) of test images (`plantdoc_near_duplicate_of_plantdoc_test`). | `combined_split_manifest.json` (4 records) |
| **Total Stage 3 Exclusions** | — | $\mathbf{-183}$ | Sum of all documented training exclusions: $117 + 53 + 9 + 4 = 183$. | `combined_split_manifest.json` (`excluded_plantdoc_records`) |
| **Stage 4: Clean PlantDoc Training Set** | **2,159** | — | Exactly $2{,}342 - 183 = 2{,}159$ clean, non-leaking training images incorporated into training split (seed 42). | `combined_split_manifest.json` (`source_counts.plantdoc_train_added`) |
| **Stage 5: Combined Training Split** | **40,151** | $+37,992$ | PlantVillage training slice ($37,992$) + clean PlantDoc training slice ($2,159$). | `combined_split_manifest.json` (`source_counts.plantvillage.train`) |
| **Stage 6: Disjoint Evaluation Partitions** | | | | |
| $\bullet$ PlantVillage Validation Slice | 5,436 | — | Disjoint holdout slice for model selection and early stopping. | `combined_split_manifest.json` (`source_counts.plantvillage.validation`) |
| $\bullet$ PlantVillage Calibration Slice | 5,429 | — | Disjoint holdout slice for temperature scaling parameter optimization. | `combined_split_manifest.json` (`source_counts.plantvillage.calibration`) |
| $\bullet$ PlantVillage Holdout Test Slice | 5,417 | — | Permanently reserved holdout slice for in-domain testing. | `combined_split_manifest.json` (`source_counts.plantvillage.test`) |
| **Total Ingested Manifest Records** | **56,433** | — | Sum: $40{,}151 + 5{,}436 + 5{,}429 + 5{,}417 = \mathbf{56,433}$ | `combined_split_manifest.json` (SHA-256: `f9ef8b9d...15868`) |
| **Stage 7: Official External Test Benchmark** | **236** | — | 236 official PlantDoc test images containing 452 whole-leaf ground truth bounding box annotations. Kept strictly disjoint. | `reports/generated/candidate_screening_manifest.json` |

---

## 3. Detailed Audit of the 183 Stage 3 Exclusions

The 183 exclusions recorded in `combined_split_manifest.json` are structured as follows:

### 3.1 Test Leakage Exclusions (13 Images Total)
To guarantee 100% data isolation between training and testing, two independent audits were executed:
1. **Exact SHA-256 Test Duplicates (9 records):**
   - Example: `train/Blueberry leaf/blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg` matched `test/Blueberry leaf/blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg`.
   - All 9 instances were purged from the training manifest.
2. **Perceptual Near-Duplicate Test Pairs (4 records):**
   - Example: `train/Tomato Early blight leaf/earlyblightpotato.jpg` matched `test/Potato leaf early blight/earlyblightpotato.jpg`.
   - Excluded to eliminate visual cross-split leakage.

### 3.2 Intra-Training Duplicate Exclusions (53 Images Total)
- Re-scanned using difference hashing ($d_H = 0$). 53 redundant image copies within the `train/` folders were discarded to avoid overfitting gradient updates on duplicated field images.

### 3.3 Corrupt or Missing Records (117 Images Total)
- Files with malformed EXIF tags, truncated JPEG blocks, or missing source links in the raw tree (e.g., `train/Apple Scab Leaf/6A642AA3-1DB1-4413-B0B6-7FBB68A5904A_original.jpeg?1528657579.jpg`).

---

## 4. Arithmetic Consistency Verification

$$\begin{aligned}
\text{Raw Repository Images} &= 2{,}598 \\
- \text{Corrupt / Unreadable Initial Scan} &= 256 \\
\hline
\text{Audited Readable Pool} &= 2{,}342 \\
- \text{Stage 3 Corrupt/Missing Records} &= 117 \\
- \text{Stage 3 Intra-Training Duplicates} &= 53 \\
- \text{Stage 3 Exact Test Duplicates} &= 9 \\
- \text{Stage 3 Near-Duplicate Test Leakage} &= 4 \\
\hline
\mathbf{Final\ Clean\ PlantDoc\ Training\ Set} &= \mathbf{2,159}
\end{aligned}$$

$$\mathbf{2,342 - (117 + 53 + 9 + 4) = 2,342 - 183 = 2,159}$$

The arithmetic is bit-for-bit exact and verifiable directly in `reports/generated/combined_split_manifest.json`.

