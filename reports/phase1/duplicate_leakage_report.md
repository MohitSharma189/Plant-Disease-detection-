# Duplicate Detection & Data Leakage Audit Report (Phase 1)

**Execution Date:** 2026-11-12  
**Target Scripts Inspected:** `training/prepare_splits.py`, `training/prepare_combined_manifest.py`  
**Dataset Scopes:** PlantVillage, PlantDoc Classification, PlantDoc Object Detection

---

## 1. Deduplication Algorithm Verification

The project's deduplication mechanism in [`training/prepare_splits.py`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/training/prepare_splits.py) combines cryptographic and perceptual hashing:

1. **Exact Duplicate Detection:**
   - Evaluates 256-bit SHA-256 cryptographic digest on raw image bytes.
   - Identical digests with the same label: marked as `redundant_exact_duplicates` and pruned from the eligible training set.
   - Identical digests with conflicting labels: marked as `conflicting_groups` and completely excluded from all partitions.

2. **Near-Duplicate Detection:**
   - Evaluates 64-bit difference hash (`dHash`): image is converted to grayscale, downsampled to $9 \times 8$ using Lanczos interpolation, and horizontal pixel gradients are thresholded (`pixel[x] > pixel[x+1]`).
   - Clusters pairs with Hamming distance $\le 3$ bits using a 4-chunk 16-bit bucketed index and Union-Find disjoint sets.

3. **Grouped Stratified Partitioning:**
   - All images belonging to the same Union-Find root are assigned together to one partition (`train`, `validation`, `calibration`, or `test`), preventing identical or near-identical images from crossing split boundaries.

---

## 2. Upstream Data Leakage Discovered in PlantDoc (Classification & Object Detection)

An empirical hash audit between `train` and `test` across the PlantDoc dataset revealed substantial upstream leakage in the published benchmark.

### A. 11 Exact Duplicate Images Across Train and Test Partitions
The following 11 images appear in both the training partition and the official evaluation partition with identical SHA-256 digests:

| # | Filename | SHA-256 Digest | Class in Train Split | Class in Test Split | Conflict Status |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `2015070295153021.jpg` | `6e777d1b6829...` | `Corn Gray leaf spot` | `Corn leaf blight` | **Conflicting Label** |
| 2 | `5816740026_d42ef24413_Phytophthora-Infestans.jpg` | `a981ee59e5e1...` | `Potato leaf early blight` | `Potato leaf late blight` | **Conflicting Label** |
| 3 | `1421_0.jpeg?itok=FMtmgePj.jpg` | `c53ad8d40c38...` | `Potato leaf early blight` | `Potato leaf late blight` | **Conflicting Label** |
| 4 | `corn-gray-leaf-spot-f4.jpg` | `9120578a76fd...` | `Corn Gray leaf spot` | `Corn leaf blight` | **Conflicting Label** |
| 5 | `irish-blight-symptoms-on-potato-leaves-atmf8b.jpg` | `9dd93b421749...` | `Potato leaf early blight` | `Potato leaf late blight` | **Conflicting Label** |
| 6 | `tomato_V8.jpg` | `e23a29c94b58...` | `Tomato Septoria leaf spot` | `Tomato leaf bacterial spot` | **Conflicting Label** |
| 7 | `blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg` | `07b9b0a3395d...` | `Blueberry leaf` | `Blueberry leaf` | Identical Label (Leakage) |
| 8 | `early-blight-septoria-ls-fig-3.jpg` | `93c989d85512...` | `Tomato Septoria leaf spot` | `Tomato Septoria leaf spot` | Identical Label (Leakage) |
| 9 | `tylcv-seminar-1-638.jpg` | `3c52144e19e9...` | `Tomato leaf yellow virus` | `Tomato leaf yellow virus` | Identical Label (Leakage) |
| 10 | `backus-056-potato-blight.jpg` | `64c0f7b5701e...` | `Potato leaf late blight` | `Potato leaf late blight` | Identical Label (Leakage) |
| 11 | `IMG_42231.jpg` | `ee3838e2249e...` | `Tomato leaf` | `Tomato leaf` | Identical Label (Leakage) |

> [!CAUTION]
> **Severe Conflict Analysis:**
> Notice that **6 of the exact duplicate images carry contradictory disease labels** between train and test in the official benchmark!
> - For example, `2015070295153021.jpg` is labeled *Gray Leaf Spot* in train, but *Northern Leaf Blight* in test.
> - `5816740026_...` is labeled *Early Blight* in train, but *Late Blight* in test.
> - `tomato_V8.jpg` is labeled *Septoria Leaf Spot* in train, but *Bacterial Spot* in test.
> If these training images were included during training, a model would either be penalized on the test set for predicting what it was taught, or artificial data leakage would inflate test performance.

---

### B. 4 Near-Duplicate Image Pairs (dHash Distance = 0)

Four additional image pairs across train and test partitions exhibit identical perceptual difference hashes (`dHash` Hamming distance $= 0$):

| # | Train Image Path | Train Class | Test Image Path | Test Class | Conflict Status |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `train/Tomato Early blight leaf/earlyblightpotato.jpg` | `Tomato Early blight leaf` | `test/Potato leaf early blight/earlyblightpotato.jpg` | `Potato leaf early blight` | **Cross-Species Conflict** (Tomato vs. Potato) |
| 2 | `train/Tomato leaf late blight/tomato-late-blight-foliar-lesions_Scot-Nelson_CC-BY-SA-2.0.jpg` | `Tomato leaf late blight` | `test/Potato leaf late blight/5816740026_d42ef24413_Phytophthora-Infestans.jpg` | `Potato leaf late blight` | **Cross-Species Conflict** (Tomato vs. Potato) |
| 3 | `train/Tomato leaf bacterial spot/bacterial-canker3x2400-2ia1kgd.jpg` | `Tomato leaf bacterial spot` | `test/Tomato leaf bacterial spot/07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg` | `Tomato leaf bacterial spot` | Identical Label (Resized / Recompressed) |
| 4 | `train/Tomato leaf late blight/late_blight1.jpg` | `Tomato leaf late blight` | `test/Tomato leaf late blight/18c.jpg` | `Tomato leaf late blight` | Identical Label (Resized / Recompressed) |

---

## 3. Algorithmic Corrections Implemented

### A. Global Perceptual Near-Duplicate Grouping (`training/prepare_splits.py`)
- **Previous Limitation:** In line 70, `bucket_key = (record["label"], (chunk << 16) | bucket_value)` constrained near-duplicate checking to within the same label. Cross-class near-duplicates escaped detection.
- **Correction Applied:** Removed `record["label"]` from `bucket_key`:
  ```python
  bucket_key = (chunk << 16) | bucket_value
  ```
  Near-duplicate search now spans across all classes globally. When cross-label near-duplicates are joined by Union-Find, they are captured by `conflicting_groups` and safely excluded from all partitions without deleting source data.
- **Verification:** Passed `pytest training/test_prepare_splits.py`.

### B. PlantDoc Test Leakage Exclusion Filter (`training/prepare_combined_manifest.py`)
- **Previous Limitation:** `build_combined_manifest` only compared candidate PlantDoc train images against PlantVillage images and accepted PlantDoc train images; it did not cross-reference the official PlantDoc test partition.
- **Correction Applied:**
  1. Pre-indexes all records in `plantdoc_manifest` where `split == "test"`.
  2. Resolves Windows-sanitized paths (handling URL-encoded and sanitized characters).
  3. Caches exact SHA-256 digests and perceptual difference hash buckets for the test partition.
  4. Scans candidate PlantDoc train records:
     - If exact match to test partition: excluded under `plantdoc_duplicate_of_plantdoc_test`.
     - If near-duplicate match ($\le 3$ Hamming distance) to test partition: excluded under `plantdoc_near_duplicate_of_plantdoc_test`.
  5. The official PlantDoc test set remains completely held out and untouched.
- **Verification:** Unit tests added in `training/test_prepare_combined_manifest.py` verifying test-leakage exclusion; passed cleanly.

---

## 4. Leakage Risks in the Deployed Baseline Checkpoint

- Checkpoint: [`saved_models/classifier.pt`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/saved_models/classifier.pt)
- Checkpoint Metadata: [`saved_models/classifier_metadata.json`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/saved_models/classifier_metadata.json)
- **Risk:** The checkpoint predates the split manifest system. Its training set records are unverified.
- **Verification Rule:** As implemented in `evaluate_classifier.py` (`verifies_training_provenance`), baseline promotion is blocked unless both checkpoints share verifiable provenance over the same split manifest.
- **Protocol:** The current production checkpoint must remain classified as an **unverified legacy baseline**. It must not be cited as an independently validated benchmark.

