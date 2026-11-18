# PlantDoc Dataset Location & Local Discovery Audit Report (Phase 1)

**Execution Date:** 2026-11-12  
**Target Environment:** Local Windows Workstation (`E:\` Drive)  
**Investigator:** Antigravity Pairing Assistant  

---

## 1. Executive Summary & Discovery Status

A comprehensive filesystem scan across the local computer confirms that **both PlantDoc datasets are already present locally on the `E:\` drive**, in **both original compressed archive formats and fully extracted directories**.

- **No downloads from external networks are required.**
- **No file uploads from the user are required.**
- **No duplicate copies need to be created in the repository.**
- Both datasets are directly accessible to Python scripts running in the project virtual environment.

---

## 2. Dataset Locations and Accessibility Matrix

| Dataset | State | Absolute Filesystem Location | Size / Item Count | Accessible to Project? |
| :--- | :--- | :--- | :--- | :---: |
| **PlantDoc Classification** | **Extracted Folder** | `E:\New folder (2)` | 2,572 files across `train/` and `test/` | **Yes (Direct Read)** |
| **PlantDoc Classification** | **ZIP Archive** | `E:\PlantDoc-Dataset-master.zip` | 984,413,268 bytes (~938.8 MB) | **Yes (Verified Valid)** |
| **PlantDoc Classification** | **Staged Directory** | `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\data\raw\PlantDoc_dataset` | 2,461 files + `source_manifest.json` | **Yes (Local Working Copy)** |
| **PlantDoc Object Detection** | **Extracted Folder** | `E:\New folder` | `TRAIN/` (4,700 files), `TEST/` (476 files), 2 CSVs | **Yes (Direct Read)** |
| **PlantDoc Object Detection** | **ZIP Archive** | `E:\PlantDoc-Object-Detection-Dataset-master.zip` | 994,774,747 bytes (~948.7 MB) | **Yes (Verified Valid)** |

---

## 3. PlantDoc Classification Dataset Inspection

### A. Directory Structure & Partitions
- **Extracted Root:** `E:\New folder (2)`
- **Partitions:**
  - `E:\New folder (2)\train` (28 class subdirectories, 2,336 images on disk)
  - `E:\New folder (2)\test` (27 class subdirectories, 236 images on disk)
  - Documentation: `LICENSE.txt`, `README.md`, `PlantDoc_Examples.png`
- **Archive Contents (`PlantDoc-Dataset-master.zip`):**
  - Total entries: 2,581 (2,342 train images, 236 test images)

### B. Class Compatibility & 28-Class Mapping
The classification dataset contains 28 disease and healthy classes. All 28 map directly to the project's 38 PlantVillage classes via [`training/plantdoc_label_mapping.json`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/training/plantdoc_label_mapping.json):

1. **Split Asymmetry in Upstream Dataset:**
   - Class `Tomato two spotted spider mites leaf`: 2 images in `train`, **0 images in `test`**.
   - As a result, `test` contains 27 classes while `train` contains 28.
2. **Unrepresented PlantVillage Classes (10 classes have 0 samples in PlantDoc):**
   - `Apple___Black_rot`
   - `Cherry_(including_sour)___Powdery_mildew`
   - `Corn_(maize)___healthy`
   - `Grape___Esca_(Black_Measles)`
   - `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
   - `Orange___Haunglongbing_(Citrus_greening)`
   - `Peach___Bacterial_spot`
   - `Potato___healthy`
   - `Strawberry___Leaf_scorch`
   - `Tomato___Target_Spot`

### C. Filename Resolution & Windows NTFS Compatibility
- **Observation:** Upstream web-scraped filenames contain URL encodings (`%20`, `%23`, `%2B`, `%29`, `%253E`) and NTFS-illegal characters (`?`, `:`).
- **Sanitization Rule:** When resolving filenames using:
  ```python
  re.sub(r'[<>:"/\\|?*%]', '_', filename).rstrip(' .')
  ```
  - **100.0% of the 236 official test images** (`236 / 236`) resolve successfully.
  - Zero test images are dropped, corrupted, or missing.
- **Test Set Integrity:** The 236 official test images are held out as an untouched benchmark for external evaluation (Requirement 2).

---

## 4. PlantDoc Object Detection Dataset Inspection

### A. Directory Structure & Files
- **Extracted Root:** `E:\New folder`
- **Image Directories:**
  - `E:\New folder\TRAIN` (4,700 images on disk)
  - `E:\New folder\TEST` (476 images on disk)
- **Annotation Files:**
  - `E:\New folder\train_labels.csv` (663,331 bytes, 8,469 annotations across 2,345 unique images)
  - `E:\New folder\test_labels.csv` (33,788 bytes, 452 annotations across 236 unique images)

### B. Annotation Integrity & Geometry Audit
- **Classes:**
  - `train_labels.csv`: 29 classes (the 28 classification classes plus `Potato leaf` [11 annotations]).
  - `test_labels.csv`: 27 classes (lacks `Potato leaf` and `Tomato two spotted spider mites leaf`).
- **Zero-Dimension Anomalies:**
  - 4 image entries in `train_labels.csv` have `width=0, height=0`:
    1. `southernrust1.png.jpg`
    2. `ppth-friskop-1-corn-rust.png.jpg`
    3. `2f73110f80014a25a53f9551c94bf164.png.jpg`
    4. `SouthernRustLeaf.png.jpg`
- **Missing Filename Anomalies on Disk:**
  - 25 filenames in `train_labels.csv` failed standard unzipping due to double URL-encoding (e.g. `%2523`, `%253E`, `?itok=...`).
  - In `test_labels.csv`, **0 images are missing** from disk once sanitized (`236 / 236` resolved).
- **Out-of-Bounds Coordinates:**
  - 19 bounding boxes in `train_labels.csv` exceed image dimensions by 1–15 pixels due to coordinate rounding in manual labeling.

### C. Fundamental Nature of Object Detection Bounding Boxes
> [!IMPORTANT]
> **Whole-Leaf Bounding Boxes (Requirement 4 Validation):**
> Empirical geometric verification confirms that **bounding boxes annotate WHOLE LEAVES**, not disease lesion boundaries or affected areas:
> - **TRAIN mean box area:** **14.92%** of full image (max: **99.91%**).
> - **TEST mean box area:** **30.08%** of full image (max: **99.08%**).
> 
> **Scientific Conclusion:** These annotations provide rigorous ground truth for **candidate leaf screening and whole-plant window extraction (Requirement 4)**. They cannot be used as disease-lesion segmentation ground truth.

---

## 5. Duplicate Leakage Audit Between Datasets

A cryptographic and perceptual hash audit between train and test partitions revealed significant upstream leakage in PlantDoc:

### A. 10 Exact Duplicate Images Across Train and Test
10 images exist in both the training set and the official test set with identical SHA-256 byte hashes.
- **6 pairs carry severe contradictory disease labels:**
  - `2015070295153021.jpg`: `Corn Gray leaf spot` (Train) vs `Corn leaf blight` (Test)
  - `corn-gray-leaf-spot-f4.jpg`: `Corn Gray leaf spot` (Train) vs `Corn leaf blight` (Test)
  - `5816740026_...`: `Potato leaf early blight` (Train) vs `Potato leaf late blight` (Test)
  - `1421_0.jpeg...`: `Potato leaf early blight` (Train) vs `Potato leaf late blight` (Test)
  - `irish-blight-...`: `Potato leaf early blight` (Train) vs `Potato leaf late blight` (Test)
  - `tomato_V8.jpg`: `Tomato Septoria leaf spot` (Train) vs `Tomato leaf bacterial spot` (Test)
- **4 pairs are identical leakage:**
  - `blueberry-leaves-...`, `early-blight-septoria-...`, `tylcv-seminar-...`, `backus-056-potato-blight.jpg` / `IMG_42231.jpg`

### B. 4 Near-Duplicate Pairs (dHash Distance = 0)
- `train/Tomato Early blight leaf/earlyblightpotato.jpg` $\leftrightarrow$ `test/Potato leaf early blight/earlyblightpotato.jpg` (**Cross-Species Conflict: Tomato vs Potato**)
- `train/Tomato leaf late blight/tomato-late-blight...` $\leftrightarrow$ `test/Potato leaf late blight/581674...` (**Cross-Species Conflict: Tomato vs Potato**)
- `train/Tomato leaf bacterial spot/bacterial-canker...` $\leftrightarrow$ `test/Tomato leaf bacterial spot/07.17.18...` (Recompressed crop)
- `train/Tomato leaf late blight/late_blight1.jpg` $\leftrightarrow$ `test/Tomato leaf late blight/18c.jpg` (Recompressed crop)

### C. Exclusion Safeguard
As implemented in [`training/prepare_combined_manifest.py`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/training/prepare_combined_manifest.py), all exact duplicate and near-duplicate images of the test set are automatically excluded from training manifests under `plantdoc_duplicate_of_plantdoc_test` and `plantdoc_near_duplicate_of_plantdoc_test`. The official test partition remains 100% held out.

---

## 6. Recommended Project Integration Strategy

1. **Zero Data Duplication:**
   Keep the original source archives and extracted folders at `E:\New folder (2)` and `E:\New folder` intact and unmodified.
2. **Path Resolution via Manifests:**
   Use the existing path normalization logic in `prepare_combined_manifest.py` to point directly to these datasets.
3. **External Evaluation Harness:**
   Execute independent evaluation (Requirement 2) on the 236 official test images in `E:\New folder (2)\test` or `PlantDoc_dataset/test`.
4. **Candidate-Region Screening (Requirement 4):**
   Evaluate whole-plant candidate region extraction against the 452 leaf bounding boxes in `E:\New folder\test_labels.csv`.

