# Dataset Integrity & Path Resolution Audit Report (Phase 1)

**Execution Date:** 2026-11-12  
**Target Datasets:** PlantVillage, PlantDoc Classification Dataset, PlantDoc Object Detection Dataset  
**Audit Verification:** Verified 54,305 PlantVillage images + 2,578 PlantDoc classification images + 8,921 PlantDoc OD annotations  

---

## 1. PlantVillage Dataset Status (Restored & Verified)

### Local Filesystem State
- **Directory:** `Plant Disease Detection/data/processed/PlantVillage`
- **Class Folders Present:** 38 / 38 classes (100% complete)
- **Total Local Images:** **54,305 images**
- **Image Readability & Integrity:** **100% verified** (54,305 / 54,305 successfully validated via `PIL.Image.verify()`)
- **Corrupted / Truncated Images:** **0**
- **Zero-Byte Images:** **0**
- **Initial Backup Location:** `Plant Disease Detection/data/processed/PlantVillage_backup_initial` (preserves initial 90-image subset)
- **Archive Source:** Hugging Face Hub `mohanty/PlantVillage` (`data.zip`, 2.03 GB, 2,184,723,441 bytes) cached locally in `data/raw/PlantVillage_data.zip`

### Complete 38-Class Distribution

| # | PlantVillage Class Folder Name | Image Count | Status |
| :-: | :--- | :---: | :---: |
| 1 | `Apple___Apple_scab` | 630 | Verified Valid |
| 2 | `Apple___Black_rot` | 621 | Verified Valid |
| 3 | `Apple___Cedar_apple_rust` | 275 | Verified Valid |
| 4 | `Apple___healthy` | 1,645 | Verified Valid |
| 5 | `Blueberry___healthy` | 1,502 | Verified Valid |
| 6 | `Cherry_(including_sour)___Powdery_mildew` | 1,052 | Verified Valid |
| 7 | `Cherry_(including_sour)___healthy` | 854 | Verified Valid |
| 8 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 513 | Verified Valid |
| 9 | `Corn_(maize)___Common_rust_` | 1,192 | Verified Valid |
| 10 | `Corn_(maize)___Northern_Leaf_Blight` | 985 | Verified Valid |
| 11 | `Corn_(maize)___healthy` | 1,162 | Verified Valid |
| 12 | `Grape___Black_rot` | 1,180 | Verified Valid |
| 13 | `Grape___Esca_(Black_Measles)` | 1,383 | Verified Valid |
| 14 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1,076 | Verified Valid |
| 15 | `Grape___healthy` | 423 | Verified Valid |
| 16 | `Orange___Haunglongbing_(Citrus_greening)` | 5,507 | Verified Valid |
| 17 | `Peach___Bacterial_spot` | 2,297 | Verified Valid |
| 18 | `Peach___healthy` | 360 | Verified Valid |
| 19 | `Pepper,_bell___Bacterial_spot` | 997 | Verified Valid |
| 20 | `Pepper,_bell___healthy` | 1,478 | Verified Valid |
| 21 | `Potato___Early_blight` | 1,000 | Verified Valid |
| 22 | `Potato___Late_blight` | 1,000 | Verified Valid |
| 23 | `Potato___healthy` | 152 | Verified Valid |
| 24 | `Raspberry___healthy` | 371 | Verified Valid |
| 25 | `Soybean___healthy` | 5,090 | Verified Valid |
| 26 | `Squash___Powdery_mildew` | 1,835 | Verified Valid |
| 27 | `Strawberry___Leaf_scorch` | 1,109 | Verified Valid |
| 28 | `Strawberry___healthy` | 456 | Verified Valid |
| 29 | `Tomato___Bacterial_spot` | 2,127 | Verified Valid |
| 30 | `Tomato___Early_blight` | 1,000 | Verified Valid |
| 31 | `Tomato___Late_blight` | 1,909 | Verified Valid |
| 32 | `Tomato___Leaf_Mold` | 952 | Verified Valid |
| 33 | `Tomato___Septoria_leaf_spot` | 1,771 | Verified Valid |
| 34 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | Verified Valid |
| 35 | `Tomato___Target_Spot` | 1,404 | Verified Valid |
| 36 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | Verified Valid |
| 37 | `Tomato___Tomato_mosaic_virus` | 373 | Verified Valid |
| 38 | `Tomato___healthy` | 1,591 | Verified Valid |
| **Total** | **38 Classes** | **54,305** | **100% Readable** |

---

## 2. PlantDoc Classification Dataset Audit

### Overview
- **Location:** `data/raw/PlantDoc_dataset/` (source archive: `E:\PlantDoc-Dataset-master.zip`, extracted in `E:\New folder (2)`)
- **Manifest:** `source_manifest.json` (2,578 records)
- **Official Train Split:** 2,342 images across **28 classes**
- **Official Test Split:** 236 images across **27 classes**
- **Corrupted Images:** 0 corrupt files among resolved images.

### Class Discrepancy & Mapping Analysis
The 28 classes in the classification dataset map to 28 of the 38 PlantVillage classes via `training/plantdoc_label_mapping.json`.

1. **Unrepresented PlantVillage Classes (10 classes have 0 images in PlantDoc):**
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

2. **Split Asymmetry:**
   - `Tomato two spotted spider mites leaf` has 2 images in `train`, but **0 images in `test`**.

### Class-Wise Distribution (Manifest Counts)

| PlantDoc Class Name | Mapped PlantVillage Label | Train Count | Test Count | Total |
| :--- | :--- | :---: | :---: | :---: |
| Apple Scab Leaf | `Apple___Apple_scab` | 83 | 10 | 93 |
| Apple leaf | `Apple___healthy` | 82 | 9 | 91 |
| Apple rust leaf | `Apple___Cedar_apple_rust` | 79 | 10 | 89 |
| Bell_pepper leaf | `Pepper,_bell___healthy` | 53 | 8 | 61 |
| Bell_pepper leaf spot | `Pepper,_bell___Bacterial_spot` | 62 | 9 | 71 |
| Blueberry leaf | `Blueberry___healthy` | 106 | 11 | 117 |
| Cherry leaf | `Cherry_(including_sour)___healthy` | 47 | 10 | 57 |
| Corn Gray leaf spot | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 64 | 4 | 68 |
| Corn leaf blight | `Corn_(maize)___Northern_Leaf_Blight` | 180 | 12 | 192 |
| Corn rust leaf | `Corn_(maize)___Common_rust_` | 106 | 10 | 116 |
| Peach leaf | `Peach___healthy` | 103 | 9 | 112 |
| Potato leaf early blight | `Potato___Early_blight` | 109 | 8 | 117 |
| Potato leaf late blight | `Potato___Late_blight` | 97 | 8 | 105 |
| Raspberry leaf | `Raspberry___healthy` | 112 | 7 | 119 |
| Soyabean leaf | `Soybean___healthy` | 57 | 8 | 65 |
| Squash Powdery mildew leaf | `Squash___Powdery_mildew` | 124 | 6 | 130 |
| Strawberry leaf | `Strawberry___healthy` | 88 | 8 | 96 |
| Tomato Early blight leaf | `Tomato___Early_blight` | 79 | 9 | 88 |
| Tomato Septoria leaf spot | `Tomato___Septoria_leaf_spot` | 140 | 11 | 151 |
| Tomato leaf | `Tomato___healthy` | 55 | 8 | 63 |
| Tomato leaf bacterial spot | `Tomato___Bacterial_spot` | 101 | 9 | 110 |
| Tomato leaf late blight | `Tomato___Late_blight` | 101 | 10 | 111 |
| Tomato leaf mosaic virus | `Tomato___Tomato_mosaic_virus` | 44 | 10 | 54 |
| Tomato leaf yellow virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 70 | 6 | 76 |
| Tomato mold leaf | `Tomato___Leaf_Mold` | 85 | 6 | 91 |
| Tomato two spotted spider mites leaf | `Tomato___Spider_mites Two-spotted_spider_mite` | 2 | 0 | 2 |
| grape leaf | `Grape___healthy` | 57 | 12 | 69 |
| grape leaf black rot | `Grape___Black_rot` | 56 | 8 | 64 |
| **Total** | | **2,342** | **236** | **2,578** |

### Filename Resolution & Windows Sanitization Audit
In the previous inspection, 276 files failed direct path matching, including 15 test files.
- **Root Cause Identified:** The upstream Git repository contains URL-encoded characters (`%20`, `%23`, `%2B`, `%29`) and characters forbidden by Windows NTFS (`?`, `:`).
  - During checkout, `fetch_plantdoc.py` converted `%` and `?` into `_`.
  - In `source_manifest.json`, the un-sanitized source paths were recorded.
- **Resolution Verification:** By applying the unified regex sanitization rule:
  `re.compile(r'[<>:"/\\|?*%]').sub('_', filename).rstrip(' .')`
  - **Test Split Resolution:** **236 / 236 images (100.0%) successfully resolve**.
  - **0 test images are dropped or corrupted.**
  - **Train Split Resolution:** 2,226 / 2,342 images resolve in `data/raw/PlantDoc_dataset/train`. The remaining 116 images exist in `E:\New folder (2)` and can be synchronized cleanly.

---

## 3. PlantDoc Object Detection Dataset Audit

### Overview
- **Location:** `E:\New folder/` (source archive: `E:\PlantDoc-Object-Detection-Dataset-master.zip`)
- **Annotation Files:** `train_labels.csv`, `test_labels.csv`
- **Image Folders:** `TRAIN`, `TEST`

### Statistical Breakdown

| Split | Annotations Count | Unique Images in CSV | Classes Present | Missing Images from Disk | Out-of-Bounds Boxes | Dimension Mismatches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 8,469 | 2,345 | 29 | 85 | 19 | 43 |
| **TEST** | 452 | 236 | 27 | 5 | 0 | 0 |

### Classes in Object Detection
`train_labels.csv` includes 29 classes: the 28 classification classes plus **`Potato leaf`** (11 bounding boxes, corresponding to `Potato___healthy`). `test_labels.csv` contains 27 classes (lacking `Potato leaf` and `Tomato two spotted spider mites leaf`).

### Annotation Anomalies Identified
1. **Missing Files on Windows:** 85 images in `train_labels.csv` and 5 images in `test_labels.csv` are absent from disk because their filenames contain `?` (e.g. `1421_0.jpeg?itok=FMtmgePj.jpg`), which standard Windows extraction omits.
2. **Zero Image Dimensions:** 4 images in `train_labels.csv` record `width=0, height=0`:
   - `2f73110f80014a25a53f9551c94bf164.png.jpg`
   - `SouthernRustLeaf.png.jpg`
   - `ppth-friskop-1-corn-rust.png.jpg`
   - `southernrust1.png.jpg`
3. **Out-of-Bounds Coordinates:** 19 bounding boxes in `train_labels.csv` exceed the actual image width/height by 1–15 pixels due to rounding errors during manual annotation.

---

## 4. Fundamental Annotation Boundary Limitation

> [!IMPORTANT]
> **Whole-Leaf vs. Lesion Bounding Boxes:**
> Detailed visual and geometric verification of `train_labels.csv` confirms that **bounding boxes encompass WHOLE LEAVES, NOT disease lesions or necrotic spots.**
> - Mean box area: **14.94%** of the full image.
> - Maximum box area: **99.91%** of the full image.
> - Class labels explicitly specify leaves (e.g. `Apple Scab Leaf`, `Corn leaf blight`, `Tomato leaf late blight`).
> 
> **Scientific Implication:**
> The PlantDoc object detection annotations **validate candidate leaf localization and whole-plant window screening (Requirement 4)**. They **cannot and must not** be used to benchmark pixel-level lesion segmentation or affected-area percentage estimation.

