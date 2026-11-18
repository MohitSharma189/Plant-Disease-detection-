# PlantDoc Object Detection Dataset Integrity & Recovery Report (Phase 1 Part B)

**Execution Date:** 2026-11-12  
**Dataset Sources:**  
- Extracted Dataset: `E:\New folder`  
- Archive: `E:\PlantDoc-Object-Detection-Dataset-master.zip` (994,774,747 bytes / ~948.7 MB)  
- Recovered Images: `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\data\processed\PlantDoc_Object_Detection`  
- Filename Resolution Mapping: `Plant Disease Detection/data/processed/PlantDoc_Object_Detection/filename_mapping.json`  
- Screening Benchmark Manifest: `Plant Disease Detection/reports/generated/candidate_screening_manifest.json`  

---

## 1. Executive Summary & Recovery Audit

An exhaustive audit of `train_labels.csv` (8,469 annotations across 2,345 unique filenames) and `test_labels.csv` (452 annotations across 236 unique filenames) was conducted against the extracted directory and the original ZIP archive:

1. **Test Split Integrity (100% Intact):**
   - **All 236 images (100.0%)** referenced in `test_labels.csv` resolve to valid images on disk.
   - **All 452 bounding boxes** are valid ($xmin < xmax, ymin < ymax$, zero out-of-bounds).
   - Zero test images are missing, corrupt, or zero-byte.
2. **Training Split Recovery:**
   - Out of 2,345 unique training images in `train_labels.csv`, **2,342 images (99.87%)** are fully resolved.
   - **22 missing images** that failed standard unzipping due to double URL-encoding (`%2523`, `%253E`) were extracted directly from `PlantDoc-Object-Detection-Dataset-master.zip` into `data/processed/PlantDoc_Object_Detection/TRAIN`.
   - **1 image** (`powdery-mildew-on-squash-leaves.jpg?w=300&h=200`) had URL query parameters stripped in the CSV.
   - **Only 2 images** (`NCLB.jpg` [2 annotations] and `early-blight-1.jpg` [6 annotations], totaling 8 / 8,469 annotations or 0.09%) are completely absent from the upstream published archive.
3. **Zero-Dimension Annotations Resolved:**
   - 4 images with `width=0, height=0` in `train_labels.csv` were located on disk, inspected, and their true pixel dimensions measured.
4. **Candidate-Screening Benchmark Manifest Generated:**
   - Created `candidate_screening_manifest.json` containing 236 test images and 452 leaf bounding boxes to evaluate whole-plant screening (Requirement 4) without training an object detector.

---

## 2. Image Recovery and Filename-Resolution Mapping

### A. Resolution Strategy
Rather than modifying the original `train_labels.csv` and `test_labels.csv` files, a non-destructive filename-resolution mapping was generated at [`data/processed/PlantDoc_Object_Detection/filename_mapping.json`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/data/processed/PlantDoc_Object_Detection/filename_mapping.json).

### B. Recovery Summary Table

| Partition | Images in CSV | Present on Disk (`E:\New folder`) | Recovered from ZIP | Truly Missing from Archive | Resolution Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TEST** | **236** | 236 | 0 (Already Complete) | 0 | **100.0% (236 / 236)** |
| **TRAIN** | **2,345** | 2,320 | 22 | 2 | **99.91% (2,342 / 2,345)** |
| **Total** | **2,581** | **2,556** | **22** | **2** | **99.92% (2,578 / 2,581)** |

### C. Details of Recovered Images (Sample)
The 22 images recovered from `PlantDoc-Object-Detection-Dataset-master.zip` into `data/processed/PlantDoc_Object_Detection/TRAIN` include:
- `SOYBEANS_2_1500x1500%253E.JPG_1474027789.jpg` (double-encoded `%253E`)
- `IMG_3166_1500x1500%253E.JPG_1473520197.jpg` (double-encoded `%253E`)
- `image_1500x1500%253E.jpeg_1473988930.jpg` (double-encoded `%253E`)
- `image_300x300%2523.jpeg_1472603885.jpg` (double-encoded `%2523`)
- `black%20leaf%20mold%20%284%29.JPG_itok=DIIUw9JU.jpg` (sanitized query `?itok=`)

---

## 3. Annotation Geometry & Boundary Audit

### A. Resolution of the 4 Zero-Dimension Annotations
In `train_labels.csv`, 4 image records listed `width=0, height=0`. The actual images on disk were opened and validated using PIL:

| Filename | CSV Width / Height | Actual Image Size (W $\times$ H) | Bounding Box ($xmin, ymin, xmax, ymax$) | Class |
| :--- | :---: | :---: | :---: | :--- |
| `2f73110f80014a25a53f9551c94bf164.png.jpg` | (0, 0) | **$525 \times 395$** | `[58, 92, 525, 386]` | `Corn Gray leaf spot` |
| `SouthernRustLeaf.png.jpg` | (0, 0) | **$666 \times 443$** | `[1, 142, 664, 349]` | `Corn rust leaf` |
| `ppth-friskop-1-corn-rust.png.jpg` | (0, 0) | **$578 \times 224$** | `[78, 1, 566, 192]` | `Corn rust leaf` |
| `southernrust1.png.jpg` | (0, 0) | **$471 \times 628$** | `[55, 1, 430, 628]` | `Corn rust leaf` |

All four images exist, are fully readable, and their bounding boxes fit cleanly within their actual physical dimensions.

### B. Coordinate Validity & Boundary Analysis

| Check | TRAIN Partition (8,469 boxes) | TEST Partition (452 boxes) |
| :--- | :---: | :---: |
| **Invalid Coordinates ($xmin \ge xmax$ or $ymin \ge ymax$)** | **0** (0.0%) | **0** (0.0%) |
| **Out-of-Bounds Boxes ($xmax > W$ or $ymax > H$)** | 21 (0.25%) | **0** (0.0%) |
| **Dimension Mismatches (CSV Dim $\neq$ File Dim)** | 41 (0.48%) | **0** (0.0%) |
| **Minimum Box Area %** | 0.00% | 0.66% |
| **Mean Box Area %** | **14.92%** | **30.64%** |
| **Maximum Box Area %** | **99.91%** | **99.08%** |

- **Root Cause of Out-of-Bounds in TRAIN:** 21 boxes in TRAIN result from manual annotation on transposed/rotated image versions (e.g., `IMG_1526.jpg` where CSV records $2448 \times 3264$ but image is stored as $3264 \times 2448$, or `Gray-leaf-spot.jpg` where dimensions were recorded as $508 \times 337$ but image is $205 \times 300$).
- **Test Set Validity:** The official TEST partition is **100% geometrically clean**, with 0 out-of-bounds boxes and 0 dimension discrepancies.

---

## 4. Fundamental Nature of Object Detection Annotations

> [!IMPORTANT]
> **Whole Leaves, Not Lesion Boundaries (Requirement 4 Ground Truth):**
> Geometric and visual inspection confirms that **annotations represent WHOLE LEAVES**:
> - Mean box area is **14.92%** in TRAIN and **30.64%** in TEST.
> - Maximum box area reaches **99.08%** in TEST and **99.91%** in TRAIN.
> - Multiple classes explicitly represent entire healthy leaves (e.g., `Apple leaf`, `Bell_pepper leaf`, `Cherry leaf`, `Potato leaf`, `Strawberry leaf`).
> 
> **Scientific Protocol for Requirement 4:**
> 1. This dataset evaluates **candidate-region leaf localization**, candidate window extraction, and whole-plant screening.
> 2. It **does not and cannot** evaluate lesion segmentation, affected-area masks, or pixel-level disease severity.

---

## 5. Candidate Screening Benchmark Manifest

The verified screening evaluation manifest has been generated at:
[`Plant Disease Detection/reports/generated/candidate_screening_manifest.json`](file:///e:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/generated/candidate_screening_manifest.json)

- **Total Held-Out Images:** 236
- **Total Validated Leaf Boxes:** 452
- **Classes Represented:** 27
- **Format:** Standard JSON array with resolved file paths, actual image dimensions, bounding box coordinates, and area percentages.
- **Usage in Phase 2:** Will be used as the ground-truth benchmark to validate candidate proposal recall, candidate window extraction, and NMS suppression for whole-plant processing.

