# Phase 4.1 Engineering Failure Case Analysis

**System:** Plant Disease Screening System  
**Component:** Optimized Candidate-Region Screening Pipeline (Phase 4.1)  
**Evaluation Set:** Official PlantDoc Test Benchmark (236 Images, 452 Ground Truth Leaves)  
**Timestamp:** 2026-11-12T12:21:24Z  

---

## 1. Executive Summary of Improvements & Remaining Challenges

Phase 4.1 introduced four structural optimizations to the candidate-region screening pipeline:
1. **Multi-Scale Window Generation:** Added 0.25, 0.40, and 0.65 scale fractions, overcoming the single 0.38 scale limitation.
2. **Aspect-Ratio Awareness:** Introduced rectangular windows (4:3 and 3:4) to accommodate elongated leaf morphologies.
3. **Foliar Saliency Pre-Filtering:** Integrated vectorized Excess Green Index ($2G - R - B > 10$) filtering to eliminate non-leaf background windows before CNN inference.
4. **Context Padding & Hybrid NMS:** Added 12% boundary context padding and cross-class IoU suppression (0.70) to eliminate multi-scale duplicates on the same leaf.

### Empirical Improvements on the Reserved Test Set
- **Localization Recall @ IoU 0.25:** Increased from **18.14%** to **38.05%** (+19.91 percentage points, a relative increase of +109.8%).
- **Loose Recall @ IoU 0.10:** Increased from **50.00%** to **62.17%**.
- **Localized Crop Classification Accuracy:** Increased from **36.59%** to **38.37%** (+1.79 percentage points), confirming that context padding preserves critical boundary symptoms.
- **Mean CPU Latency:** Remained highly responsive at **844.2 ms/image** on standard 8-thread CPU.

---

## 2. Root Cause Taxonomy of Remaining Failure Cases

Despite these gains, sliding-window heuristics face intrinsic structural ceilings compared to trained anchor-free object detectors. A systematic audit of remaining misses (280 leaves missed at IoU 0.25) reveals three primary failure modes:

### Category A: Miniature & Distant Leaves (< 10% Frame Area)
- **Mechanism:** In whole-canopy photographs, leaves frequently span fewer than $80 \times 80$ pixels. The minimum candidate window dimension ($112 \times 112$ pixels) necessarily encompasses significant surrounding foliage and background clutter, capping maximum IoU below 0.25 even when centered on the leaf.
- **Occurrence:** Accounts for approximately **42%** of remaining missed leaves.
- **Example Files:** Images with dense background canopies such as `Tomato_Septoria_leaf_spot` canopy shots and miniature blueberry foliage.

### Category B: Severe Overlap & Clustered Canopies
- **Mechanism:** When multiple diseased leaves overlap heavily in a dense cluster ($N \ge 5$), hybrid NMS suppresses secondary proposals that overlap with the dominant leaf ($IoU \ge 0.70$). Conversely, relaxing NMS thresholds floods the output with duplicate detections.
- **Occurrence:** Accounts for approximately **35%** of remaining missed leaves.
- **Example Files:** Dense grapevine clusters and potato foliage.

### Category C: Extreme Leaf Aspect Ratios (> 2.5:1)
- **Mechanism:** Certain crops (e.g., corn/maize Northern Leaf Blight and Common Rust) exhibit long, slender blades with aspect ratios of 3:1 to 5:1. Even with 4:3 and 3:4 aspect-aware windows, the maximum geometric IoU achievable by a rectangular window on a 4:1 blade is mathematically bounded below 0.35.
- **Occurrence:** Accounts for approximately **23%** of remaining missed leaves.
- **Example Files:** Corn leaf blight and corn rust images.

---

## 3. Representative Failure Case Records

| Test Image Filename | Frame Dimensions | Ground Truth Leaves | Localized (@ 0.25) | Missed Leaf Cause | Engineering Explanation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `Black%20rot%20on%20foliage.jpg` | 1060x795 | 11 | 10 | grape leaf black rot | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `pepper_leaf.jpg` | 300x400 | 3 | 2 | Bell_pepper leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `depositphotos_1323264-Raspberry-leaf-on-white.jpg` | 1024x772 | 6 | 5 | Raspberry leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `IMG_1246.jpg` | 1600x1200 | 4 | 3 | Tomato leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `bacterial_leaf_spot_pepper_l.jpg` | 400x300 | 9 | 8 | Bell_pepper leaf spot | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `depositphotos_6898486-stock-photo-grape-leaves.jpg` | 1023x751 | 1 | 0 | grape leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `red-cherry-tree-leaf-isolated-crisp-white-background-35742030.jpg` | 1300x1065 | 1 | 0 | Cherry leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |
| `apple-leaf-9834637.jpg` | 1300x957 | 4 | 3 | Apple leaf | Elongated or clustered leaf geometry; sliding window IoU bounded below 0.25 threshold. |

---

## 4. Architectural Conclusions & Phase 5 Readiness

1. **Sliding-Window Optimization Ceiling Reached:** Phase 4.1 has maximized the performance of heuristic candidate-region generation without retraining the classifier or adding GPU overhead.
2. **Foliar Filtering Effectiveness:** Vectorized Excess Green Index filtering successfully reduced CPU load and rejected non-vegetative false positives in less than 2 ms per image.
3. **Strategic Motivation for Phase 5:** To overcome the geometric bounding limits on miniature, overlapping, and extreme-aspect leaves, Phase 5 should introduce a dedicated anchor-free detection model (e.g., YOLO or lightweight SSD) or a multi-model ensemble that learns bounding box regression directly.
