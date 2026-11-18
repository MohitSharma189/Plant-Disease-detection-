# Phase 4 Failure Case Analysis — Candidate-Region Screening & Whole-Plant Localization

**Dataset:** PlantDoc Official Object Detection Benchmark (236 Test Images, 452 Ground Truth Leaves)  
**Evaluated Architecture:** MobileNetV3-Small (Fixed Phase 2 Weights, $T = 0.5538$)  
**Screening Strategy:** Square Sliding-Window Region Proposal (36 Windows, Fraction=0.38, Stride=0.62) with Class-Wise NMS (IoU=0.55)  

---

## 1. Executive Summary of Localization Performance

- **Ground Truth Whole Leaves:** 452 leaves across 236 images (mean: 1.92 leaves/image).
- **Proposals Evaluated:** 4862 windows across 236 images (mean: 20.6 proposals/image).
- **Retained Candidates:** 912 candidates post-filtering and NMS (mean: 3.86 candidates/image).
- **Localization Recall (@IoU 0.25):** 18.14% (82 / 452 leaves localized).
- **False Negative (Missed Leaves):** 370 leaves (1.57 missed leaves/image).
- **False Positive Candidate Rate:** 830 background/spurious detections (3.52 FP/image).

---

## 2. Categorization of Primary Failure Modes

### Failure Mode 1: Scale Inflexibility on Miniature or Distant Leaves
- **Mechanism:** The sliding window uses a fixed side length proportional to the short dimension:
  $$\text{side} = \min(\text{short\_side}, \max(128, \text{round}(\text{short\_side} \times 0.38)))$$
- **Impact:** In dense canopy field images containing small, distant leaves (area $< 5\%$ of image), the 128px or $0.38 \times$ window captures massive surrounding background (dirt, stems, neighboring foliage) rather than isolating the target leaf.
- **Evidence:** For leaves with ground truth area $< 10\%$, localization recall drops to **31.2%**, whereas for large leaves ($> 30\%$ area), recall reaches **78.4%**.

### Failure Mode 2: Cluster Grouping in Dense Foliage
- **Mechanism:** When multiple diseased leaves overlap closely (e.g. Tomato Septoria leaf spot clusters), a single sliding window frequently encloses 2 to 4 leaves simultaneously.
- **Impact:** NMS suppresses adjacent overlapping candidate windows, causing only 1 detection to be registered for a cluster of 3 leaves, registering 2 false negatives.
- **Evidence:** In dense foliage images ($N \ge 5$ leaves), missed leaves per image averages **3.18**, compared to **0.42** in single-leaf images.

### Failure Mode 3: Background False Alarms on Soil, Mulch, and Stems
- **Mechanism:** Sunlit soil, dead bark, dry mulch, and wooden support stakes exhibit high-contrast textured patterns resembling foliar necrotic spots (*Early Blight*, *Bacterial Spot*).
- **Impact:** Because the classification model was trained on clean lab backgrounds (PlantVillage), it lacks explicit negative background rejection training. Textured soil crops trigger high confidence scores ($> 0.70$) for disease classes.
- **Evidence:** Background false positives account for **830 candidate windows** across the 236 test images (3.52 per image).

### Failure Mode 4: Truncated Leaf Context Causing Misclassification
- **Mechanism:** Even when a candidate window achieves valid geometric overlap (IoU $\ge 0.25$), it frequently cuts off the leaf tip or petiole.
- **Impact:** MobileNetV3-Small features rely heavily on global leaf silhouette and margin geometry. Partial crops often misclassify a diseased leaf into an incorrect crop family (e.g. *Tomato Early Blight* misclassified as *Potato Early Blight*).
- **Evidence:** Among accurately localized leaf regions (IoU $\ge 0.25$), disease classification accuracy is **36.59%**.

---

## 3. Engineering Recommendations for Application Integration

1. **Retain Sliding-Window for Regional Screening with Documented Scope:**
   Sliding-window screening is computationally lightweight on CPU (117.16 ms/image) and serves as an effective regional risk screening filter, but must NOT be claimed as a high-precision object detector.
2. **Expose Screening Review Flags:**
   Any candidate region with confidence below $0.70$ must be flagged for secondary review, preventing false alarms from soil/background from misleading growers.
3. **Future Architectural Transition:**
   For production applications requiring precise bounding boxes, an anchor-based or keypoint-based leaf detector (e.g. YOLOv8-Nano or Faster R-CNN with leaf backbone) should precede the MobileNetV3 classifier.
