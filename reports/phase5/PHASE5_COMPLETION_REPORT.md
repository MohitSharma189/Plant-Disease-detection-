# Phase 5: Architecture Comparison & Benchmarking Completion Report

**Date:** 2026-11-12 22:25:34 UTC  
**Dataset:** PlantVillage + PlantDoc Multimodal Dataset (40,151 train / 5,436 val / 5,429 calib / 5,417 test / 236 external PlantDoc)  
**Hardware Platform:** AMD Ryzen 7 5800HS (8 CPU cores / 16 threads, deterministic evaluation, seed 42)  
**Evaluated Architectures:** MobileNetV3-Small (baseline), MobileNetV3-Large, EfficientNet-B0, ResNet18 (ShuffleNetV2 skipped per design)  

---

## 1. Executive Summary & Architecture Selection

Phase 5 systematically benchmarked four candidate convolutional architectures across four axes of evaluation:
1. **Classification Accuracy & Calibration:** Top-1, Top-3, Balanced Accuracy, and Post-Calibration ECE on holdout PlantVillage.
2. **In-the-Wild Generalization:** Uncurated complex-background test performance on 236 PlantDoc images.
3. **End-to-End Latency:** Single-image classification latency and Phase 4.1 multi-scale candidate screening pipeline latency.
4. **Model Footprint:** Parameter count and serialized disk footprint.

### Key Highlights:
- **Highest Holdout Accuracy:** **MobileNetV3-Large** achieved **95.96%** Top-1 accuracy and **94.52%** Macro-F1 on the holdout test set.
- **Strongest In-The-Wild Generalization:** **MobileNetV3-Large** delivered **41.53%** Top-1 and **68.22%** Top-3 accuracy on external PlantDoc.
- **Lowest Latency & Footprint:** **MobileNetV3-Small** remained the most efficient with **9.42 ms** per single inference and only **6.06 MB** disk footprint.
- **Best Calibrated Model:** **MobileNetV3-Small** reached an ECE of **9.94%** after temperature scaling.

---

## 2. Comprehensive Architecture Comparison Table

| Architecture | Parameters | Size (MB) | Val Acc (%) | Test Acc (%) | Test Bal. Acc (%) | Test Macro-F1 (%) | PlantDoc Top-1 (%) | PlantDoc Top-3 (%) | Single Lat. (ms) | Pipeline Lat. (ms) | Opt. Temp | Post-Cal ECE (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MobileNetV3-Small** | 1,556,806 | 6.06 | 95.14% | 95.02% | 95.06% | 93.13% | 38.98% | 64.83% | 9.42 | 662.10 | 0.9902 | 9.94% |
| **MobileNetV3-Large** | 4,250,710 | 16.40 | 96.38% | 95.96% | 95.97% | 94.52% | 41.53% | 68.22% | 16.87 | 1585.19 | 0.9240 | 10.80% |
| **EfficientNet-B0** | 4,056,226 | 15.74 | 96.10% | 95.77% | 95.38% | 94.26% | 34.32% | 62.71% | 24.58 | 2457.42 | 0.8846 | 11.62% |
| **ResNet18** | 11,196,006 | 42.79 | 95.93% | 95.92% | 95.60% | 94.30% | 41.10% | 67.80% | 25.54 | 2025.86 | 1.0392 | 10.21% |

---

## 3. Uncertainty Calibration & Probability Reliability

Models were calibrated via post-hoc Temperature Scaling using L-BFGS on the isolated calibration partition (5,429 samples), and evaluated on the holdout test set:

| Architecture | Optimal Temperature | Pre-Calib ECE (%) | Post-Calib ECE (%) | Pre-Calib NLL | Post-Calib NLL | Pre-Calib Brier | Post-Calib Brier |
|---|---|---|---|---|---|---|---|
| **MobileNetV3-Small** | 0.9902 | 10.28% | 9.94% | 0.2554 | 0.2513 | 0.0993 | 0.0981 |
| **MobileNetV3-Large** | 0.9240 | 14.00% | 10.80% | 0.2755 | 0.2341 | 0.0986 | 0.0854 |
| **EfficientNet-B0** | 0.8846 | 16.43% | 11.62% | 0.3162 | 0.2542 | 0.1156 | 0.0945 |
| **ResNet18** | 1.0392 | 8.99% | 10.21% | 0.2133 | 0.2277 | 0.0809 | 0.0851 |

---

## 4. Checkpoint Provenance & Cryptographic Verification

All model checkpoints and evaluation outputs are rigorously tracked:

| Architecture | Checkpoint Path | SHA-256 Hash | Size (bytes) |
|---|---|---|---|
| **MobileNetV3-Small** | `saved_models\experiments\architecture_benchmark\mobilenet_v3_small\best_model.pth` | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` | 6,353,709 |
| **MobileNetV3-Large** | `saved_models\experiments\architecture_benchmark\mobilenet_v3_large\best_model.pth` | `8ebfb512a071915ca9a7b508e52a9f3ac983ac31618674c43dfdbd6ac168394a` | 17,198,721 |
| **EfficientNet-B0** | `saved_models\experiments\architecture_benchmark\efficientnet_b0\best_model.pth` | `faba42992b190b4c6c37409634070f0a376e26dd6330bd1a6c930bc341f020cb` | 16,508,977 |
| **ResNet18** | `saved_models\experiments\architecture_benchmark\resnet18\best_model.pth` | `f2991bef8630f6e162923fe3224d8d68ba9128ae55ef5bb3ba1ab672f7ea55cf` | 44,863,627 |

---

## 5. Visual Artifacts
- **Performance Comparison Charts:** `reports/phase5/performance_comparison_charts.png`
- **Confusion Matrices:** `reports/phase5/confusion_matrices/`
- **Tabular Data:**
  - `reports/phase5/architecture_comparison.csv`
  - `reports/phase5/classification_metrics.csv`
  - `reports/phase5/external_plantdoc_results.csv`
  - `reports/phase5/calibration_comparison.csv`
  - `reports/phase5/efficiency_benchmark.csv`
  - `reports/phase5/per_class_metrics.csv`

---
*End of Phase 5 Benchmark Report. Ready for user review.*
