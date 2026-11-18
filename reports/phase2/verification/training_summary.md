# Phase 2 — MobileNetV3-Small Training & Provenance Report

## Experiment Classification
> **NOTE:** This run is strictly a **preliminary pipeline verification experiment** to confirm data loading, loss progression, convergence, and checkpoint provenance. It is **not** the final research baseline.

## Executive Summary
- **Model Architecture:** MobileNetV3-Small (Fine-tuned from verified PyTorch Hub weights)
- **Experiment Type:** Preliminary Pipeline Verification Run
- **Training Timestamp:** 2026-11-12T07:33:56.914228+00:00
- **Environment:** Windows 64-bit | AMD Ryzen 7 5800HS CPU (8 threads) | PyTorch 2.15.0+cpu (CPU-only)
- **Dataset Composition:**
  - Total Training Samples: 40,151 (PlantVillage: 37,992 + Cleaned PlantDoc: 2,159)
  - Validation Samples: 5,436
  - Calibration Samples: 5,429 (Excluded from training/validation)
  - PlantVillage Holdout Test Samples: 5,417 (Excluded from training/validation)
  - **Leakage Prevention:** 13 duplicate/near-duplicate images overlapping with official PlantDoc Test set were strictly excluded. Official PlantDoc Test partition remained 100% untouched.
- **Resource Utilization:**
  - Peak Python RAM: 63.33 MB
  - Total Training Time: 41.68 minutes (2500.9 seconds)

---

## Checkpoint Provenance Signatures
- **Pretrained Weights Checksum (SHA-256):** `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- **Split Manifest Checksum (SHA-256):** `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868`
- **Generated Model Checkpoints:**
  - `best_model.pth`: `792888f40713ada7dd7a741073616b477d7fec6020e72fae6a75d0aa81b3d2aa`
  - `best_state.pt`: `43d027fb41f973ce2aa2cd0d6ead3e4be1709becd834866e7f442cb4bd346b39`
  - `classifier.pt`: `361db5755fe4189576bce04e94f622a7bc043f32fabba15accdf38b4fecf49c3`

---

## Training History & Convergence
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | Backbone Status | Elapsed (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.5259 | 0.8232 | 0.3481 | 0.9347 | 0.9125 | Frozen (Warmup) | 1065.5 |
| 2 | 2.0604 | 0.6496 | 0.2781 | 0.9389 | 0.9153 | Fine-Tuned | 1435.3 |

---

## Final Validation Performance
- **Best Epoch:** 2
- **Best Validation Loss:** 0.2781
- **Validation Accuracy:** 93.89%
- **Validation Macro Precision:** 90.61%
- **Validation Macro Recall:** 93.17%
- **Validation Macro F1-Score:** 91.53%
- **Validation Weighted F1-Score:** 93.96%
- **Total Training Duration:** 41.68 minutes

---

## Deliverables Generated
1. `best_model.pth`: PyTorch full checkpoint containing state dictionary, class mappings, and hyperparameters.
2. `best_state.pt`: Raw model state dictionary.
3. `classifier.pt`: TorchScript scripted model for deployment.
4. `checkpoint_provenance.json`: Cryptographic and environment audit trail.
5. `training_metrics.json`: Detailed step and epoch metrics.
6. `classification_report.csv`: Complete 38-class precision, recall, and F1 table.
7. `confusion_matrix.png`: Normalized 38x38 confusion matrix heatmap.
8. `training_curves.png`: Loss and accuracy/F1 progression plots.
9. `experiment_config.yaml`: Frozen snapshot of experiment hyperparameters.
10. `training_summary.md`: This comprehensive verification report.
