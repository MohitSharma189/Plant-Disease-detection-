# PHASE 2 COMPLETION REPORT — Definitive MobileNetV3-Small Baseline

**Project:** Plant Disease Screening System  
**Baseline Model:** MobileNetV3-Small (Reproducible Fine-Tuning)  
**Execution Timestamp:** 2026-11-12T08:40:54.173456+00:00  
**Experiment Directory:** `saved_models\experiments\mobilenet_v3_small_final`  
**Report Directory:** `reports\phase2\final`  
**Status:** **GO — Phase 2 Successfully Completed**

---

## 1. Executive Summary & Verification Decision
- **Architecture:** MobileNetV3-Small (Pretrained weights initialized from torchvision)
- **Deterministic Seed:** 42 (`torch.use_deterministic_algorithms(True)`)
- **Total Training Duration:** 65.51 minutes (3930.7 seconds)
- **Peak RAM Utilization:** 63.63 MB
- **Best Validation Epoch:** Epoch 3
- **Best Validation Loss:** 0.2514
- **Validation Accuracy:** 95.14%
- **Validation Macro-F1 Score:** 93.19%
- **Validation Weighted-F1 Score:** 95.25%
- **Independent Inference Verification:** **PASSED** (Confidence: 17.84%, NaN: False, Inf: False)

---

## 2. Checkpoint Provenance & Cryptographic Signatures

| Component | Identifier / Path | SHA-256 Checksum |
|---|---|---|
| **Pretrained Base Weights** | `mobilenet_v3_small-047dcff4.pth` | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |
| **Combined Split Manifest** | `reports\generated\combined_split_manifest.json` | `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868` |
| **Best Model Checkpoint** | `best_model.pth` | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` |
| **Best State Dict** | `best_state.pt` | `97cfe371e54b040bfa7b722e3fb9e4a1d81fbc5e850337acbcd42394455852cc` |
| **Scripted TorchScript** | `classifier.pt` | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` |
| **Latest Resumption Checkpoint** | `latest_checkpoint.pth` | `79551f03e04d37ac65ab2ab5756d9aeb5a37a277149b726d5b25d179f3e3080d` |

---

## 3. Dataset Composition & Partition Isolation Audit

The dataset manifest was strictly applied without modification:

- **Training Partition (40,151 images):**
  - PlantVillage clean training: 37,992 images
  - Cleaned, deduplicated PlantDoc train: 2,159 images
  - **Gradient Updates:** Active only during training forward/backward loops.
- **Validation Partition (5,436 images):**
  - Used strictly for epoch-level evaluation, learning-rate scheduling, and early stopping model selection.
- **Calibration Partition (5,429 images):**
  - **100% Isolated.** Zero images from this split were loaded into training or validation loaders.
- **Holdout Test Partitions:**
  - **PlantVillage Holdout Test (5,417 images):** Excluded from training and model selection.
  - **Official PlantDoc Test Partition (505 images):** Completely untouched and isolated.
  - **Leakage Prevention:** 13 duplicate/near-duplicate images overlapping with official PlantDoc Test were excluded from training. 53 internal PlantDoc duplicates were excluded.

---

## 4. Training History & Convergence

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | Backbone Status | Elapsed (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.5259 | 0.8232 | 0.3481 | 0.9347 | 0.9125 | Frozen (Warmup) | 644.5 |
| 2 | 2.0604 | 0.6496 | 0.2781 | 0.9389 | 0.9153 | Fine-Tuned | 1877.2 |
| 3 | 1.4269 | 0.8506 | 0.2514 | 0.9514 | 0.9319 | Fine-Tuned | 1405.6 |

---

## 5. Deliverables & File Verification Audit
All artifacts have been generated and verified in both `saved_models\experiments\mobilenet_v3_small_final` and `reports\phase2\final`:

1. `best_model.pth` — Complete PyTorch model archive containing weights, class mapping, and training hyperparameters.
2. `best_state.pt` — Raw model state dictionary for lightweight loading.
3. `classifier.pt` — TorchScript traced/scripted model for offline inference deployment.
4. `latest_checkpoint.pth` — Interruption-safe checkpoint containing optimizer state, scheduler state, epoch/step indices, and RNG states.
5. `class_mapping.json` — 38-class ordered mapping.
6. `classifier_metadata.json` — Environment and training metadata.
7. `checkpoint_provenance.json` — Complete audit trail with SHA-256 signatures.
8. `training_metrics.json` & `training_metrics.csv` — Step-level and epoch-level loss, accuracy, and macro-F1 history in JSON and tabular CSV format.
9. `classification_report.csv` — Comprehensive per-class precision, recall, and F1 table across all 38 classes.
10. `confusion_matrix.png` — Normalized 38x38 confusion matrix heatmap.
11. `training_curves.png` — High-resolution loss and accuracy progression curves.
12. `experiment_config.yaml` — Exact configuration snapshot.
13. `PHASE2_COMPLETION_REPORT.md` — This definitive completion document.

---

## 6. Preservation of Prior Checkpoints
- Legacy production baseline (`saved_models/classifier.pt`): **Preserved untouched** (Timestamp: 2026-11-11).
- Legacy candidate baseline (`saved_models/candidate/best_state.pt`): **Preserved untouched**.
- Preliminary verification experiment (`saved_models/experiments/mobilenet_v3_small_verification/`): **Preserved untouched**.

---

## 7. Limitations & Recommendations for Phase 3
1. **CPU Execution Profile:** Training on CPU is memory-efficient and deterministic, but throughput is approximately 40 images/sec on AMD Ryzen 7 5800HS.
2. **Holdout Evaluation Required:** Per research protocol, Phase 2 evaluated only on the internal validation split. Formal external evaluation on the official PlantDoc Test benchmark must be performed in Phase 3.
3. **Next Phase:** Phase 3 — Independent External Evaluation & Temperature Calibration.
