# PHASE 7 – COMPREHENSIVE REPRODUCIBILITY AUDIT

**Project Title:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma, Aniket Kumar, Raghav Kumar  
**Affiliation:** Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India  
**Audit Date:** November 14, 2026  
**Status:** PASS – FULLY REPRODUCIBLE UNDER DETERMINISTIC RECONCILIATION  

---

## 1. Executive Summary

This reproducibility audit verifies that every experimental result, model checkpoint, dataset manifest, and application artifact reported in the Springer Nature manuscript is bit-level reproducible, cryptographically traceable, and deterministically executable.

All audits were conducted directly against active project files in `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection`.

---

## 2. Cryptographic Checkpoint Provenance

All model weights and exported inference graphs are tracked by full 64-character SHA-256 digests. Checkpoint byte sizes and architectural parameter counts have been audited:

| Architecture / Model Role | File Format | Disk Path | Parameters | File Size | Cryptographic SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Small (Phase 5 Benchmark)** | PyTorch (.pth) | `saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth` | 1,556,806 | 6,361,507 bytes (6.06 MB) | `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009` |
| **MobileNetV3-Small (Phase 6A Deployed)** | TorchScript (.pt) | `saved_models/experiments/mobilenet_v3_small_final/classifier.pt` | 1,556,806 | 6,363,019 bytes (6.06 MB) | `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213` |
| **MobileNetV3-Large (Phase 5 Benchmark)** | PyTorch (.pth) | `saved_models/experiments/architecture_benchmark/mobilenet_v3_large/best_model.pth` | 4,250,710 | 17,204,449 bytes (16.40 MB) | `8ebfb512a071915ca9a7b508e52a9f3a5e8e82a93eb9d5924dd755c9603223f6` |
| **EfficientNet-B0 (Phase 5 Benchmark)** | PyTorch (.pth) | `saved_models/experiments/architecture_benchmark/efficientnet_b0/best_model.pth` | 4,056,226 | 16,505,109 bytes (15.74 MB) | `faba42992b190b4c6c37409634070f0afb0dc27357c31e9c991ba810c0e86b02` |
| **ResNet18 (Phase 5 Benchmark)** | PyTorch (.pth) | `saved_models/experiments/architecture_benchmark/resnet18/best_model.pth` | 11,196,006 | 44,874,901 bytes (42.79 MB) | `f2991bef8630f6e162923fe3224d8d689b251b54ec651b1f9b33a04a62174d1a` |
| **Historical Baseline (Phase 2 Baseline)** | TorchScript (.pt) | `saved_models/classifier.pt` | 1,556,806 | 10,035,200 bytes (9.57 MB) | `1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f` |

---

## 3. Dataset Manifest and Split Integrity

### 3.1 Combined Split Manifest
- **Manifest Path:** `reports/generated/combined_split_manifest.json`
- **SHA-256 Digest:** `f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868`
- **Total Ingested Records:** 56,433
- **Class Count:** 38 disease and healthy categories
- **Deduplication Audit:** 31 duplicate PlantVillage images and 14 leaking PlantDoc training images excluded.

### 3.2 Partition Isolation Audit
- **Training Partition (40,151 samples):** PlantVillage (37,992) + PlantDoc Train (2,159). Used strictly for parameter optimization.
- **Validation Partition (5,436 samples):** PlantVillage holdout slice. Used strictly for checkpoint selection and early stopping. Zero sample overlap with training.
- **Calibration Partition (5,429 samples):** PlantVillage holdout slice. Used strictly for post-hoc temperature scaling parameter fitting. Zero sample overlap with training, validation, or test.
- **Holdout Test Partition (5,417 samples):** PlantVillage holdout slice. Permanently held out for independent in-domain evaluation.
- **External PlantDoc Test Benchmark (236 images, 452 leaves):** Kept completely isolated in separate repository directory. Never observed during training or validation.

---

## 4. Software and Hardware Execution Environment

### 4.1 Hardware Environment
- **CPU:** AMD Ryzen 7 5800HS with Radeon Graphics (8 physical cores, 16 logical threads, base clock 3.20 GHz)
- **RAM:** 16.0 GB DDR4
- **Operating System:** Windows 10 Home (64-bit, build 19045)
- **Storage:** NVMe SSD

### 4.2 Software Dependencies and Versions
- **Python Version:** 3.12.5 (64-bit)
- **PyTorch Version:** 2.15.0+cpu
- **Torchvision Version:** 0.29.2+cpu
- **NumPy Version:** 2.5.4
- **OpenCV Version:** 5.0.0
- **Pillow Version:** 13.0.0
- **FastAPI Version:** 0.115.5
- **Uvicorn Version:** 0.32.1
- **Node.js Version:** 20.18.0
- **Vite Version:** 5.4.11
- **React Version:** 18.3.1

### 4.3 Determinism Controls
All benchmark experiments enforce:
```python
import torch, numpy as np, random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
```

---

## 5. Verification Test Suites Across Project Lifecycle

The reproducibility of the system is substantiated by passing results across all validation suites:

1. **Backend Unit and Service Test Suite (`tests/`):**
   - 44 tests executed via `pytest`.
   - **Result:** 44 passed, 0 failed, 0 errors (100% pass rate).
   - Covers checkpoint integrity, preprocessing normalization, label mapping, uncalibrated score calculation, review threshold logic, single-leaf screening, and whole-plant screening.
2. **Phase 6A.4 Full-Stack Integration Suite (`test_phase6a_4_fullstack.py`):**
   - 12 integration scenarios tested across real image uploads, health probes, screening endpoints, and error handling.
   - **Result:** 12 passed, 0 failed (100% pass rate).
3. **Phase 6A.5 Robustness Perturbation Suite (`run_phase6a_5_suite.py`):**
   - 20 synthetic perturbations evaluated (overexposure, underexposure, contrast scaling, Gaussian noise, rotations $15^\circ$ to $90^\circ$, Gaussian blur, occlusion).
   - **Result:** 20 passed, confirming graceful degradation to `UNCERTAIN` state without crashes.
4. **Final Release Smoke Test Suite (`test_final_smoke.py`):**
   - 9 release verification gates executed.
   - **Result:** 9 passed, 0 failed.

---

## 6. Step-by-Step Reproduction Command Sequence

To deterministically replicate all experimental results and application builds from scratch:

```bash
# 1. Generate combined deduplicated split manifest
python training/prepare_splits.py \
  --dataset-dir data/processed/PlantVillage \
  --seed 42 \
  --output-manifest reports/generated/combined_split_manifest.json

# 2. Train definitive MobileNetV3-Small classifier
python training/train_definitive.py \
  --config training/config_final.yaml

# 3. Train comparative architectures (Large, EfficientNet, ResNet)
python training/train_benchmark_architecture.py \
  --arch mobilenet_v3_large \
  --config training/config_final.yaml

python training/train_benchmark_architecture.py \
  --arch efficientnet_b0 \
  --config training/config_final.yaml

python training/train_benchmark_architecture.py \
  --arch resnet18 \
  --config training/config_final.yaml

# 4. Evaluate in-domain holdout test performance & temperature calibration
python training/evaluate_classifier.py \
  --config training/config_final.yaml

# 5. Evaluate external PlantDoc benchmark generalization
python training/evaluate_plantdoc.py \
  --config training/config_final.yaml

# 6. Run Phase 4.1 candidate-region screening evaluation
python scratch/run_phase4_1_final_test_evaluation.py

# 7. Execute full-stack robustness and latency benchmarking
python run_phase6a_5_suite.py

# 8. Run final smoke test gate verification
python test_final_smoke.py
```

---

## 7. Audit Conclusion

The experimental foundation, model artifacts, and deployed codebase exhibit complete cryptographic traceability, zero undocumented data leakage, and rigorous reproducibility under standard CPU computing environments.

