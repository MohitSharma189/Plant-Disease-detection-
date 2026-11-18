# PHASE 7.1 – ARCHITECTURE TRAINING SCHEDULE AND CONFIGURATION AUDIT REPORT

**Project:** Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps  
**Authors:** Mohit Sharma, Aniket Kumar, Raghav Kumar  
**Date:** November 14, 2026  
**Auditor:** Antigravity Research Verification System  
**Audit Scope:** Original training scripts, configuration YAML/JSON files, and epoch-level training logs for all 4 benchmark architectures  
**Status:** PASS – VERIFIED FROM PRIMARY SOURCE LOGS  

---

## 1. Executive Summary

Phase 7.1 independently audited the actual training configurations and execution logs across all four benchmark architectures:
- MobileNetV3-Small (`reports/phase2/final/training_metrics.json`)
- MobileNetV3-Large (`saved_models/experiments/architecture_benchmark/mobilenet_v3_large/training_log.jsonl`)
- EfficientNet-B0 (`saved_models/experiments/architecture_benchmark/efficientnet_b0/training_log.jsonl`)
- ResNet18 (`saved_models/experiments/architecture_benchmark/resnet18/training_log.jsonl`)

**Key Findings:**
1. **Unified Training Protocol:** All four architectures were trained under the **identical hyperparameter protocol** defined in `training/train_benchmark_architecture.py` and `training/config_final.yaml`.
2. **Batch Size:** Exactly **8** (not 64 as claimed in placeholder drafts).
3. **Completed Epochs:** Exactly **3 completed epochs** across all 4 architectures: 1 head-warmup epoch with frozen backbone, followed by 2 full fine-tuning epochs.
4. **Learning Rate Schedule:** Two-stage schedule utilizing **CosineAnnealingLR**:
   - Epoch 1 (Warmup): Head-only learning rate $\eta_{\text{head}} = 1 \times 10^{-3}$ ($0.001$).
   - Epoch 2 (Fine-Tuning): Backbone unfrozen at $\eta_{\text{backbone}} = 1 \times 10^{-5}$ and head at $\eta_{\text{head}} = 1 \times 10^{-4}$.
   - Epoch 3 (Fine-Tuning): Cosine annealing decayed learning rates to $0.0$.
5. **Scheduler Discrepancy Resolved:** Preliminary Phase 1 exploratory scripts used `ReduceLROnPlateau`. However, all definitive models (Phase 2 final MobileNetV3-Small and Phase 5 multi-architecture benchmark models) used `CosineAnnealingLR` with 1-epoch head warmup.
6. **Weight Decay:** Exactly **$1 \times 10^{-4}$** ($0.0001$, not $1 \times 10^{-2}$).

---

## 2. Parameter-by-Parameter Verification Matrix

| Parameter / Configuration | MobileNetV3-Small | MobileNetV3-Large | EfficientNet-B0 | ResNet18 | Unified Protocol Verified? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | **YES (Identical)** |
| **Optimizer Betas** | $(\beta_1=0.9, \beta_2=0.999)$ | $(\beta_1=0.9, \beta_2=0.999)$ | $(\beta_1=0.9, \beta_2=0.999)$ | $(\beta_1=0.9, \beta_2=0.999)$ | **YES (Identical)** |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | **YES (Identical)** |
| **Batch Size** | **8** | **8** | **8** | **8** | **YES (Identical)** |
| **Warmup Epochs** | 1 (frozen backbone) | 1 (frozen backbone) | 1 (frozen backbone) | 1 (frozen backbone) | **YES (Identical)** |
| **Warmup Head LR** | $1 \times 10^{-3}$ ($0.001$) | $1 \times 10^{-3}$ ($0.001$) | $1 \times 10^{-3}$ ($0.001$) | $1 \times 10^{-3}$ ($0.001$) | **YES (Identical)** |
| **Fine-Tuning Epochs** | 2 | 2 | 2 | 2 | **YES (Identical)** |
| **Backbone Fine-Tune LR**| $1 \times 10^{-5}$ | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ | **YES (Identical)** |
| **Head Fine-Tune LR** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | **YES (Identical)** |
| **LR Scheduler** | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | **YES (Identical)** |
| **Total Epochs** | **3** | **3** | **3** | **3** | **YES (Identical)** |
| **Label Smoothing** | 0.10 | 0.10 | 0.10 | 0.10 | **YES (Identical)** |
| **Input Dimensions** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | **YES (Identical)** |
| **Pretrained Initialization**| ImageNet weights | ImageNet weights | ImageNet weights | ImageNet weights | **YES (Identical)** |
| **Random Seed** | 42 | 42 | 42 | 42 | **YES (Identical)** |
| **Deterministic Mode** | `torch.use_deterministic_algorithms(True)` | `torch.use_deterministic_algorithms(True)` | `torch.use_deterministic_algorithms(True)` | `torch.use_deterministic_algorithms(True)` | **YES (Identical)** |
| **Hardware** | AMD Ryzen 7 5800HS (CPU) | AMD Ryzen 7 5800HS (CPU) | AMD Ryzen 7 5800HS (CPU) | AMD Ryzen 7 5800HS (CPU) | **YES (Identical)** |

---

## 3. Direct Source Log Evidence by Architecture

### 3.1 MobileNetV3-Small
*Source Artifact:* `reports/phase2/final/training_metrics.json`
- **Epoch 1 (Warmup):** `train_loss: 1.5259`, `val_loss: 0.3481`, `val_acc: 0.9347`, `val_macro_f1: 0.9125`, `lr: [0.001]`, `fine_tuned_backbone: false`, `time: 644.5 s`.
- **Epoch 2 (Fine-Tune):** `train_loss: 2.0604`, `val_loss: 0.2781`, `val_acc: 0.9389`, `val_macro_f1: 0.9153`, `lr: [1e-05, 0.0001]`, `fine_tuned_backbone: true`, `time: 1877.2 s`.
- **Epoch 3 (Fine-Tune):** `train_loss: 1.4269`, `val_loss: 0.2514`, `val_acc: 0.9514`, `val_macro_f1: 0.9319`, `lr: [0.0, 0.0]`, `fine_tuned_backbone: true`, `time: 1405.6 s`.
- **Best Selection:** Epoch 3 (`val_loss: 0.2514`, `val_acc: 0.9514`, `val_macro_f1: 0.9319`). Checkpoint SHA-256: `6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009`.

### 3.2 MobileNetV3-Large
*Source Artifact:* `saved_models/experiments/architecture_benchmark/mobilenet_v3_large/training_log.jsonl`
- **Epoch 1 (Warmup):** `train_loss: 1.7914`, `val_loss: 0.4511`, `val_acc: 0.8727`, `val_macro_f1: 0.8403`, `lr: [0.001]`, `fine_tuned_backbone: false`, `time: 1667.7 s`.
- **Epoch 2 (Fine-Tune):** `train_loss: 1.4641`, `val_loss: 0.3062`, `val_acc: 0.9472`, `val_macro_f1: 0.9307`, `lr: [1e-05, 0.0001]`, `fine_tuned_backbone: true`, `time: 2943.8 s`.
- **Epoch 3 (Fine-Tune):** `train_loss: 1.2879`, `val_loss: 0.2677`, `val_acc: 0.9638`, `val_macro_f1: 0.9488`, `lr: [0.0, 0.0]`, `fine_tuned_backbone: true`, `time: 3736.9 s`.
- **Best Selection:** Epoch 3 (`val_loss: 0.2677`, `val_acc: 0.9638`, `val_macro_f1: 0.9488`). Checkpoint SHA-256: `8ebfb512a071915ca9a7b508e52a9f3a5e8e82a93eb9d5924dd755c9603223f6`.

### 3.3 EfficientNet-B0
*Source Artifact:* `saved_models/experiments/architecture_benchmark/efficientnet_b0/training_log.jsonl`
- **Epoch 1 (Warmup):** `train_loss: 1.8969`, `val_loss: 0.5472`, `val_acc: 0.8916`, `val_macro_f1: 0.8590`, `lr: [0.001]`, `fine_tuned_backbone: false`, `time: 1717.5 s`.
- **Epoch 2 (Fine-Tune):** `train_loss: 1.6193`, `val_loss: 0.3751`, `val_acc: 0.9432`, `val_macro_f1: 0.9244`, `lr: [1e-05, 0.0001]`, `fine_tuned_backbone: true`, `time: 5360.9 s`.
- **Epoch 3 (Fine-Tune):** `train_loss: 1.4301`, `val_loss: 0.3035`, `val_acc: 0.9610`, `val_macro_f1: 0.9472`, `lr: [0.0, 0.0]`, `fine_tuned_backbone: true`, `time: 5713.2 s`.
- **Best Selection:** Epoch 3 (`val_loss: 0.3035`, `val_acc: 0.9610`, `val_macro_f1: 0.9472`). Checkpoint SHA-256: `faba42992b190b4c6c37409634070f0afb0dc27357c31e9c991ba810c0e86b02`.

### 3.4 ResNet18
*Source Artifact:* `saved_models/experiments/architecture_benchmark/resnet18/training_log.jsonl`
- **Epoch 1 (Warmup):** `train_loss: 2.2699`, `val_loss: 0.6200`, `val_acc: 0.8407`, `val_macro_f1: 0.7796`, `lr: [0.001]`, `fine_tuned_backbone: false`, `time: 1859.4 s`.
- **Epoch 2 (Fine-Tune):** `train_loss: 1.8048`, `val_loss: 0.2474`, `val_acc: 0.9459`, `val_macro_f1: 0.9277`, `lr: [1e-05, 0.0001]`, `fine_tuned_backbone: true`, `time: 4516.5 s`.
- **Epoch 3 (Fine-Tune):** `train_loss: 1.5124`, `val_loss: 0.2105`, `val_acc: 0.9593`, `val_macro_f1: 0.9478`, `lr: [0.0, 0.0]`, `fine_tuned_backbone: true`, `time: 3876.8 s`.
- **Best Selection:** Epoch 3 (`val_loss: 0.2105`, `val_acc: 0.9593`, `val_macro_f1: 0.9478`). Checkpoint SHA-256: `f2991bef8630f6e162923fe3224d8d689b251b54ec651b1f9b33a04a62174d1a`.

---

## 4. Manuscript Corrections Implemented

Table 2 in `manuscript/phase7_1_final/main.tex` and the Abstract have been corrected:
- `Batch size: 8` (corrected from 64).
- `Total completed epochs: 3` (corrected from 20).
- `Warmup: 1 epoch at lr 0.001 (frozen backbone)`.
- `Fine-tuning: 2 epochs at backbone lr 1e-05 and head lr 1e-04 with CosineAnnealingLR`.
- `Weight decay: 1e-04` (corrected from 1e-02).
- Zero unverified claims: The paper now reflects the exact parameters executed by the training scripts.

