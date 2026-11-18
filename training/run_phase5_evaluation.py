"""Comprehensive Phase 5 Architecture Benchmark Evaluation Suite.

Evaluates 4 architectures:
- MobileNetV3-Small (baseline)
- MobileNetV3-Large
- EfficientNet-B0
- ResNet18
(Note: ShuffleNetV2 is skipped per experimental instructions)

Deliverables:
1. architecture_comparison.csv
2. classification_metrics.csv
3. external_plantdoc_results.csv
4. calibration_comparison.csv
5. efficiency_benchmark.csv
6. per_class_metrics.csv
7. confusion matrices in reports/phase5/confusion_matrices/
8. performance_comparison_charts.png
9. checkpoint_provenance.json
10. PHASE5_COMPLETION_REPORT.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(".").resolve()
MANIFEST_PATH = PROJECT_ROOT / "reports/generated/combined_split_manifest.json"
DATA_DIR = PROJECT_ROOT / "data/processed/PlantVillage"
PLANTDOC_MAPPING_PATH = PROJECT_ROOT / "training/plantdoc_label_mapping.json"
PLANTDOC_MANIFEST_PATH = PROJECT_ROOT / "reports/generated/candidate_screening_manifest.json"
PHASE4_1_CONFIG_PATH = PROJECT_ROOT / "backend/config/phase4_1/screening_config.json"
REPORT_DIR = PROJECT_ROOT / "reports/phase5"
CONF_MAT_DIR = REPORT_DIR / "confusion_matrices"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")
torch.set_num_threads(8)
torch.manual_seed(42)
np.random.seed(42)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
class_names = manifest["class_names"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

with open(PLANTDOC_MAPPING_PATH, "r", encoding="utf-8") as f:
    plantdoc_mapping = json.load(f)

with open(PLANTDOC_MANIFEST_PATH, "r", encoding="utf-8") as f:
    plantdoc_manifest = json.load(f)

with open(PHASE4_1_CONFIG_PATH, "r", encoding="utf-8") as f:
    screening_config = json.load(f)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    p = base_dir / rel_path
    if p.is_file():
        return p
    parts = Path(rel_path.replace("\\", "/")).parts
    cur = base_dir
    for part in parts:
        exact = cur / part
        if exact.exists():
            cur = exact
            continue
        import re, urllib.parse
        sanitized = re.sub(r'[<>:"/\\|?*%]', '_', part).rstrip(' .')
        cand = cur / sanitized
        if cand.exists():
            cur = cand
            continue
        unquoted = urllib.parse.unquote(part)
        unquoted_cand = cur / unquoted
        if unquoted_cand.exists():
            cur = unquoted_cand
            continue
        break
    return cur if cur.is_file() else p

class ManifestDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform
        self.labels: list[int] = [class_to_idx[r["label"]] for r in records]
        self.paths: list[Path] = []
        for r in records:
            if "dataset_root" in r:
                base = (DATA_DIR / Path(r["dataset_root"])).resolve()
                self.paths.append(_resolve_image_path(base, r["path"]))
            else:
                self.paths.append(DATA_DIR / Path(r["path"]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.labels[index]

def box_iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def generate_windows_optimized(w: int, h: int) -> List[List[int]]:
    short_side = min(w, h)
    configs = [(0.25, 1.0, 1.0), (0.40, 1.0, 1.0), (0.40, 1.333, 1.0), (0.40, 1.0, 1.333), (0.65, 1.0, 1.0)]
    windows = []
    for scale, rw, rh in configs:
        base_side = min(short_side, max(112, round(short_side * scale)))
        win_w = min(w, round(base_side * rw))
        win_h = min(h, round(base_side * rh))
        stride_x = max(64, round(win_w * 0.70))
        stride_y = max(64, round(win_h * 0.70))
        xs = list(range(0, max(1, w - win_w + 1), stride_x))
        if xs[-1] != w - win_w and w >= win_w:
            xs.append(w - win_w)
        ys = list(range(0, max(1, h - win_h + 1), stride_y))
        if ys[-1] != h - win_h and h >= win_h:
            ys.append(h - win_h)
        for y in ys:
            for x in xs:
                windows.append([x, y, x + win_w, y + win_h])
    return windows

def filter_foliar_saliency(img_np: np.ndarray, windows: List[List[int]], min_foliar_pct: float = 0.08) -> List[List[int]]:
    R = img_np[:, :, 0].astype(np.int16)
    G = img_np[:, :, 1].astype(np.int16)
    B = img_np[:, :, 2].astype(np.int16)
    plant_mask = ((2 * G - R - B) > 10) | ((G > 50) & (R > 40) & (B < 100))
    valid_windows = []
    for win in windows:
        x1, y1, x2, y2 = win
        crop_mask = plant_mask[y1:y2, x1:x2]
        if crop_mask.size == 0:
            continue
        pct = np.count_nonzero(crop_mask) / crop_mask.size
        if pct >= min_foliar_pct:
            valid_windows.append(win)
    return valid_windows if valid_windows else windows[:10]

def apply_hybrid_nms(candidates: List[Dict[str, Any]], class_iou: float = 0.50, cross_iou: float = 0.70, max_cands: int = 10) -> List[Dict[str, Any]]:
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    selected = []
    for cand in sorted_cands:
        suppress = False
        for ex in selected:
            iou = box_iou(cand["box"], ex["box"])
            if cand["predicted_class"] == ex["predicted_class"] and iou >= class_iou:
                suppress = True
                break
            if cross_iou is not None and iou >= cross_iou:
                suppress = True
                break
        if not suppress:
            selected.append(cand)
            if len(selected) == max_cands:
                break
    return selected

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def load_architecture_model(arch_name: str, checkpoint_path: Path) -> nn.Module:
    print(f"  Loading {arch_name} from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if arch_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Unknown architecture {arch_name}")

    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            ece += np.abs(conf - acc) * prop
    return float(ece)

def compute_nll_brier(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    n = len(labels)
    correct_conf = probs[np.arange(n), labels]
    nll = -float(np.mean(np.log(np.clip(correct_conf, eps, 1.0))))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return nll, brier

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(eval_loss)
    return float(temperature.item())

def evaluate_loader(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    logits_list, labels_list = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    logits_t = torch.cat(logits_list, dim=0)
    labels_np = torch.cat(labels_list, dim=0).numpy()
    probs_np = torch.softmax(logits_t, dim=1).numpy()
    preds_np = np.argmax(probs_np, axis=1)

    loss = float(nn.CrossEntropyLoss()(logits_t, torch.tensor(labels_np)).item())
    acc = float(accuracy_score(labels_np, preds_np))
    bal_acc = float(balanced_accuracy_score(labels_np, preds_np))

    # Top-3 Accuracy
    top3_indices = np.argsort(probs_np, axis=1)[:, -3:]
    top3_acc = float(np.mean([labels_np[i] in top3_indices[i] for i in range(len(labels_np))]))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels_np, preds_np, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(labels_np, preds_np, average="weighted", zero_division=0)

    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(labels_np, preds_np, average=None, labels=list(range(num_classes)), zero_division=0)

    return {
        "loss": loss,
        "top1_acc": acc,
        "top3_acc": top3_acc,
        "balanced_acc": bal_acc,
        "macro_prec": float(p_macro),
        "macro_rec": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "per_class": {
            "precision": p_cls,
            "recall": r_cls,
            "f1": f1_cls,
            "support": supp_cls,
        },
        "logits": logits_t,
        "probs": probs_np,
        "preds": preds_np,
        "labels": labels_np,
    }

def evaluate_plantdoc(model: nn.Module) -> dict[str, float]:
    records = plantdoc_manifest["records"]
    y_true, y_pred, y_top3 = [], [], []
    for item in records:
        leaf_anns = item.get("leaf_annotations", [])
        if not leaf_anns:
            continue
        gt_raw = leaf_anns[0]["class"]
        gt_mapped = plantdoc_mapping.get(gt_raw, gt_raw)
        if gt_mapped not in class_to_idx:
            continue
        gt_idx = class_to_idx[gt_mapped]
        with Image.open(Path(item["resolved_path"])) as src:
            img = src.convert("RGB")
        tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top3 = torch.topk(probs, 3).indices.tolist()
        y_true.append(gt_idx)
        y_pred.append(top3[0])
        y_top3.append(gt_idx in top3)

    acc = float(accuracy_score(y_true, y_pred))
    top3_acc = float(np.mean(y_top3))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "top1_acc": acc,
        "top3_acc": top3_acc,
        "macro_prec": float(p_macro),
        "macro_rec": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "n_samples": len(y_true),
    }

def benchmark_single_latency(model: nn.Module, n_runs: int = 100) -> tuple[float, float]:
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.inference_mode():
        for _ in range(15):
            _ = model(dummy)
        lats = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            lats.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(lats)), float(np.std(lats))

def benchmark_screening_pipeline(model: nn.Module, temp: float, n_test_images: int = 50) -> float:
    records = plantdoc_manifest["records"][:n_test_images]
    latencies = []
    for item in records:
        t0 = time.time()
        with Image.open(Path(item["resolved_path"])) as src:
            img = src.convert("RGB")
            w, h = img.size
            img_np = np.asarray(img)
        raw_wins = generate_windows_optimized(w, h)
        wins = filter_foliar_saliency(img_np, raw_wins, min_foliar_pct=0.08)
        crops = []
        valid_wins = []
        for win in wins:
            bw, bh = win[2] - win[0], win[3] - win[1]
            pad_x, pad_y = round(bw * 0.12), round(bh * 0.12)
            crop_box = [max(0, win[0] - pad_x), max(0, win[1] - pad_y), min(w, win[2] + pad_x), min(h, win[3] + pad_y)]
            crops.append(EVAL_TRANSFORM(img.crop(crop_box)))
            valid_wins.append(win)
        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / temp, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.55:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        _ = apply_hybrid_nms(cands, class_iou=0.50, cross_iou=0.70, max_cands=10)
        latencies.append((time.time() - t0) * 1000.0)
    return float(np.mean(latencies))

def plot_confusion_matrix(cm_norm: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        cbar=True,
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_performance_comparison(results: List[Dict[str, Any]], save_path: Path) -> None:
    arch_labels = [r["display_name"] for r in results]
    x = np.arange(len(arch_labels))
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. PV Test Accuracy & Balanced Accuracy
    ax = axes[0, 0]
    ax.bar(x - width/2, [r["test_top1_acc"] * 100 for r in results], width, label="Top-1 Acc", color="#2b5c8f")
    ax.bar(x + width/2, [r["test_balanced_acc"] * 100 for r in results], width, label="Balanced Acc", color="#4682b4")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PlantVillage Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.set_ylim(80, 102)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. PV Macro & Weighted F1
    ax = axes[0, 1]
    ax.bar(x - width/2, [r["test_macro_f1"] * 100 for r in results], width, label="Macro F1", color="#2e8b57")
    ax.bar(x + width/2, [r["test_weighted_f1"] * 100 for r in results], width, label="Weighted F1", color="#3cb371")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("PlantVillage Test F1 Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.set_ylim(80, 102)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. External PlantDoc Accuracy
    ax = axes[0, 2]
    ax.bar(x - width/2, [r["plantdoc_top1_acc"] * 100 for r in results], width, label="PlantDoc Top-1", color="#d2691e")
    ax.bar(x + width/2, [r["plantdoc_top3_acc"] * 100 for r in results], width, label="PlantDoc Top-3", color="#f4a460")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("External PlantDoc Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Latency Benchmarks
    ax = axes[1, 0]
    ax.bar(x - width/2, [r["single_latency_ms"] for r in results], width, label="Single Image (ms)", color="#8b008b")
    ax.bar(x + width/2, [r["pipeline_latency_ms"] / 10 for r in results], width, label="Pipeline (ms / 10)", color="#ba55d3")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency (CPU, 8 threads)")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 5. Model Footprint (Parameters & Disk Size)
    ax = axes[1, 1]
    ax.bar(x - width/2, [r["param_count"] / 1e6 for r in results], width, label="Parameters (M)", color="#4169e1")
    ax.bar(x + width/2, [r["ckpt_size_mb"] for r in results], width, label="Checkpoint Size (MB)", color="#6495ed")
    ax.set_ylabel("Size")
    ax.set_title("Model Footprint")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # 6. Expected Calibration Error (Pre vs Post)
    ax = axes[1, 2]
    ax.bar(x - width/2, [r["calib_pre_ece"] * 100 for r in results], width, label="Pre-Calib ECE (%)", color="#dc143c")
    ax.bar(x + width/2, [r["calib_post_ece"] * 100 for r in results], width, label="Post-Calib ECE (%)", color="#20b2aa")
    ax.set_ylabel("ECE (%)")
    ax.set_title("Calibration Error (Test Split)")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Phase 5 Architecture Comparison Summary", fontsize=18, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    print("=" * 65)
    print("PHASE 5 CONSOLIDATED EVALUATION SUITE")
    print("=" * 65)

    MODELS_TO_EVALUATE = [
        {
            "arch_id": "mobilenet_v3_small",
            "display_name": "MobileNetV3-Small",
            "ckpt_path": PROJECT_ROOT / "saved_models/experiments/architecture_benchmark/mobilenet_v3_small/best_model.pth",
        },
        {
            "arch_id": "mobilenet_v3_large",
            "display_name": "MobileNetV3-Large",
            "ckpt_path": PROJECT_ROOT / "saved_models/experiments/architecture_benchmark/mobilenet_v3_large/best_model.pth",
        },
        {
            "arch_id": "efficientnet_b0",
            "display_name": "EfficientNet-B0",
            "ckpt_path": PROJECT_ROOT / "saved_models/experiments/architecture_benchmark/efficientnet_b0/best_model.pth",
        },
        {
            "arch_id": "resnet18",
            "display_name": "ResNet18",
            "ckpt_path": PROJECT_ROOT / "saved_models/experiments/architecture_benchmark/resnet18/best_model.pth",
        },
    ]

    # Verify checkpoints
    for m in MODELS_TO_EVALUATE:
        if not m["ckpt_path"].exists():
            print(f"ERROR: Checkpoint not found: {m['ckpt_path']}")
            sys.exit(1)

    # Build datasets
    val_records = [r for r in manifest["records"] if r["split"] == "validation"]
    test_records = [r for r in manifest["records"] if r["split"] == "test"]
    calib_records = [r for r in manifest["records"] if r["split"] == "calibration"]

    print(f"Data partitions loaded:")
    print(f"  Validation:  {len(val_records)} images")
    print(f"  Test:        {len(test_records)} images")
    print(f"  Calibration: {len(calib_records)} images")

    val_loader = DataLoader(ManifestDataset(val_records, EVAL_TRANSFORM), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(ManifestDataset(test_records, EVAL_TRANSFORM), batch_size=32, shuffle=False, num_workers=0)
    calib_loader = DataLoader(ManifestDataset(calib_records, EVAL_TRANSFORM), batch_size=32, shuffle=False, num_workers=0)

    all_arch_results = []
    per_class_rows = []
    checkpoint_provenance = {}

    for target in MODELS_TO_EVALUATE:
        arch_id = target["arch_id"]
        display_name = target["display_name"]
        ckpt_path = target["ckpt_path"]

        print(f"\n=======================================================")
        print(f"Evaluating Architecture: {display_name} ({arch_id})")
        print(f"=======================================================")

        sha256_hash = compute_sha256(ckpt_path)
        file_size_bytes = ckpt_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        model = load_architecture_model(arch_id, ckpt_path)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        checkpoint_provenance[arch_id] = {
            "display_name": display_name,
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_hash,
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_mb, 2),
            "parameters": param_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 1. Validation evaluation
        print("  Evaluating on Validation Split...")
        val_res = evaluate_loader(model, val_loader)
        print(f"    Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['top1_acc']*100:.2f}% | Val Macro-F1: {val_res['macro_f1']*100:.2f}%")

        # 2. Test evaluation
        print("  Evaluating on Test Split (Holdout)...")
        test_res = evaluate_loader(model, test_loader)
        print(f"    Test Loss: {test_res['loss']:.4f} | Test Acc: {test_res['top1_acc']*100:.2f}% | Test Macro-F1: {test_res['macro_f1']*100:.2f}%")

        # 3. Calibration evaluation
        print("  Evaluating Calibration on Calibration Split & Test...")
        calib_res = evaluate_loader(model, calib_loader)
        opt_temp = calibrate_temperature(calib_res["logits"], torch.tensor(calib_res["labels"]))
        print(f"    Optimal Temperature: {opt_temp:.4f}")

        # Compute pre/post calibration metrics on Test Split
        test_probs_raw = test_res["probs"]
        test_labels = test_res["labels"]
        pre_ece = compute_ece(test_probs_raw, test_labels)
        pre_nll, pre_brier = compute_nll_brier(test_probs_raw, test_labels)

        test_probs_cal = torch.softmax(test_res["logits"] / opt_temp, dim=1).numpy()
        post_ece = compute_ece(test_probs_cal, test_labels)
        post_nll, post_brier = compute_nll_brier(test_probs_cal, test_labels)
        print(f"    Pre-Calib ECE: {pre_ece*100:.2f}% -> Post-Calib ECE: {post_ece*100:.2f}%")

        # 4. External PlantDoc evaluation
        print("  Evaluating on External PlantDoc Test Set (236 images)...")
        pdoc_res = evaluate_plantdoc(model)
        print(f"    PlantDoc Top-1 Acc: {pdoc_res['top1_acc']*100:.2f}% | Top-3 Acc: {pdoc_res['top3_acc']*100:.2f}% | Macro F1: {pdoc_res['macro_f1']*100:.2f}%")

        # 5. Latency benchmarks
        print("  Benchmarking Single-Image Inference Latency (100 runs)...")
        single_lat_mean, single_lat_std = benchmark_single_latency(model, n_runs=100)
        print(f"    Single Image Latency: {single_lat_mean:.2f} ± {single_lat_std:.2f} ms")

        print("  Benchmarking Phase 4.1 Screening Pipeline Latency (50 images)...")
        pipe_lat_mean = benchmark_screening_pipeline(model, temp=opt_temp, n_test_images=50)
        print(f"    End-to-End Pipeline Latency: {pipe_lat_mean:.2f} ms")

        # 6. Confusion matrix
        print("  Generating Normalized Confusion Matrix...")
        cm_norm = confusion_matrix(test_labels, test_res["preds"], labels=list(range(num_classes)), normalize="true")
        cm_path = CONF_MAT_DIR / f"{arch_id}_confusion_matrix.png"
        plot_confusion_matrix(cm_norm, cm_path, f"Normalized Confusion Matrix - {display_name}")

        # 7. Per-class metrics
        for c_idx, c_name in enumerate(class_names):
            per_class_rows.append({
                "architecture": arch_id,
                "display_name": display_name,
                "class_idx": c_idx,
                "class_name": c_name,
                "precision": round(float(test_res["per_class"]["precision"][c_idx]), 4),
                "recall": round(float(test_res["per_class"]["recall"][c_idx]), 4),
                "f1_score": round(float(test_res["per_class"]["f1"][c_idx]), 4),
                "support": int(test_res["per_class"]["support"][c_idx]),
            })

        # Store summary row
        all_arch_results.append({
            "arch_id": arch_id,
            "display_name": display_name,
            "param_count": param_count,
            "ckpt_size_mb": round(file_size_mb, 2),
            "sha256": sha256_hash,
            # Validation
            "val_loss": round(val_res["loss"], 4),
            "val_top1_acc": round(val_res["top1_acc"], 4),
            "val_balanced_acc": round(val_res["balanced_acc"], 4),
            "val_macro_f1": round(val_res["macro_f1"], 4),
            "val_weighted_f1": round(val_res["weighted_f1"], 4),
            # Test
            "test_loss": round(test_res["loss"], 4),
            "test_top1_acc": round(test_res["top1_acc"], 4),
            "test_top3_acc": round(test_res["top3_acc"], 4),
            "test_balanced_acc": round(test_res["balanced_acc"], 4),
            "test_macro_prec": round(test_res["macro_prec"], 4),
            "test_macro_rec": round(test_res["macro_rec"], 4),
            "test_macro_f1": round(test_res["macro_f1"], 4),
            "test_weighted_f1": round(test_res["weighted_f1"], 4),
            # Calibration
            "optimal_temp": round(opt_temp, 4),
            "calib_pre_ece": round(pre_ece, 4),
            "calib_post_ece": round(post_ece, 4),
            "calib_pre_nll": round(pre_nll, 4),
            "calib_post_nll": round(post_nll, 4),
            "calib_pre_brier": round(pre_brier, 4),
            "calib_post_brier": round(post_brier, 4),
            # External PlantDoc
            "plantdoc_top1_acc": round(pdoc_res["top1_acc"], 4),
            "plantdoc_top3_acc": round(pdoc_res["top3_acc"], 4),
            "plantdoc_macro_f1": round(pdoc_res["macro_f1"], 4),
            "plantdoc_weighted_f1": round(pdoc_res["weighted_f1"], 4),
            # Efficiency
            "single_latency_ms": round(single_lat_mean, 2),
            "single_latency_std": round(single_lat_std, 2),
            "pipeline_latency_ms": round(pipe_lat_mean, 2),
        })

    # =========================================================================
    # Write Output Artifacts
    # =========================================================================
    print("\n" + "=" * 65)
    print("Writing Phase 5 Benchmark Deliverables...")
    print("=" * 65)

    # 1. architecture_comparison.csv
    arch_comp_path = REPORT_DIR / "architecture_comparison.csv"
    with open(arch_comp_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "arch_id", "display_name", "param_count", "ckpt_size_mb",
            "val_top1_acc", "val_macro_f1",
            "test_top1_acc", "test_top3_acc", "test_balanced_acc", "test_macro_f1",
            "optimal_temp", "calib_post_ece",
            "plantdoc_top1_acc", "plantdoc_top3_acc",
            "single_latency_ms", "pipeline_latency_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_arch_results:
            writer.writerow(r)
    print(f"  [OK] {arch_comp_path}")

    # 2. classification_metrics.csv
    cls_metrics_path = REPORT_DIR / "classification_metrics.csv"
    with open(cls_metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "arch_id", "display_name",
            "val_loss", "val_top1_acc", "val_balanced_acc", "val_macro_f1", "val_weighted_f1",
            "test_loss", "test_top1_acc", "test_top3_acc", "test_balanced_acc", "test_macro_prec", "test_macro_rec", "test_macro_f1", "test_weighted_f1"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_arch_results:
            writer.writerow(r)
    print(f"  [OK] {cls_metrics_path}")

    # 3. external_plantdoc_results.csv
    pdoc_csv_path = REPORT_DIR / "external_plantdoc_results.csv"
    with open(pdoc_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arch_id", "display_name", "plantdoc_top1_acc", "plantdoc_top3_acc", "plantdoc_macro_f1", "plantdoc_weighted_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_arch_results:
            writer.writerow(r)
    print(f"  [OK] {pdoc_csv_path}")

    # 4. calibration_comparison.csv
    cal_csv_path = REPORT_DIR / "calibration_comparison.csv"
    with open(cal_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arch_id", "display_name", "optimal_temp", "calib_pre_ece", "calib_post_ece", "calib_pre_nll", "calib_post_nll", "calib_pre_brier", "calib_post_brier"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_arch_results:
            writer.writerow(r)
    print(f"  [OK] {cal_csv_path}")

    # 5. efficiency_benchmark.csv
    eff_csv_path = REPORT_DIR / "efficiency_benchmark.csv"
    with open(eff_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arch_id", "display_name", "param_count", "ckpt_size_mb", "single_latency_ms", "single_latency_std", "pipeline_latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_arch_results:
            writer.writerow(r)
    print(f"  [OK] {eff_csv_path}")

    # 6. per_class_metrics.csv
    per_cls_csv_path = REPORT_DIR / "per_class_metrics.csv"
    with open(per_cls_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["architecture", "display_name", "class_idx", "class_name", "precision", "recall", "f1_score", "support"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_class_rows:
            writer.writerow(r)
    print(f"  [OK] {per_cls_csv_path}")

    # 7. checkpoint_provenance.json
    prov_path = REPORT_DIR / "checkpoint_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_provenance, f, indent=2)
    print(f"  [OK] {prov_path}")

    # 8. performance_comparison_charts.png
    charts_path = REPORT_DIR / "performance_comparison_charts.png"
    plot_performance_comparison(all_arch_results, charts_path)
    print(f"  [OK] {charts_path}")

    # 9. PHASE5_COMPLETION_REPORT.md
    report_md_path = REPORT_DIR / "PHASE5_COMPLETION_REPORT.md"
    generate_markdown_report(all_arch_results, checkpoint_provenance, report_md_path)
    print(f"  [OK] {report_md_path}")

    print("\nPhase 5 Evaluation Suite Completed Successfully!")

def generate_markdown_report(results: List[Dict[str, Any]], provenance: Dict[str, Any], save_path: Path) -> None:
    best_pv = max(results, key=lambda r: r["test_top1_acc"])
    best_pdoc = max(results, key=lambda r: r["plantdoc_top1_acc"])
    best_eff = min(results, key=lambda r: r["single_latency_ms"])
    best_cal = min(results, key=lambda r: r["calib_post_ece"])

    content = f"""# Phase 5: Architecture Comparison & Benchmarking Completion Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
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
- **Highest Holdout Accuracy:** **{best_pv['display_name']}** achieved **{best_pv['test_top1_acc']*100:.2f}%** Top-1 accuracy and **{best_pv['test_macro_f1']*100:.2f}%** Macro-F1 on the holdout test set.
- **Strongest In-The-Wild Generalization:** **{best_pdoc['display_name']}** delivered **{best_pdoc['plantdoc_top1_acc']*100:.2f}%** Top-1 and **{best_pdoc['plantdoc_top3_acc']*100:.2f}%** Top-3 accuracy on external PlantDoc.
- **Lowest Latency & Footprint:** **{best_eff['display_name']}** remained the most efficient with **{best_eff['single_latency_ms']:.2f} ms** per single inference and only **{best_eff['ckpt_size_mb']:.2f} MB** disk footprint.
- **Best Calibrated Model:** **{best_cal['display_name']}** reached an ECE of **{best_cal['calib_post_ece']*100:.2f}%** after temperature scaling.

---

## 2. Comprehensive Architecture Comparison Table

| Architecture | Parameters | Size (MB) | Val Acc (%) | Test Acc (%) | Test Bal. Acc (%) | Test Macro-F1 (%) | PlantDoc Top-1 (%) | PlantDoc Top-3 (%) | Single Lat. (ms) | Pipeline Lat. (ms) | Opt. Temp | Post-Cal ECE (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for r in results:
        content += f"| **{r['display_name']}** | {r['param_count']:,} | {r['ckpt_size_mb']:.2f} | {r['val_top1_acc']*100:.2f}% | {r['test_top1_acc']*100:.2f}% | {r['test_balanced_acc']*100:.2f}% | {r['test_macro_f1']*100:.2f}% | {r['plantdoc_top1_acc']*100:.2f}% | {r['plantdoc_top3_acc']*100:.2f}% | {r['single_latency_ms']:.2f} | {r['pipeline_latency_ms']:.2f} | {r['optimal_temp']:.4f} | {r['calib_post_ece']*100:.2f}% |\n"

    content += f"""
---

## 3. Uncertainty Calibration & Probability Reliability

Models were calibrated via post-hoc Temperature Scaling using L-BFGS on the isolated calibration partition (5,429 samples), and evaluated on the holdout test set:

| Architecture | Optimal Temperature | Pre-Calib ECE (%) | Post-Calib ECE (%) | Pre-Calib NLL | Post-Calib NLL | Pre-Calib Brier | Post-Calib Brier |
|---|---|---|---|---|---|---|---|
"""
    for r in results:
        content += f"| **{r['display_name']}** | {r['optimal_temp']:.4f} | {r['calib_pre_ece']*100:.2f}% | {r['calib_post_ece']*100:.2f}% | {r['calib_pre_nll']:.4f} | {r['calib_post_nll']:.4f} | {r['calib_pre_brier']:.4f} | {r['calib_post_brier']:.4f} |\n"

    content += f"""
---

## 4. Checkpoint Provenance & Cryptographic Verification

All model checkpoints and evaluation outputs are rigorously tracked:

| Architecture | Checkpoint Path | SHA-256 Hash | Size (bytes) |
|---|---|---|---|
"""
    for arch_id, p in provenance.items():
        content += f"| **{p['display_name']}** | `{p['checkpoint_path']}` | `{p['sha256']}` | {p['size_bytes']:,} |\n"

    content += f"""
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
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
