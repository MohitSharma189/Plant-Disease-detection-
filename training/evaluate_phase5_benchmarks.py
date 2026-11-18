"""Phase 5 Architecture Comparison and Benchmarking Evaluation Suite.

Evaluates all benchmarked architectures across:
1. PlantVillage Internal Validation Split (5,436 images)
2. PlantVillage Untouched Holdout Test Split (5,417 images)
3. PlantDoc Independent External Test Partition (236 images)
4. Calibration Split (5,429 images): Temperature scaling, ECE, NLL, Brier score
5. CPU Efficiency & Latency Profiling (single-image classification + Phase 4.1 screening)
6. Per-class metrics and normalized confusion matrices
"""

import sys
sys.path.insert(0, ".")

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
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

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
class_names = manifest["class_names"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

with open(PLANTDOC_MAPPING_PATH, "r", encoding="utf-8") as f:
    plantdoc_mapping = json.load(f)

with open(PLANTDOC_MANIFEST_PATH, "r", encoding="utf-8") as f:
    plantdoc_test_manifest = json.load(f)

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
    if cur.is_file():
        return cur
    return p

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

# Helper for temperature scaling
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval_loss)
        return self.temperature.item()

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def compute_nll_brier(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    n = len(labels)
    # NLL
    correct_conf = probs[np.arange(n), labels]
    nll = -float(np.mean(np.log(np.clip(correct_conf, eps, 1.0))))
    # Brier
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return nll, brier

def load_architecture_model(arch_name: str, checkpoint_path: Path) -> nn.Module:
    print(f"Loading {arch_name} from {checkpoint_path}...")
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
    elif arch_name == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Unknown architecture {arch_name}")

    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

def evaluate_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    probs_cat = torch.softmax(logits_cat, dim=1).numpy()
    return logits_cat.numpy(), probs_cat, labels_cat

def evaluate_plantdoc_test(model: nn.Module) -> dict[str, float]:
    records = plantdoc_test_manifest["records"]
    y_true = []
    y_pred = []
    y_top3 = []

    for item in records:
        leaf_anns = item.get("leaf_annotations", [])
        if not leaf_anns:
            continue
        # Primary leaf class
        gt_raw = leaf_anns[0]["class"]
        gt_mapped = plantdoc_mapping.get(gt_raw, gt_raw)
        if gt_mapped not in class_to_idx:
            continue
        gt_idx = class_to_idx[gt_mapped]

        img_path = Path(item["resolved_path"])
        with Image.open(img_path) as src:
            img = src.convert("RGB")
        tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top3_idx = torch.topk(probs, 3).indices.tolist()

        y_true.append(gt_idx)
        y_pred.append(top3_idx[0])
        y_top3.append(gt_idx in top3_idx)

    top1 = accuracy_score(y_true, y_pred)
    top3 = np.mean(y_top3)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "plantdoc_top1_accuracy": top1,
        "plantdoc_top3_accuracy": top3,
        "plantdoc_macro_precision": p_macro,
        "plantdoc_macro_recall": r_macro,
        "plantdoc_macro_f1": f1_macro,
        "plantdoc_weighted_f1": f1_wt,
        "evaluated_images": len(y_true),
    }

def benchmark_cpu_latency(model: nn.Module, n_warmup: int = 20, n_runs: int = 100) -> tuple[float, float]:
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(dummy)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies)), float(np.std(latencies))

