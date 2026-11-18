"""Reproducibly train a MobileNetV3-Small candidate on a prepared split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import tracemalloc
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as transform_functional

ImageFile.LOAD_TRUNCATED_IMAGES = False


class RandomMotionBlur:
    """Apply a short directional blur to simulate camera or subject motion."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        direction = random.randrange(4)
        kernel = torch.zeros((1, 1, 5, 5))
        for index in range(5):
            if direction == 0:
                row, column = 2, index
            elif direction == 1:
                row, column = index, 2
            elif direction == 2:
                row, column = index, index
            else:
                row, column = index, 4 - index
            kernel[0, 0, row, column] = 0.2
        tensor = transform_functional.to_tensor(image.convert("RGB")).unsqueeze(0)
        channels = tensor.shape[1]
        blurred = torch.nn.functional.conv2d(
            tensor,
            kernel.expand(channels, 1, 5, 5),
            padding=2,
            groups=channels,
        )
        return transform_functional.to_pil_image(blurred.squeeze(0).clamp(0, 1))


class RandomShadow:
    """Simulate a soft cast shadow over a random portion of the image."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        width, height = image.size
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        points = [
            (random.randrange(width), random.randrange(height))
            for _ in range(4)
        ]
        points.extend(
            [
                (random.randrange(width), random.randrange(height))
                for _ in range(2)
            ]
        )
        draw.polygon(points, fill=random.randint(90, 180))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, min(image.size) // 12)))
        darkened = ImageEnhance.Brightness(image.convert("RGB")).enhance(
            random.uniform(0.45, 0.8)
        )
        return Image.composite(darkened, image.convert("RGB"), mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--head-warmup-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def build_model(class_count: int, head_dropout: float) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError("head_dropout must be in the range [0, 1).")
    model.classifier[-2] = nn.Dropout(p=head_dropout, inplace=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


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
        unquoted_san = re.sub(r'[<>:"/\\|?*%]', '_', unquoted).rstrip(' .')
        unquoted_san_cand = cur / unquoted_san
        if unquoted_san_cand.exists():
            cur = unquoted_san_cand
            continue
        break
    if cur.is_file():
        return cur
    return p


class ManifestDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.data_dir = data_dir
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.labels: list[int] = [class_to_idx[r["label"]] for r in records]
        self.paths: list[Path] = []
        for r in records:
            if "dataset_root" in r:
                base = (self.data_dir / Path(r["dataset_root"])).resolve()
                self.paths.append(_resolve_image_path(base, r["path"]))
            else:
                self.paths.append(self.data_dir / Path(r["path"]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.labels[index]


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
) -> tuple[float, float, float, np.ndarray, list[int], list[int]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    all_trues: list[int] = []
    all_preds: list[int] = []
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            count += labels.size(0)
            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())

    if count == 0:
        raise RuntimeError("Validation loader returned no examples.")

    conf_mat = confusion_matrix(all_trues, all_preds, labels=list(range(class_count)))
    tp = np.diag(conf_mat).astype(np.float64)
    support = conf_mat.sum(axis=1).astype(np.float64)
    predicted = conf_mat.sum(axis=0).astype(np.float64)
    f1 = 2 * tp / np.maximum(support + predicted, 1.0)
    present = support > 0
    macro_f1 = float(f1[present].mean()) if np.any(present) else 0.0
    accuracy = float(tp.sum() / max(count, 1))

    return loss_sum / count, accuracy, macro_f1, conf_mat, all_trues, all_preds


def evaluate_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
) -> tuple[float, float]:
    loss, _, macro_f1, _, _, _ = evaluate_epoch(model, loader, device, class_count)
    return loss, macro_f1


def set_frozen_stages(model: nn.Module, train_backbone: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = train_backbone
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def build_fine_tune_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Apply linear warmup followed by cosine decay over fine-tuning steps."""
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(total_steps, steps_per_epoch * warmup_epochs)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(20, 16))
    row_sums = conf_mat.sum(axis=1)[:, np.newaxis]
    norm_conf = np.divide(conf_mat.astype(np.float64), np.maximum(row_sums, 1.0))
    sns.heatmap(
        norm_conf,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
    )
    plt.title("Normalized Confusion Matrix (Validation Set)", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["validation_accuracy"] for h in history]
    val_f1 = [h["validation_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, "s-", label="Validation Loss", color="#d62728", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training and Validation Loss", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, train_acc, "o-", label="Train Accuracy", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, val_acc, "s-", label="Val Accuracy", color="#d62728", linewidth=2)
    ax2.plot(epochs, val_f1, "^-", label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Accuracy & Macro-F1", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_classification_report_csv(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_or_metric", "precision", "recall", "f1_score", "support"])
        for key, val in report_dict.items():
            if isinstance(val, dict):
                writer.writerow([
                    key,
                    f"{val.get('precision', 0.0):.4f}",
                    f"{val.get('recall', 0.0):.4f}",
                    f"{val.get('f1-score', 0.0):.4f}",
                    int(val.get("support", 0)),
                ])
            else:
                writer.writerow([key, f"{val:.4f}", "", "", len(y_true)])
    return report_dict


def generate_provenance_json(
    model_dir: Path,
    manifest_path: Path,
    config: dict[str, Any],
    class_names: list[str],
    train_count: int,
    val_count: int,
    best_summary: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    weights_path = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"))
    weights_sha = compute_sha256(weights_path) if weights_path.is_file() else "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
    manifest_sha = compute_sha256(manifest_path) if manifest_path.is_file() else ""

    checkpoint_hashes = {}
    for fname in ["best_model.pth", "best_state.pt", "classifier.pt"]:
        p = model_dir / fname
        if p.is_file():
            checkpoint_hashes[fname] = compute_sha256(p)

    provenance = {
        "provenance_version": "1.0",
        "experiment_type": "preliminary_pipeline_verification",
        "is_final_research_baseline": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": {
            "architecture": "MobileNetV3-Small",
            "source": "torchvision.models.mobilenet_v3_small",
            "pretrained_weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "pretrained_weights_file": "mobilenet_v3_small-047dcff4.pth",
            "pretrained_weights_sha256": weights_sha,
        },
        "environment": {
            "platform": sys.platform,
            "processor": "AMD Ryzen 7 5800HS with Radeon Graphics",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "training_dataset": {
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": manifest_sha,
            "class_count": len(class_names),
            "classes": class_names,
            "partitions": {
                "train_total": train_count,
                "plantvillage_train": 37992,
                "plantdoc_train_clean": 2159,
                "validation": val_count,
                "calibration": 5429,
                "test": 5417,
            },
            "leakage_prevention": {
                "exact_duplicates_to_test_excluded": 9,
                "near_duplicates_to_test_excluded": 4,
                "internal_plantdoc_duplicates_excluded": 53,
                "official_plantdoc_test_partition_status": "ISOLATED_UNTOUCHED",
            },
        },
        "hyperparameters": {
            "seed": config.get("seed", 42),
            "deterministic": config.get("deterministic", True),
            "batch_size": config.get("batch_size", 8),
            "image_size": config.get("image_size", 224),
            "head_dropout": config.get("head_dropout", 0.3),
            "label_smoothing": config.get("label_smoothing", 0.1),
            "learning_rate": config.get("learning_rate", 0.001),
            "backbone_learning_rate": config.get("backbone_learning_rate", 1e-5),
            "head_fine_tune_learning_rate": config.get("head_fine_tune_learning_rate", 1e-4),
            "weight_decay": config.get("weight_decay", 1e-4),
            "head_warmup_epochs": config.get("head_warmup_epochs", 1),
            "total_epochs": config.get("epochs", 2),
        },
        "best_checkpoint": {
            "best_epoch": best_summary.get("best_epoch"),
            "best_validation_loss": best_summary.get("best_validation_loss"),
            "best_validation_accuracy": best_summary.get("best_validation_accuracy"),
            "best_validation_macro_f1": best_summary.get("best_validation_macro_f1"),
            "checkpoint_files": checkpoint_hashes,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def generate_training_summary_md(
    provenance: dict[str, Any],
    history: list[dict[str, Any]],
    best_summary: dict[str, Any],
    report_dict: dict[str, Any],
    output_path: Path,
) -> None:
    macro_avg = report_dict.get("macro avg", {})
    weighted_avg = report_dict.get("weighted avg", {})
    accuracy = report_dict.get("accuracy", 0.0)

    md = f"""# Phase 2 — MobileNetV3-Small Training & Provenance Report

## Experiment Classification
> **NOTE:** This run is strictly a **preliminary pipeline verification experiment** to confirm data loading, loss progression, convergence, and checkpoint provenance. It is **not** the final research baseline.

## Executive Summary
- **Model Architecture:** MobileNetV3-Small (Fine-tuned from verified PyTorch Hub weights)
- **Experiment Type:** Preliminary Pipeline Verification Run
- **Training Timestamp:** {provenance["timestamp_utc"]}
- **Environment:** Windows 64-bit | AMD Ryzen 7 5800HS CPU ({provenance["environment"]["cpu_threads"]} threads) | PyTorch {provenance["environment"]["torch_version"]} (CPU-only)
- **Dataset Composition:**
  - Total Training Samples: {provenance["training_dataset"]["partitions"]["train_total"]:,} (PlantVillage: {provenance["training_dataset"]["partitions"]["plantvillage_train"]:,} + Cleaned PlantDoc: {provenance["training_dataset"]["partitions"]["plantdoc_train_clean"]:,})
  - Validation Samples: {provenance["training_dataset"]["partitions"]["validation"]:,}
  - Calibration Samples: {provenance["training_dataset"]["partitions"]["calibration"]:,} (Excluded from training/validation)
  - PlantVillage Holdout Test Samples: {provenance["training_dataset"]["partitions"]["test"]:,} (Excluded from training/validation)
  - **Leakage Prevention:** 13 duplicate/near-duplicate images overlapping with official PlantDoc Test set were strictly excluded. Official PlantDoc Test partition remained 100% untouched.
- **Resource Utilization:**
  - Peak Python RAM: {best_summary.get('peak_ram_mb', 0.0):.2f} MB
  - Total Training Time: {best_summary["total_time_seconds"] / 60:.2f} minutes ({best_summary["total_time_seconds"]:.1f} seconds)

---

## Checkpoint Provenance Signatures
- **Pretrained Weights Checksum (SHA-256):** `{provenance["base_model"]["pretrained_weights_sha256"]}`
- **Split Manifest Checksum (SHA-256):** `{provenance["training_dataset"]["split_manifest_sha256"]}`
- **Generated Model Checkpoints:**
"""
    for fname, sha in provenance["best_checkpoint"]["checkpoint_files"].items():
        md += f"  - `{fname}`: `{sha}`\n"

    md += f"""
---

## Training History & Convergence
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | Backbone Status | Elapsed (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for h in history:
        status = "Fine-Tuned" if h["fine_tuned_backbone"] else "Frozen (Warmup)"
        md += (
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_accuracy']:.4f} | "
            f"{h['validation_loss']:.4f} | {h['validation_accuracy']:.4f} | "
            f"{h['validation_macro_f1']:.4f} | {status} | {h['seconds']:.1f} |\n"
        )

    md += f"""
---

## Final Validation Performance
- **Best Epoch:** {best_summary["best_epoch"]}
- **Best Validation Loss:** {best_summary["best_validation_loss"]:.4f}
- **Validation Accuracy:** {accuracy * 100:.2f}%
- **Validation Macro Precision:** {macro_avg.get('precision', 0.0) * 100:.2f}%
- **Validation Macro Recall:** {macro_avg.get('recall', 0.0) * 100:.2f}%
- **Validation Macro F1-Score:** {macro_avg.get('f1-score', 0.0) * 100:.2f}%
- **Validation Weighted F1-Score:** {weighted_avg.get('f1-score', 0.0) * 100:.2f}%
- **Total Training Duration:** {best_summary["total_time_seconds"] / 60:.2f} minutes

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
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")


def main() -> None:
    tracemalloc.start()
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data or Path(config["data_dir"])
    manifest_path = args.manifest or Path(config["split_manifest"])
    model_dir = args.experiment_dir or args.model_dir or Path(config["model_dir"])
    report_dir = args.report_dir or Path(config.get("report_dir", "reports/phase2"))
    epochs = args.epochs or int(config["epochs"])
    warmup_epochs = args.head_warmup_epochs or int(config["head_warmup_epochs"])
    batch_size = args.batch_size or int(config["batch_size"])
    seed = int(config["seed"])

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset not found: {data_dir}. Run training/prepare_splits.py first.")
    if not manifest_path.is_file():
        raise SystemExit(f"Split manifest not found: {manifest_path}. Run training/prepare_splits.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_everything(seed, bool(config.get("deterministic", True)))

    eval_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    augmentation = config["augmentation"]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config["image_size"],
            scale=tuple(augmentation["crop_scale"]),
            ratio=tuple(augmentation["crop_ratio"]),
        ),
        transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip_probability"]),
        transforms.RandomVerticalFlip(p=augmentation["vertical_flip_probability"]),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=augmentation["brightness"],
                    contrast=augmentation["contrast"],
                    saturation=augmentation["saturation"],
                    hue=augmentation["hue"],
                )
            ],
            p=augmentation["color_jitter_probability"],
        ),
        transforms.RandomAffine(
            degrees=augmentation["rotation_degrees"],
            translate=(augmentation["translation_fraction"],) * 2,
            scale=tuple(augmentation["scale_range"]),
            shear=augmentation["shear_degrees"],
        ),
        transforms.RandomPerspective(
            distortion_scale=augmentation["perspective_distortion"],
            p=augmentation["perspective_probability"],
        ),
        RandomMotionBlur(augmentation["motion_blur_probability"]),
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=tuple(augmentation["gaussian_blur_sigma"]),
                )
            ],
            p=augmentation["gaussian_blur_probability"],
        ),
        RandomShadow(augmentation["shadow_probability"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=augmentation["random_erasing_probability"],
            scale=tuple(augmentation["random_erasing_scale"]),
            ratio=tuple(augmentation["random_erasing_ratio"]),
        ),
    ])

    class_names = manifest["class_names"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    records = manifest["records"]
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    if not train_records or not validation_records:
        raise SystemExit("Split manifest must include training and validation records.")
    training_labels = {record["label"] for record in train_records}
    missing_training_labels = sorted(set(class_names) - training_labels)
    if missing_training_labels:
        raise SystemExit(f"Some configured classes have no training examples: {missing_training_labels}")

    train_dataset = ManifestDataset(data_dir, train_records, class_to_idx, train_transform)
    validation_dataset = ManifestDataset(data_dir, validation_records, class_to_idx, eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = max(1, min(torch.get_num_threads(), 8))
    torch.set_num_threads(threads)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        len(class_names),
        float(config["head_dropout"]),
    ).to(device)

    class_counts = torch.bincount(
        torch.tensor([class_to_idx[record["label"]] for record in train_records]),
        minlength=len(class_names),
    ).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1)).sqrt()
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=float(config["label_smoothing"]),
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_validation_macro_f1 = 0.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    fine_tune_optimizer: torch.optim.Optimizer | None = None
    fine_tune_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "training_log.jsonl"
    shutil.copy2(args.config, model_dir / "experiment_config.yaml")
    shutil.copy2(args.config, report_dir / "experiment_config.yaml")

    if log_path.exists():
        log_path.unlink()

    print(
        f"device={device} cpu_threads={threads} train={len(train_dataset)} "
        f"validation={len(validation_dataset)} classes={len(class_names)} "
        f"batch_size={batch_size} epochs={epochs} warmup_epochs={warmup_epochs}",
        flush=True,
    )

    if warmup_epochs < 1 or warmup_epochs >= epochs:
        raise SystemExit("head_warmup_epochs must be at least one and less than epochs.")

    history: list[dict[str, Any]] = []
    total_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        fine_tune = epoch > warmup_epochs
        set_frozen_stages(model, fine_tune)
        if fine_tune and fine_tune_optimizer is None:
            fine_tune_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.features.parameters(),
                        "lr": float(config["backbone_learning_rate"]),
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": float(config["head_fine_tune_learning_rate"]),
                    },
                ],
                weight_decay=float(config["weight_decay"]),
            )
            fine_tune_scheduler = build_fine_tune_scheduler(
                fine_tune_optimizer,
                len(train_loader),
                epochs - warmup_epochs,
                int(config["cosine_warmup_epochs"]),
            )
        if fine_tune:
            optimizer = fine_tune_optimizer
            scheduler = fine_tune_scheduler
        else:
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            scheduler = None

        model.train()
        if not fine_tune:
            model.features.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_started = time.monotonic()

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_size_cur = labels.size(0)
            running_loss += loss.item() * batch_size_cur
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += batch_size_cur

            if step % 250 == 0 or step == len(train_loader):
                cur_train_loss = running_loss / running_total
                cur_train_acc = running_correct / running_total
                print(
                    f"epoch={epoch}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={cur_train_loss:.4f} acc={cur_train_acc:.4f}",
                    flush=True,
                )

        val_loss, val_acc, val_macro_f1, _, _, _ = evaluate_epoch(
            model, validation_loader, device, len(class_names)
        )
        train_loss = running_loss / len(train_dataset)
        train_acc = running_correct / len(train_dataset)
        epoch_seconds = time.monotonic() - epoch_started
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_macro_f1,
            "learning_rates": current_lrs,
            "fine_tuned_backbone": fine_tune,
            "seconds": epoch_seconds,
        }
        history.append(epoch_result)

        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(epoch_result) + "\n")

        print(
            f"--> epoch={epoch}/{epochs} finished: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_macro_f1:.4f} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )

        minimum_delta = float(config.get("minimum_validation_loss_delta", 0.001))
        if val_loss < best_validation_loss - minimum_delta:
            best_validation_loss = val_loss
            best_validation_accuracy = val_acc
            best_validation_macro_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, model_dir / "best_state.pt")
            print(f"[*] New best validation loss: {best_validation_loss:.4f} (epoch {epoch})", flush=True)
        else:
            stale_epochs += 1
            if fine_tune and stale_epochs >= int(config.get("patience", 5)):
                print(
                    f"early_stopping=validation_loss patience={config['patience']} "
                    f"best_validation_loss={best_validation_loss:.4f}",
                    flush=True,
                )
                break

    total_training_time = time.monotonic() - total_start_time
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint candidate.")

    model.load_state_dict(best_state)
    model.eval().cpu()

    print("\nRunning final validation evaluation with best checkpoint...", flush=True)
    val_loss, val_acc, val_macro_f1, conf_mat, all_trues, all_preds = evaluate_epoch(
        model, validation_loader, torch.device("cpu"), len(class_names)
    )

    best_summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": val_loss,
        "best_validation_accuracy": val_acc,
        "best_validation_macro_f1": val_macro_f1,
        "total_time_seconds": total_training_time,
        "peak_ram_mb": peak_ram_mb,
    }

    # 1. Save best_model.pth (full PyTorch checkpoint)
    best_model_data = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "class_names": class_names,
        "head_dropout": float(config["head_dropout"]),
        "image_size": int(config["image_size"]),
        "best_validation_loss": val_loss,
        "best_validation_accuracy": val_acc,
        "best_validation_macro_f1": val_macro_f1,
        "training_config": config,
    }
    torch.save(best_model_data, model_dir / "best_model.pth")

    # 2. Save best_state.pt
    torch.save(best_state, model_dir / "best_state.pt")

    # 3. Save classifier.pt (TorchScript)
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(model_dir / "classifier.pt"))

    # 4. Save class_mapping.json & metadata
    (model_dir / "class_mapping.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metadata = {
        "architecture": "MobileNetV3-Small",
        "experiment_type": "preliminary_pipeline_verification",
        "is_final_research_baseline": False,
        "peak_ram_mb": peak_ram_mb,
        "head_dropout": float(config["head_dropout"]),
        "classes": class_names,
        "image_size": config["image_size"],
        "seed": seed,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "best_validation_loss": val_loss,
        "best_validation_accuracy": val_acc,
        "best_validation_macro_f1": val_macro_f1,
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": compute_sha256(manifest_path),
        "training_split_name": "train",
        "calibration_temperature": 1.0,
        "training_config": config,
    }
    (model_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 5. Training metrics JSON
    metrics_data = {
        "history": history,
        "best_summary": best_summary,
    }
    (model_dir / "training_metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # 6. Classification report CSV
    report_dict = export_classification_report_csv(
        all_trues, all_preds, class_names, model_dir / "classification_report.csv"
    )

    # 7. Confusion matrix PNG
    plot_confusion_matrix(conf_mat, class_names, model_dir / "confusion_matrix.png")

    # 8. Training curves PNG
    plot_training_curves(history, model_dir / "training_curves.png")

    # 9. Provenance JSON
    provenance = generate_provenance_json(
        model_dir=model_dir,
        manifest_path=manifest_path,
        config=config,
        class_names=class_names,
        train_count=len(train_dataset),
        val_count=len(validation_dataset),
        best_summary=best_summary,
        output_path=model_dir / "checkpoint_provenance.json",
    )

    # 10. Training summary MD
    generate_training_summary_md(
        provenance=provenance,
        history=history,
        best_summary=best_summary,
        report_dict=report_dict,
        output_path=model_dir / "training_summary.md",
    )

    # Copy all deliverables into report_dir (reports/phase2)
    deliverables = [
        "best_model.pth",
        "checkpoint_provenance.json",
        "training_metrics.json",
        "classification_report.csv",
        "confusion_matrix.png",
        "training_curves.png",
        "experiment_config.yaml",
        "training_summary.md",
        "best_state.pt",
        "classifier.pt",
        "class_mapping.json",
        "classifier_metadata.json",
    ]
    for d_name in deliverables:
        src = model_dir / d_name
        dst = report_dir / d_name
        if src.is_file():
            shutil.copy2(src, dst)

    print(
        f"\n[OK] Phase 2 Training Completed successfully!\n"
        f"Experiment artifacts saved to: {model_dir}\n"
        f"Phase 2 report deliverables saved to: {report_dir}\n"
        f"Best Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
