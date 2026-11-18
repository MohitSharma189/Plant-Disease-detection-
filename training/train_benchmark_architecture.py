"""Unified Architecture Benchmark Training for Phase 5.

Supports:
- mobilenet_v3_small (baseline reference)
- mobilenet_v3_large
- efficientnet_b0
- resnet18
- shufflenet_v2

Maintains 100% parity with Phase 2 definitive training:
- Exact same data partitions, augmentations, seed=42, batch_size=8, CPU execution
- 1 warmup epoch (frozen backbone), 2 fine-tuning epochs (cosine annealing, label smoothing=0.1)
- Resumable intra-epoch checkpoints (every 500 steps) and atomic end-of-epoch saves
"""

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
from pathlib import Path
from typing import Any

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as transform_functional

ImageFile.LOAD_TRUNCATED_IMAGES = False


class RandomMotionBlur:
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
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        width, height = image.size
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        points = [(random.randrange(width), random.randrange(height)) for _ in range(4)]
        points.extend([(random.randrange(width), random.randrange(height)) for _ in range(2)])
        draw.polygon(points, fill=random.randint(90, 180))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, min(image.size) // 12)))
        darkened = ImageEnhance.Brightness(image.convert("RGB")).enhance(random.uniform(0.45, 0.8))
        return Image.composite(darkened, image.convert("RGB"), mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", type=str, required=True, choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "shufflenet_v2"])
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/PlantVillage"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/generated/combined_split_manifest.json"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Explicit path to checkpoint to resume from.")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-resume from latest_checkpoint.pth if present.")
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


def build_model(architecture: str, class_count: int, head_dropout: float) -> tuple[nn.Module, list[str]]:
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-2] = nn.Dropout(p=head_dropout, inplace=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        return model, ["classifier"]

    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-2] = nn.Dropout(p=head_dropout, inplace=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        return model, ["classifier"]

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[0] = nn.Dropout(p=head_dropout, inplace=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
        return model, ["classifier"]

    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(in_features, class_count))
        return model, ["fc"]

    elif architecture == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(in_features, class_count))
        return model, ["fc"]

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


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
    def __init__(self, data_dir: Path, records: list[dict[str, Any]], class_to_idx: dict[str, int], transform: transforms.Compose) -> None:
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


def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int) -> tuple[float, float, float, np.ndarray, list[int], list[int]]:
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


def set_frozen_stages(model: nn.Module, head_prefix_names: list[str], fine_tune: bool) -> None:
    for name, param in model.named_parameters():
        is_head = any(name.startswith(h) for h in head_prefix_names)
        param.requires_grad = True if (fine_tune or is_head) else False


def build_fine_tune_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, fine_tune_epochs: int, warmup_epochs: int = 1) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * fine_tune_epochs)
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp_path)
    if os.path.exists(path):
        os.replace(tmp_path, path)
    else:
        tmp_path.rename(path)


def train_architecture():
    args = parse_args()
    arch_name = args.architecture
    model_dir = args.model_dir.resolve()
    report_dir = args.report_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"PHASE 5 BENCHMARK TRAINING: {arch_name.upper()}")
    print(f"Model Dir:  {model_dir}")
    print(f"Report Dir: {report_dir}")
    print(f"=======================================================\n")

    tracemalloc.start()
    total_start_time = time.monotonic()

    # Load baseline config as template
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = args.seed
    seed_everything(seed, deterministic=True)

    data_dir = args.data.resolve()
    manifest_path = args.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    augmentation = config["augmentation"]
    image_size = config["image_size"]
    batch_size = args.batch_size
    epochs = args.epochs
    warmup_epochs = args.head_warmup_epochs
    patience = args.patience

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=tuple(augmentation["crop_scale"]), ratio=tuple(augmentation["crop_ratio"])),
        transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip_probability"]),
        transforms.RandomVerticalFlip(p=augmentation["vertical_flip_probability"]),
        transforms.RandomApply([transforms.ColorJitter(brightness=augmentation["brightness"], contrast=augmentation["contrast"], saturation=augmentation["saturation"], hue=augmentation["hue"])], p=augmentation["color_jitter_probability"]),
        transforms.RandomAffine(degrees=augmentation["rotation_degrees"], translate=(augmentation["translation_fraction"],) * 2, scale=tuple(augmentation["scale_range"]), shear=augmentation["shear_degrees"]),
        transforms.RandomPerspective(distortion_scale=augmentation["perspective_distortion"], p=augmentation["perspective_probability"]),
        RandomMotionBlur(augmentation["motion_blur_probability"]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=tuple(augmentation["gaussian_blur_sigma"]))], p=augmentation["gaussian_blur_probability"]),
        RandomShadow(augmentation["shadow_probability"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=augmentation["random_erasing_probability"], scale=tuple(augmentation["random_erasing_scale"]), ratio=tuple(augmentation["random_erasing_ratio"])),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class_names = manifest["class_names"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    records = manifest["records"]
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    assert len(train_records) == 40151, f"Expected 40,151 training records, found {len(train_records)}"
    assert len(validation_records) == 5436, f"Expected 5,436 validation records, found {len(validation_records)}"

    train_dataset = ManifestDataset(data_dir, train_records, class_to_idx, train_transform)
    validation_dataset = ManifestDataset(data_dir, validation_records, class_to_idx, eval_transform)
    device = torch.device("cpu")
    threads = max(1, min(torch.get_num_threads(), 8))
    torch.set_num_threads(threads)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model, head_prefix_names = build_model(arch_name, len(class_names), float(config["head_dropout"]))
    model = model.to(device)

    # Class weighting
    class_counts = torch.bincount(torch.tensor([class_to_idx[r["label"]] for r in train_records]), minlength=len(class_names)).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1)).sqrt()
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(config["label_smoothing"]))

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_validation_macro_f1 = 0.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    log_path = model_dir / "training_log.jsonl"
    config_snapshot = config.copy()
    config_snapshot["architecture"] = arch_name
    config_snapshot["model_dir"] = str(model_dir)
    config_snapshot["report_dir"] = str(report_dir)
    with open(model_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)
    with open(report_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    # Check for auto-resume
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        cand_resume = model_dir / "latest_checkpoint.pth"
        if cand_resume.is_file():
            resume_path = cand_resume

    start_epoch = 1
    start_step = 0
    resume_running_loss = 0.0
    resume_running_correct = 0
    resume_running_total = 0
    resume_ckpt = None

    if resume_path and resume_path.is_file():
        print(f"[*] Resuming from checkpoint: {resume_path}", flush=True)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        ckpt_step = resume_ckpt.get("step", len(train_loader))
        ckpt_epoch = resume_ckpt["epoch"]
        if ckpt_step < len(train_loader):
            start_epoch = ckpt_epoch
            start_step = ckpt_step
            resume_running_loss = resume_ckpt.get("running_loss", 0.0)
            resume_running_correct = resume_ckpt.get("running_correct", 0)
            resume_running_total = resume_ckpt.get("running_total", 0)
            print(f"[*] Mid-epoch resume detected at Epoch {start_epoch}, Step {start_step}/{len(train_loader)}", flush=True)
        else:
            start_epoch = ckpt_epoch + 1
            start_step = 0
            print(f"[*] End-of-epoch resume detected for Epoch {start_epoch}", flush=True)

        best_validation_loss = resume_ckpt.get("best_validation_loss", float("inf"))
        best_validation_accuracy = resume_ckpt.get("best_validation_accuracy", 0.0)
        best_validation_macro_f1 = resume_ckpt.get("best_validation_macro_f1", 0.0)
        best_epoch = resume_ckpt.get("best_epoch", 1)
        stale_epochs = resume_ckpt.get("stale_epochs", 0)
        history = resume_ckpt.get("history", [])
        if "torch_rng_state" in resume_ckpt:
            torch.set_rng_state(resume_ckpt["torch_rng_state"].cpu())
        if "numpy_rng_state" in resume_ckpt:
            np.random.set_state(resume_ckpt["numpy_rng_state"])
        if "python_rng_state" in resume_ckpt:
            random.setstate(resume_ckpt["python_rng_state"])

    print(f"device={device} cpu_threads={threads} train={len(train_dataset)} validation={len(validation_dataset)} classes={len(class_names)} batch_size={batch_size} epochs={epochs} start_epoch={start_epoch}", flush=True)

    fine_tune_optimizer = None
    fine_tune_scheduler = None

    for epoch in range(start_epoch, epochs + 1):
        fine_tune = epoch > warmup_epochs
        set_frozen_stages(model, head_prefix_names, fine_tune)

        backbone_params = [p for n, p in model.named_parameters() if not any(n.startswith(h) for h in head_prefix_names)]
        head_params = [p for n, p in model.named_parameters() if any(n.startswith(h) for h in head_prefix_names)]

        if fine_tune and fine_tune_optimizer is None:
            fine_tune_optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(config["backbone_learning_rate"])},
                    {"params": head_params, "lr": float(config["head_fine_tune_learning_rate"])},
                ],
                weight_decay=float(config["weight_decay"]),
            )
            fine_tune_scheduler = build_fine_tune_scheduler(fine_tune_optimizer, len(train_loader), epochs - warmup_epochs, int(config["cosine_warmup_epochs"]))

        if fine_tune:
            optimizer = fine_tune_optimizer
            scheduler = fine_tune_scheduler
        else:
            optimizer = torch.optim.AdamW(head_params, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
            scheduler = None

        if resume_ckpt and epoch == start_epoch:
            if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"]:
                try:
                    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                    print(f"[*] Successfully restored optimizer state at epoch {epoch}", flush=True)
                except Exception as e:
                    print(f"[!] Warning: could not restore optimizer state: {e}", flush=True)
            if scheduler and "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"]:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                    print(f"[*] Successfully restored scheduler state at epoch {epoch}", flush=True)
                except Exception as e:
                    print(f"[!] Warning: could not restore scheduler state: {e}", flush=True)
            resume_ckpt = None

        model.train()
        if epoch == start_epoch and start_step > 0:
            running_loss = resume_running_loss
            running_correct = resume_running_correct
            running_total = resume_running_total
            print(f"[*] Fast-forwarding {start_step} batches to resume mid-epoch...", flush=True)
        else:
            running_loss = 0.0
            running_correct = 0
            running_total = 0

        epoch_started = time.monotonic()
        for step, (images, labels) in enumerate(train_loader, start=1):
            if epoch == start_epoch and step <= start_step:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            bs_cur = labels.size(0)
            running_loss += loss.item() * bs_cur
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += bs_cur

            if step % 500 == 0 or step == len(train_loader):
                cur_train_loss = running_loss / max(running_total, 1)
                cur_train_acc = running_correct / max(running_total, 1)
                print(f"[{arch_name}] epoch={epoch}/{epochs} step={step}/{len(train_loader)} loss={cur_train_loss:.4f} acc={cur_train_acc:.4f}", flush=True)

            # Periodic intra-epoch recovery checkpoint
            if step % 1000 == 0:
                intra_ckpt = {
                    "architecture": arch_name,
                    "epoch": epoch,
                    "step": step,
                    "running_loss": running_loss,
                    "running_correct": running_correct,
                    "running_total": running_total,
                    "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "best_validation_loss": best_validation_loss,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_validation_macro_f1": best_validation_macro_f1,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "history": history,
                    "config": config_snapshot,
                    "class_names": class_names,
                }
                save_atomic(intra_ckpt, model_dir / "latest_checkpoint.pth")

        val_loss, val_acc, val_macro_f1, conf_mat, all_trues, all_preds = evaluate_epoch(model, validation_loader, device, len(class_names))
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

        print(f"--> [{arch_name}] epoch={epoch}/{epochs} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_macro_f1:.4f} ({epoch_seconds:.1f}s)", flush=True)

        min_delta = float(config.get("minimum_validation_loss_delta", 0.001))
        if val_loss < best_validation_loss - min_delta:
            best_validation_loss = val_loss
            best_validation_accuracy = val_acc
            best_validation_macro_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_atomic(best_state, model_dir / "best_state.pt")
            print(f"[*] New best validation loss: {best_validation_loss:.4f} (epoch {epoch})", flush=True)
        else:
            stale_epochs += 1
            if fine_tune and stale_epochs >= patience:
                print(f"early_stopping=validation_loss patience={patience}", flush=True)
                break

        # Save end-of-epoch checkpoint
        epoch_ckpt = {
            "architecture": arch_name,
            "epoch": epoch,
            "step": len(train_loader),
            "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_macro_f1": best_validation_macro_f1,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": config_snapshot,
            "class_names": class_names,
        }
        save_atomic(epoch_ckpt, model_dir / f"checkpoint_epoch_{epoch}.pth")
        save_atomic(epoch_ckpt, model_dir / "latest_checkpoint.pth")

    total_training_time = time.monotonic() - total_start_time
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    if best_state is None:
        if (model_dir / "best_state.pt").is_file():
            best_state = torch.load(model_dir / "best_state.pt", map_location="cpu")
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save best_model.pth
    best_model_data = {
        "architecture": arch_name,
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "class_names": class_names,
        "head_dropout": float(config["head_dropout"]),
        "image_size": int(config["image_size"]),
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_macro_f1": best_validation_macro_f1,
        "total_training_time_seconds": total_training_time,
        "peak_ram_mb": peak_ram_mb,
        "training_config": config_snapshot,
    }
    save_atomic(best_model_data, model_dir / "best_model.pth")
    save_atomic(best_state, model_dir / "best_state.pt")

    # Also save to report_dir
    save_atomic(best_model_data, report_dir / "best_model.pth")
    save_atomic(best_state, report_dir / "best_state.pt")

    print(f"\nTraining for {arch_name} complete in {total_training_time/60:.1f} min! Peak RAM: {peak_ram_mb:.1f} MB.")
    print(f"Best validation loss: {best_validation_loss:.4f}, Accuracy: {best_validation_accuracy:.4f}, Macro-F1: {best_validation_macro_f1:.4f}")

if __name__ == "__main__":
    train_architecture()

