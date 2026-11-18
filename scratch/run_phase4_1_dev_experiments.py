"""Phase 4.1 Development Set Experiments:
1. Candidate generation comparison (Configs A-E)
2. NMS & suppression ablation
3. Context padding & crop classification ablation
4. Latency profiling
"""

import sys
from pathlib import Path
import os
import csv
import json
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".").resolve()
BEST_MODEL_PATH = PROJECT_ROOT / "saved_models/experiments/mobilenet_v3_small_final/best_model.pth"
DEV_SET_PATH = PROJECT_ROOT / "scratch/dev_set.json"
PLANTDOC_MAPPING_PATH = PROJECT_ROOT / "training/plantdoc_label_mapping.json"
CAL_CONFIG_PATH = PROJECT_ROOT / "reports/phase3/calibrated_inference_config.json"
OUTPUT_DIR = PROJECT_ROOT / "reports/phase4_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

# Load calibration
with open(CAL_CONFIG_PATH, "r", encoding="utf-8") as f:
    cal_cfg = json.load(f)
CALIBRATED_TEMPERATURE = float(cal_cfg["calibration_temperature"])
CALIBRATED_THRESHOLD = float(cal_cfg["review_threshold"])

# Load plantdoc mapping
with open(PLANTDOC_MAPPING_PATH, "r", encoding="utf-8") as f:
    plantdoc_mapping = json.load(f)

# Load model
ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
class_names = ckpt["class_names"]
num_classes = len(class_names)

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
model.to(DEVICE)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def box_iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Window generators
def generate_windows_baseline(w: int, h: int) -> List[List[int]]:
    short_side = min(w, h)
    fraction = 0.38
    side = min(short_side, max(128, round(short_side * fraction)))
    stride = max(64, round(side * 0.62))

    def positions(length: int) -> List[int]:
        if length <= side:
            return [0]
        res = list(range(0, length - side + 1, stride))
        end = length - side
        if res[-1] != end:
            res.append(end)
        return res

    xs = positions(w)
    ys = positions(h)

    def evenly_sample(values: List[int], limit: int) -> List[int]:
        if len(values) <= limit:
            return values
        indices = [round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)]
        return [values[idx] for idx in indices]

    xs = evenly_sample(xs, 6)
    ys = evenly_sample(ys, 6)
    return [[x, y, x + side, y + side] for y in ys for x in xs]

def generate_windows_multiscale(w: int, h: int) -> List[List[int]]:
    short_side = min(w, h)
    fractions = [0.24, 0.40, 0.65]
    windows = []
    for frac in fractions:
        side = min(short_side, max(112, round(short_side * frac)))
        stride = max(64, round(side * 0.65))
        xs = list(range(0, max(1, w - side + 1), stride))
        if xs[-1] != w - side and w >= side:
            xs.append(w - side)
        ys = list(range(0, max(1, h - side + 1), stride))
        if ys[-1] != h - side and h >= side:
            ys.append(h - side)
        if frac == 0.24:
            xs = xs[::2] if len(xs) > 6 else xs
            ys = ys[::2] if len(ys) > 6 else ys
        for y in ys:
            for x in xs:
                windows.append([x, y, x + side, y + side])
    return windows

def generate_windows_aspect_aware(w: int, h: int) -> List[List[int]]:
    short_side = min(w, h)
    configs = [
        (0.25, 1.0, 1.0),
        (0.40, 1.0, 1.0),
        (0.40, 1.33, 1.0),
        (0.40, 1.0, 1.33),
        (0.65, 1.0, 1.0),
    ]
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

def apply_classwise_nms(candidates: List[Dict[str, Any]], iou_threshold: float = 0.55, max_candidates: int = 8) -> List[Dict[str, Any]]:
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    dedup = []
    for c in sorted_cands:
        if any(c["predicted_class"] == ex["predicted_class"] and box_iou(c["box"], ex["box"]) >= iou_threshold for ex in dedup):
            continue
        dedup.append(c)
        if len(dedup) == max_candidates:
            break
    return dedup

def apply_nms(candidates: List[Dict[str, Any]], iou_thresh: float = 0.55, cross_class_thresh: float | None = 0.75, max_cands: int = 8) -> List[Dict[str, Any]]:
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    selected = []
    for cand in sorted_cands:
        suppress = False
        for ex in selected:
            iou = box_iou(cand["box"], ex["box"])
            if cand["predicted_class"] == ex["predicted_class"] and iou >= iou_thresh:
                suppress = True
                break
            if cross_class_thresh is not None and iou >= cross_class_thresh:
                suppress = True
                break
        if not suppress:
            selected.append(cand)
            if len(selected) == max_cands:
                break
    return selected

def evaluate_dev_config(dev_records: List[Dict[str, Any]], config_name: str, config_fn) -> Dict[str, Any]:
    print(f"\nEvaluating: {config_name} on {len(dev_records)} dev images...")
    loc_stats = {0.10: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0},
                 0.25: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0},
                 0.50: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0}}
    all_ious = []
    all_latencies = []
    total_retained = 0

    for item in dev_records:
        t0 = time.time()
        img_path = Path(item["resolved_path"])
        with Image.open(img_path) as src:
            img = src.convert("RGB")
            w, h = img.size
            img_np = np.asarray(img)

        cands = config_fn(img, img_np, w, h)
        total_retained += len(cands)
        latency = (time.time() - t0) * 1000.0
        all_latencies.append(latency)

        gt_leaves = []
        for ann in item["leaf_annotations"]:
            raw_cls = ann["class"]
            mapped_cls = plantdoc_mapping.get(raw_cls, raw_cls)
            gt_leaves.append({"class": mapped_cls, "box": ann["box"]})

        for thresh in [0.10, 0.25, 0.50]:
            if not cands or not gt_leaves:
                loc_stats[thresh]["fn"] += len(gt_leaves)
                loc_stats[thresh]["fp"] += len(cands)
                continue

            ious = np.zeros((len(cands), len(gt_leaves)))
            for ci, c in enumerate(cands):
                for gj, g in enumerate(gt_leaves):
                    ious[ci, gj] = box_iou(c["box"], g["box"])
                    if thresh == 0.25:
                        all_ious.append(ious[ci, gj])

            gt_covered = [False] * len(gt_leaves)
            cand_used = [False] * len(cands)

            pairs = []
            for ci in range(len(cands)):
                for gj in range(len(gt_leaves)):
                    if ious[ci, gj] >= thresh:
                        pairs.append((ious[ci, gj], ci, gj))
            pairs.sort(key=lambda x: x[0], reverse=True)

            for iou_val, ci, gj in pairs:
                if not gt_covered[gj] and not cand_used[ci]:
                    gt_covered[gj] = True
                    cand_used[ci] = True
                    loc_stats[thresh]["tp"] += 1
                    if cands[ci]["predicted_class"] == gt_leaves[gj]["class"]:
                        loc_stats[thresh]["cls_tp"] += 1
                elif gt_covered[gj] and not cand_used[ci]:
                    loc_stats[thresh]["dups"] += 1

            loc_stats[thresh]["fp"] += sum(not u for u in cand_used)
            loc_stats[thresh]["fn"] += sum(not c for c in gt_covered)

    res = {
        "config_name": config_name,
        "mean_latency_ms": np.mean(all_latencies),
        "mean_iou": np.mean(all_ious) if all_ious else 0.0,
        "mean_retained": total_retained / len(dev_records),
    }
    for thresh in [0.10, 0.25, 0.50]:
        st = loc_stats[thresh]
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        acc = st["cls_tp"] / tp if tp > 0 else 0.0
        thresh_str = f"{thresh:.2f}"
        res[f"recall_{thresh_str}"] = r
        res[f"precision_{thresh_str}"] = p
        res[f"f1_{thresh_str}"] = f1
        res[f"fp_per_img_{thresh_str}"] = fp / len(dev_records)
        res[f"class_acc_{thresh_str}"] = acc
        print(f"  [IoU {thresh:.2f}] Recall: {r*100:.2f}%, Prec: {p*100:.2f}%, F1: {f1*100:.2f}%, FP/img: {fp/len(dev_records):.2f}, ClassAcc: {acc*100:.2f}%")

    return res

# -------------------------------------------------------------------------
# EXPERIMENT 1: CANDIDATE GENERATION STRATEGIES
# -------------------------------------------------------------------------
def run_experiment_candidate_generation(dev_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n=======================================================")
    print("EXPERIMENT 1: CANDIDATE GENERATION STRATEGIES (DEV SET)")
    print("=======================================================")

    def run_config_A(img, img_np, w, h):
        wins = generate_windows_baseline(w, h)
        crops = [EVAL_TRANSFORM(img.crop(win)) for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 128]
        valid_wins = [win for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 128]
        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.60:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        return apply_classwise_nms(cands, iou_threshold=0.55, max_candidates=8)

    def run_config_B(img, img_np, w, h):
        wins = generate_windows_multiscale(w, h)
        crops = [EVAL_TRANSFORM(img.crop(win)) for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        valid_wins = [win for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.60:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        return apply_nms(cands, iou_thresh=0.55, cross_class_thresh=0.75, max_cands=8)

    def run_config_C(img, img_np, w, h):
        wins = generate_windows_aspect_aware(w, h)
        crops = [EVAL_TRANSFORM(img.crop(win)) for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        valid_wins = [win for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.60:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        return apply_nms(cands, iou_thresh=0.55, cross_class_thresh=0.75, max_cands=8)

    def run_config_D(img, img_np, w, h):
        raw_wins = generate_windows_multiscale(w, h)
        wins = filter_foliar_saliency(img_np, raw_wins, min_foliar_pct=0.08)
        crops = [EVAL_TRANSFORM(img.crop(win)) for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        valid_wins = [win for win in wins if min(win[2]-win[0], win[3]-win[1]) >= 112]
        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.60:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        return apply_nms(cands, iou_thresh=0.55, cross_class_thresh=0.75, max_cands=8)

    def run_config_E(img, img_np, w, h):
        raw_wins = generate_windows_aspect_aware(w, h)
        wins = filter_foliar_saliency(img_np, raw_wins, min_foliar_pct=0.08)
        crops = []
        valid_wins = []
        for win in wins:
            bw = win[2] - win[0]
            bh = win[3] - win[1]
            pad_x = round(bw * 0.10)
            pad_y = round(bh * 0.10)
            crop_box = [max(0, win[0] - pad_x), max(0, win[1] - pad_y), min(w, win[2] + pad_x), min(h, win[3] + pad_y)]
            crops.append(EVAL_TRANSFORM(img.crop(crop_box)))
            valid_wins.append(win)

        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.55:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})
        return apply_nms(cands, iou_thresh=0.50, cross_class_thresh=0.70, max_cands=10)

    configs = [
        ("Config A (Phase 4 Baseline)", run_config_A),
        ("Config B (Multi-Scale Windows)", run_config_B),
        ("Config C (Aspect-Ratio Aware)", run_config_C),
        ("Config D (Foliar Saliency Pre-filtering)", run_config_D),
        ("Config E (Optimized Composite Pipeline)", run_config_E),
    ]

    results = []
    for name, fn in configs:
        res = evaluate_dev_config(dev_records, name, fn)
        results.append(res)

    out_csv = OUTPUT_DIR / "candidate_generation_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config_name", "mean_latency_ms", "mean_iou", "mean_retained",
            "recall_0.10", "precision_0.10", "f1_0.10", "fp_per_img_0.10", "class_acc_0.10",
            "recall_0.25", "precision_0.25", "f1_0.25", "fp_per_img_0.25", "class_acc_0.25",
            "recall_0.50", "precision_0.50", "f1_0.50", "fp_per_img_0.50", "class_acc_0.50",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for k in fieldnames:
                val = r.get(k)
                if val is not None:
                    if isinstance(val, (int, float)):
                        row[k] = f"{val:.1f}" if "ms" in k else f"{val:.4f}"
                    else:
                        row[k] = str(val)
                else:
                    row[k] = ""
            writer.writerow(row)
    print(f"Saved: {out_csv}")

    # Plot Configuration Comparison Chart
    plt.figure(figsize=(11, 5.5))
    names = ["Config A\nBaseline", "Config B\nMulti-Scale", "Config C\nAspect-Aware", "Config D\nFoliar Filter", "Config E\nComposite Opt"]
    r10 = [r["recall_0.10"] * 100 for r in results]
    r25 = [r["recall_0.25"] * 100 for r in results]
    f1_25 = [r["f1_0.25"] * 100 for r in results]
    c_acc = [r["class_acc_0.25"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.20
    plt.bar(x - 1.5*width, r10, width, label="Recall @ IoU 0.10 (%)", color="#4361EE")
    plt.bar(x - 0.5*width, r25, width, label="Recall @ IoU 0.25 (%)", color="#3A0CA3")
    plt.bar(x + 0.5*width, f1_25, width, label="F1-Score @ IoU 0.25 (%)", color="#2EC4B6")
    plt.bar(x + 1.5*width, c_acc, width, label="Class Accuracy on Crop (%)", color="#FF9F1C")

    plt.title("Candidate Screening Generation Comparison on Development Set (100 Images)", fontsize=12, fontweight="bold")
    plt.xticks(x, names, fontsize=9)
    plt.ylabel("Performance Metric (%)", fontsize=10)
    plt.ylim(0, 85)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "configuration_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved: {chart_path}")

    return results

# -------------------------------------------------------------------------
# EXPERIMENT 2: NMS & SUPPRESSION ABLATION
# -------------------------------------------------------------------------
def run_experiment_nms(dev_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n=======================================================")
    print("EXPERIMENT 2: NMS & CANDIDATE SUPPRESSION ABLATION (DEV SET)")
    print("=======================================================")

    # Pre-extract raw candidate proposals across all dev images using composite aspect-aware + foliar filter
    print("Extracting raw candidate pool for NMS ablation...")
    dev_pools = []
    for item in dev_records:
        img_path = Path(item["resolved_path"])
        with Image.open(img_path) as src:
            img = src.convert("RGB")
            w, h = img.size
            img_np = np.asarray(img)

        raw_wins = generate_windows_aspect_aware(w, h)
        wins = filter_foliar_saliency(img_np, raw_wins, min_foliar_pct=0.08)
        crops = []
        valid_wins = []
        for win in wins:
            bw = win[2] - win[0]
            bh = win[3] - win[1]
            pad_x = round(bw * 0.10)
            pad_y = round(bh * 0.10)
            crop_box = [max(0, win[0] - pad_x), max(0, win[1] - pad_y), min(w, win[2] + pad_x), min(h, win[3] + pad_y)]
            crops.append(EVAL_TRANSFORM(img.crop(crop_box)))
            valid_wins.append(win)

        cands = []
        if crops:
            with torch.inference_mode():
                logits = model(torch.stack(crops))
                probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                confs, preds = probs.max(dim=1)
            for win, conf, pred in zip(valid_wins, confs.tolist(), preds.tolist()):
                lbl = class_names[pred]
                if "healthy" in lbl.lower() or conf < 0.50:
                    continue
                cands.append({"box": win, "predicted_class": lbl, "confidence": conf})

        gt_leaves = []
        for ann in item["leaf_annotations"]:
            raw_cls = ann["class"]
            gt_leaves.append({"class": plantdoc_mapping.get(raw_cls, raw_cls), "box": ann["box"]})

        dev_pools.append({"cands": cands, "gt": gt_leaves})

    nms_variations = [
        {"name": "Class-wise IoU 0.55 (Baseline NMS)", "class_iou": 0.55, "cross_iou": None, "max_cands": 8},
        {"name": "Class-wise IoU 0.45", "class_iou": 0.45, "cross_iou": None, "max_cands": 8},
        {"name": "Class-wise IoU 0.65", "class_iou": 0.65, "cross_iou": None, "max_cands": 8},
        {"name": "Hybrid NMS (Class 0.55, Cross 0.75, Max 8)", "class_iou": 0.55, "cross_iou": 0.75, "max_cands": 8},
        {"name": "Hybrid NMS (Class 0.50, Cross 0.70, Max 8)", "class_iou": 0.50, "cross_iou": 0.70, "max_cands": 8},
        {"name": "Hybrid NMS (Class 0.50, Cross 0.70, Max 10) [Selected]", "class_iou": 0.50, "cross_iou": 0.70, "max_cands": 10},
        {"name": "Hybrid NMS (Class 0.50, Cross 0.70, Max 12)", "class_iou": 0.50, "cross_iou": 0.70, "max_cands": 12},
        {"name": "Aggressive Cross-Class (Class 0.45, Cross 0.60, Max 8)", "class_iou": 0.45, "cross_iou": 0.60, "max_cands": 8},
    ]

    nms_results = []
    for var in nms_variations:
        loc_stats = {0.10: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0},
                     0.25: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0},
                     0.50: {"tp": 0, "fp": 0, "fn": 0, "cls_tp": 0, "dups": 0}}
        total_retained = 0

        for pool in dev_pools:
            filtered = apply_nms(pool["cands"], iou_thresh=var["class_iou"], cross_class_thresh=var["cross_iou"], max_cands=var["max_cands"])
            total_retained += len(filtered)
            gt_leaves = pool["gt"]

            for thresh in [0.10, 0.25, 0.50]:
                if not filtered or not gt_leaves:
                    loc_stats[thresh]["fn"] += len(gt_leaves)
                    loc_stats[thresh]["fp"] += len(filtered)
                    continue

                ious = np.zeros((len(filtered), len(gt_leaves)))
                for ci, c in enumerate(filtered):
                    for gj, g in enumerate(gt_leaves):
                        ious[ci, gj] = box_iou(c["box"], g["box"])

                gt_covered = [False] * len(gt_leaves)
                cand_used = [False] * len(filtered)
                pairs = []
                for ci in range(len(filtered)):
                    for gj in range(len(gt_leaves)):
                        if ious[ci, gj] >= thresh:
                            pairs.append((ious[ci, gj], ci, gj))
                pairs.sort(key=lambda x: x[0], reverse=True)

                for iou_val, ci, gj in pairs:
                    if not gt_covered[gj] and not cand_used[ci]:
                        gt_covered[gj] = True
                        cand_used[ci] = True
                        loc_stats[thresh]["tp"] += 1
                        if filtered[ci]["predicted_class"] == gt_leaves[gj]["class"]:
                            loc_stats[thresh]["cls_tp"] += 1
                    elif gt_covered[gj] and not cand_used[ci]:
                        loc_stats[thresh]["dups"] += 1

                loc_stats[thresh]["fp"] += sum(not u for u in cand_used)
                loc_stats[thresh]["fn"] += sum(not c for c in gt_covered)

        st25 = loc_stats[0.25]
        tp, fp, fn = st25["tp"], st25["fp"], st25["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        st10 = loc_stats[0.10]
        tp10, fp10, fn10 = st10["tp"], st10["fp"], st10["fn"]
        r10 = tp10 / (tp10 + fn10) if (tp10 + fn10) > 0 else 0.0
        p10 = tp10 / (tp10 + fp10) if (tp10 + fp10) > 0 else 0.0

        st50 = loc_stats[0.50]
        tp50, fp50, fn50 = st50["tp"], st50["fp"], st50["fn"]
        r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0

        nms_results.append({
            "nms_strategy": var["name"],
            "class_iou": var["class_iou"],
            "cross_iou": var["cross_iou"] if var["cross_iou"] is not None else "None",
            "max_candidates": var["max_cands"],
            "mean_retained": total_retained / len(dev_pools),
            "recall_iou10": r10,
            "precision_iou10": p10,
            "recall_iou25": r,
            "precision_iou25": p,
            "f1_iou25": f1,
            "recall_iou50": r50,
            "fp_per_img_iou25": fp / len(dev_pools),
            "dups_iou25": st25["dups"],
        })
        print(f"  {var['name']} -> Recall@0.25: {r*100:.2f}%, Prec@0.25: {p*100:.2f}%, F1: {f1*100:.2f}%, FP/img: {fp/len(dev_pools):.2f}, Dups: {st25['dups']}")

    out_csv = OUTPUT_DIR / "nms_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "nms_strategy", "class_iou", "cross_iou", "max_candidates", "mean_retained",
            "recall_iou10", "precision_iou10", "recall_iou25", "precision_iou25", "f1_iou25",
            "recall_iou50", "fp_per_img_iou25", "dups_iou25"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in nms_results:
            row = r.copy()
            for k in ["mean_retained", "recall_iou10", "precision_iou10", "recall_iou25", "precision_iou25", "f1_iou25", "recall_iou50", "fp_per_img_iou25"]:
                row[k] = f"{float(row[k]):.4f}"
            writer.writerow(row)
    print(f"Saved: {out_csv}")
    return nms_results

# -------------------------------------------------------------------------
# EXPERIMENT 3: CONTEXT PADDING & CROP CLASSIFICATION ACCURACY
# -------------------------------------------------------------------------
def run_experiment_context_padding(dev_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n=======================================================")
    print("EXPERIMENT 3: CONTEXT PADDING & CROP CLASSIFICATION (DEV SET)")
    print("=======================================================")

    padding_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    padding_results = []

    for pad_frac in padding_levels:
        correct_count = 0
        total_eval = 0
        confidences = []

        for item in dev_records:
            img_path = Path(item["resolved_path"])
            with Image.open(img_path) as src:
                img = src.convert("RGB")
                w, h = img.size

            for ann in item["leaf_annotations"]:
                gt_raw = ann["class"]
                gt_mapped = plantdoc_mapping.get(gt_raw, gt_raw)
                box = ann["box"]
                bw = box[2] - box[0]
                bh = box[3] - box[1]

                pad_x = round(bw * pad_frac)
                pad_y = round(bh * pad_frac)
                crop_box = (
                    max(0, box[0] - pad_x),
                    max(0, box[1] - pad_y),
                    min(w, box[2] + pad_x),
                    min(h, box[3] + pad_y)
                )
                crop = img.crop(crop_box)
                if min(crop.size) < 32:
                    continue

                tensor = EVAL_TRANSFORM(crop).unsqueeze(0)
                with torch.inference_mode():
                    logits = model(tensor)
                    probs = torch.softmax(logits / CALIBRATED_TEMPERATURE, dim=1)
                    conf, pred = probs.max(dim=1)

                pred_class = class_names[pred.item()]
                if pred_class == gt_mapped:
                    correct_count += 1
                total_eval += 1
                confidences.append(conf.item())

        acc = correct_count / total_eval if total_eval > 0 else 0.0
        mean_conf = np.mean(confidences) if confidences else 0.0
        padding_results.append({
            "context_padding_pct": f"{int(pad_frac * 100)}%",
            "padding_fraction": pad_frac,
            "total_crops_evaluated": total_eval,
            "correct_predictions": correct_count,
            "top1_classification_accuracy": acc,
            "mean_calibrated_confidence": mean_conf,
        })
        print(f"  Padding {int(pad_frac*100)}% -> Top-1 Accuracy: {acc*100:.2f}%, Mean Calibrated Conf: {mean_conf:.3f} (N={total_eval})")

    out_csv = OUTPUT_DIR / "crop_classification_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["context_padding_pct", "padding_fraction", "total_crops_evaluated", "correct_predictions", "top1_classification_accuracy", "mean_calibrated_confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in padding_results:
            row = r.copy()
            row["top1_classification_accuracy"] = f"{row['top1_classification_accuracy']:.4f}"
            row["mean_calibrated_confidence"] = f"{row['mean_calibrated_confidence']:.4f}"
            writer.writerow(row)
    print(f"Saved: {out_csv}")
    return padding_results

def main():
    print(f"Loading dev set from: {DEV_SET_PATH}")
    with open(DEV_SET_PATH, "r", encoding="utf-8") as f:
        dev_records = json.load(f)
    print(f"Total dev set images: {len(dev_records)}")

    # Run experiments
    gen_res = run_experiment_candidate_generation(dev_records)
    nms_res = run_experiment_nms(dev_records)
    pad_res = run_experiment_context_padding(dev_records)

    print("\nPhase 4.1 development experiments completed successfully!")

if __name__ == "__main__":
    main()

