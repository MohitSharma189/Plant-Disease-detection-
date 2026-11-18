"""
Verified experimental data from Phase 2-5 reports.
All values traced to source CSV/JSON files.
"""

# ── Phase 2: Definitive MobileNetV3-Small Checkpoint ──────────────────────────
PHASE2 = {
    "architecture": "MobileNetV3-Small",
    "seed": 42,
    "training_images": 40151,  # 37992 PV + 2159 PlantDoc train
    "val_images": 5436,
    "calib_images": 5429,
    "holdout_images": 5417,
    "plantdoc_external": 236,
    "val_acc": 95.14,
    "val_macro_f1": 93.19,
    "val_weighted_f1": 95.25,
    "best_epoch": 3,
    "best_val_loss": 0.2514,
    "training_minutes": 65.51,
    "checkpoint_sha256": "6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009",
    "manifest_sha256": "f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868",
    "pretrained_sha256": "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
    "epochs": [
        {"epoch": 1, "train_loss": 1.5259, "train_acc": 82.32, "val_loss": 0.3481, "val_acc": 93.47, "val_f1": 91.25, "backbone": "Frozen"},
        {"epoch": 2, "train_loss": 2.0604, "train_acc": 64.96, "val_loss": 0.2781, "val_acc": 93.89, "val_f1": 91.53, "backbone": "Fine-Tuned"},
        {"epoch": 3, "train_loss": 1.4269, "train_acc": 85.06, "val_loss": 0.2514, "val_acc": 95.14, "val_f1": 93.19, "backbone": "Fine-Tuned"},
    ],
}

# ── Phase 3: Calibration & External Evaluation ────────────────────────────────
PHASE3_CALIB = {
    "temperature": 0.5538,
    "pre_ece": 10.25,   # calibration partition
    "post_ece": 0.58,   # calibration partition
    "pre_nll": 0.2426,
    "post_nll": 0.1454,
    "pre_brier": 0.0936,
    "post_brier": 0.0710,
}

PHASE3_HOLDOUT = {
    "samples": 5417,
    "accuracy": 95.02,
    "balanced_acc": 95.06,
    "macro_precision": 92.16,
    "macro_recall": 95.06,
    "macro_f1": 93.13,
    "weighted_precision": 95.51,
    "weighted_recall": 95.02,
    "weighted_f1": 95.14,
    "pre_ece": 10.28,
    "post_ece": 0.74,
    "pre_nll": 0.2554,
    "post_nll": 0.1637,
    "pre_brier": 0.0993,
    "post_brier": 0.0761,
    "temperature": 0.5538,
}

PHASE3_PLANTDOC = {
    "samples": 236,
    "evaluated_classes": 27,
    "unrepresented_classes": 11,
    "accuracy": 38.98,
    "macro_precision": 44.66,
    "macro_recall": 37.96,
    "macro_f1": 36.84,
    "weighted_precision": 45.09,
    "weighted_recall": 38.98,
    "weighted_f1": 37.41,
    "domain_gap_drop": 56.03,
    "selective_coverage": 93.22,
    "selective_accuracy": 40.91,
}

# ── Phase 4 Baseline vs Phase 4.1 Optimized ───────────────────────────────────
PHASE4_VS_41 = [
    ("Recall @ IoU 0.10",    50.00, 62.17, "+24.34%"),
    ("Precision @ IoU 0.10", 24.78, 17.31, "-30.13%"),
    ("F1 @ IoU 0.10",        33.14, 27.08, "-18.27%"),
    ("Recall @ IoU 0.25",    18.14, 38.05, "+109.76%"),
    ("Precision @ IoU 0.25",  8.99, 10.60, "+17.87%"),
    ("F1 @ IoU 0.25",        12.02, 16.58, "+37.88%"),
    ("Recall @ IoU 0.50",     3.10,  7.96, "+157.14%"),
    ("Precision @ IoU 0.50",  1.54,  2.22, "+44.49%"),
    ("F1 @ IoU 0.50",         2.05,  3.47, "+69.03%"),
    ("False Pos/Image @0.25", 3.52,  6.15, "+74.82%"),
    ("CPU Latency (ms)",     167.0, 844.2, "+405.49%"),
]

# ── Phase 4.1 Foliage Density Breakdown ──────────────────────────────────────
FOLIAGE_DENSITY = [
    # stratum, N_images, recall@0.25, precision@0.25, f1@0.25, crop_acc, fp_per_img, latency_ms
    ("Single Leaf (N=1)",       157, 49.04, 6.86, 12.03, 40.26, 6.66, 820.6),
    ("Moderate Foliage (N=2–4)", 60, 40.12, 17.91, 24.77, 38.81, 5.12, 938.1),
    ("Dense Foliage (N≥5)",      19, 21.88, 22.22, 22.05, 32.14, 5.16, 743.5),
]

# ── Phase 4.1 Candidate-Generation Ablation ───────────────────────────────────
CANDIDATE_ABLATION = [
    # name, latency_ms, recall@0.10, precision@0.10, recall@0.25, precision@0.25, f1@0.25
    ("Config A – Phase 4 Baseline",         175.0, 40.52, 25.12, 20.07, 12.44, 15.36),
    ("Config B – Multi-Scale Windows",      444.6, 50.19, 22.61, 31.23, 14.07, 19.40),
    ("Config C – Aspect-Ratio Aware",       838.1, 53.16, 22.17, 33.83, 14.11, 19.91),
    ("Config D – Foliar Saliency Filter",   443.6, 50.19, 23.40, 31.23, 14.56, 19.86),
    ("Config E – Optimized Composite",      863.4, 53.90, 19.44, 32.71, 11.80, 17.34),
]

# ── Phase 4.1 NMS Ablation ────────────────────────────────────────────────────
NMS_ABLATION = [
    # strategy, class_iou, cross_iou, max_cand, recall@0.25, precision@0.25, f1@0.25, fp_per_img
    ("Class-wise IoU 0.55 (Baseline)",       0.55, None, 8,  30.86, 12.83, 18.12, 5.64),
    ("Class-wise IoU 0.45",                  0.45, None, 8,  31.23, 13.04, 18.40, 5.60),
    ("Class-wise IoU 0.65",                  0.65, None, 8,  29.37, 12.10, 17.14, 5.74),
    ("Hybrid (Class 0.55, Cross 0.75, Max 8)",0.55, 0.75, 8, 30.86, 12.89, 18.18, 5.61),
    ("Hybrid (Class 0.50, Cross 0.70, Max 8)",0.50, 0.70, 8, 30.11, 12.62, 17.78, 5.61),
    ("Hybrid (Class 0.50, Cross 0.70, Max 10) [Selected]", 0.50, 0.70, 10, 34.57, 11.94, 17.75, 6.86),
    ("Hybrid (Class 0.50, Cross 0.70, Max 12)",0.50, 0.70, 12, 39.03, 11.50, 17.77, 8.08),
    ("Aggressive Cross-Class (0.45/0.60/8)", 0.45, 0.60, 8,  31.60, 13.28, 18.70, 5.55),
]

# ── Phase 5: Architecture Comparison ─────────────────────────────────────────
ARCH_COMPARISON = [
    # arch, params_M, size_mb, val_acc, test_acc, test_bal_acc, macro_f1,
    # plantdoc_top1, plantdoc_top3, latency_ms, pipeline_ms, temp, post_ece
    ("MobileNetV3-Small", 1.56, 6.06,  95.14, 95.02, 95.06, 93.13, 38.98, 64.83,  9.42,  662.1, 0.9902,  9.94),
    ("MobileNetV3-Large", 4.25, 16.40, 96.38, 95.96, 95.97, 94.52, 41.53, 68.22, 16.87, 1585.2, 0.9240, 10.80),
    ("EfficientNet-B0",   4.06, 15.74, 96.10, 95.77, 95.38, 94.26, 34.32, 62.71, 24.58, 2457.4, 0.8846, 11.62),
    ("ResNet18",         11.20, 42.79, 95.93, 95.92, 95.60, 94.30, 41.10, 67.80, 25.54, 2025.9, 1.0392, 10.21),
]

# ── Phase 5: Calibration Comparison ──────────────────────────────────────────
CALIB_COMPARISON = [
    # arch, temp, pre_ece, post_ece, pre_nll, post_nll, pre_brier, post_brier
    ("MobileNetV3-Small", 0.9902, 10.28,  9.94, 0.2554, 0.2513, 0.0993, 0.0981),
    ("MobileNetV3-Large", 0.9240, 14.00, 10.80, 0.2755, 0.2341, 0.0986, 0.0854),
    ("EfficientNet-B0",   0.8846, 16.43, 11.62, 0.3162, 0.2542, 0.1156, 0.0945),
    ("ResNet18",          1.0392,  8.99, 10.21, 0.2133, 0.2277, 0.0809, 0.0851),
]

# ── Latency Breakdown ─────────────────────────────────────────────────────────
LATENCY_BREAKDOWN = [
    ("Window Proposal Generation (Multi-scale + Aspect-aware)", 0.04,  0.10, "NumPy Grid Sampling"),
    ("Foliar Saliency Pre-Filtering (Excess Green Index)",      0.00, 24.16, "NumPy Vectorized Array Ops"),
    ("Crop Extraction, Padding & Tensor Prep (EVAL_TRANSFORM)", 67.07, 358.19, "PIL Slicing & Torch Normalize"),
    ("MobileNetV3-Small Batch Crop Forward Inference",          99.87, 461.62, "PyTorch 2.15.0+cpu (8 threads)"),
    ("Hybrid Two-Stage NMS Post-Processing",                     0.04,  0.18, "Python Greedy IoU Suppression"),
    ("Total End-to-End Latency per Image",                     167.02, 844.25, "Complete Inference Pipeline"),
]

# ── Dataset Partitions ────────────────────────────────────────────────────────
DATASET = {
    "pv_source_images": 54305,
    "pv_cross_label_duplicates_excluded": 10,
    "pv_intra_class_exact_duplicates_removed": 21,
    "pv_total_exclusions": 31,
    "pv_eligible_partitioned": 54274,
    "classes": 38,
    "crops": 14,
    "train_pv": 37992,
    "plantdoc_train_added": 2159,
    "train_total": 40151,       # 37992 PV + 2159 PlantDoc
    "val": 5436,
    "calib": 5429,
    "holdout_test": 5417,
    "combined_manifest_total": 56433,
    "plantdoc_external_test": 236,
    "plantdoc_ground_truth_leaves": 452,
}

# ── Selected Phase 4.1 Configuration ─────────────────────────────────────────
SCREENING_CONFIG = {
    "scales": [0.25, 0.40, 0.65],
    "aspect_ratios": [[1.0, 1.0], [1.333, 1.0], [1.0, 1.333]],
    "stride_factor": 0.70,
    "min_crop_dim": 112,
    "context_padding": 0.12,
    "foliar_saliency": True,
    "excess_green_threshold": 10,
    "min_foliar_coverage": 0.08,
    "min_candidate_score": 0.55,
    "nms_class_iou": 0.50,
    "nms_cross_iou": 0.70,
    "max_candidates": 10,
}

# ── Calibration Discrepancy (Phase 3 vs Phase 5) ──────────────────────────────
CALIB_DISCREPANCY = {
    "phase3_temp": 0.5538,
    "phase3_post_ece_calib": 0.58,
    "phase3_post_ece_holdout": 0.74,
    "phase5_temp": 0.9902,
    "phase5_post_ece_test": 9.94,
    "same_checkpoint": True,
    "same_partition": True,
    "explanation": (
        "Phase 3 utilized evaluate_classifier.py optimizing log-temperature with L-BFGS "
        "(lr=0.1, max_iter=80, line_search_fn='strong_wolfe') on the calibration split, "
        "converging to T*=0.5538 and reducing ECE from 10.25% to 0.58% (0.74% on holdout). "
        "Phase 5 batch benchmarking utilized run_phase5_evaluation.py optimizing raw temperature "
        "initialized at T0=1.5 with L-BFGS (lr=0.01, max_iter=50, without line search), reporting "
        "a test-split ECE of 9.94% at T*=0.9902. While both scripts evaluated checkpoint 6190f54c... "
        "on partition f9ef8b9d..., the exact sensitivity of the loss landscape to these differing "
        "optimizer hyperparameters remains unverified under controlled ablation. Consequently, this "
        "discrepancy is documented transparently as an unresolved methodological variation between "
        "evaluation pipelines, rather than attributed definitively to optimizer failure."
    ),
}

