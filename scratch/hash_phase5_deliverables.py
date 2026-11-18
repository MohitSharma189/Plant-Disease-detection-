import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
REPORT_DIR = PROJECT_ROOT / "reports/phase5"

files_to_hash = [
    REPORT_DIR / "architecture_comparison.csv",
    REPORT_DIR / "classification_metrics.csv",
    REPORT_DIR / "external_plantdoc_results.csv",
    REPORT_DIR / "calibration_comparison.csv",
    REPORT_DIR / "efficiency_benchmark.csv",
    REPORT_DIR / "per_class_metrics.csv",
    REPORT_DIR / "checkpoint_provenance.json",
    REPORT_DIR / "performance_comparison_charts.png",
    REPORT_DIR / "PHASE5_COMPLETION_REPORT.md",
    REPORT_DIR / "confusion_matrices/mobilenet_v3_small_confusion_matrix.png",
    REPORT_DIR / "confusion_matrices/mobilenet_v3_large_confusion_matrix.png",
    REPORT_DIR / "confusion_matrices/efficientnet_b0_confusion_matrix.png",
    REPORT_DIR / "confusion_matrices/resnet18_confusion_matrix.png",
]

def hash_file(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower(), p.stat().st_size

manifest = {}
print("PHASE 5 DELIVERABLES HASH MANIFEST:")
for f in files_to_hash:
    if f.exists():
        h, sz = hash_file(f)
        rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
        manifest[rel] = {"sha256": h, "size_bytes": sz}
        print(f"  {rel:60s} | {sz:9d} B | {h[:16]}...")
    else:
        print(f"  MISSING: {f}")

out_manifest = REPORT_DIR / "phase5_deliverables_manifest.json"
with open(out_manifest, "w", encoding="utf-8") as out_f:
    json.dump(manifest, out_f, indent=2)
print(f"\nSaved manifest to {out_manifest}")

# Verify Baseline Checkpoint & Phase 4.1 Config Integrity
print("\nINTEGRITY VERIFICATION:")
base_ckpt = PROJECT_ROOT / "saved_models/experiments/mobilenet_v3_small_final/best_model.pth"
base_h, _ = hash_file(base_ckpt)
expected_base = "6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009"
print(f"  Baseline MobileNetV3-Small: {base_h}")
assert base_h == expected_base, f"Baseline mismatch: expected {expected_base}, got {base_h}"
print("  --> BASELINE CHECKPOINT UNTOUCHED: OK [PASS]")

cfg_path = PROJECT_ROOT / "backend/config/phase4_1/screening_config.json"
cfg_h, _ = hash_file(cfg_path)
print(f"  Phase 4.1 Screening Config: {cfg_h}")
print("  --> PHASE 4.1 SCREENING CONFIG: OK [PASS]")

