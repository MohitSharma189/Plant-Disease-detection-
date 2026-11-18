import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import torchvision

print("=== Phase 5 Pre-Execution Verification ===")

# 1. Verify Baseline Checkpoint
ckpt_path = Path("saved_models/experiments/mobilenet_v3_small_final/best_model.pth")
if not ckpt_path.is_file():
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    sys.exit(1)
ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
expected_ckpt_hash = "6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009"
print(f"MobileNetV3-Small checkpoint: {ckpt_path} ({ckpt_path.stat().st_size} bytes)")
print(f"SHA-256: {ckpt_hash}")
assert ckpt_hash == expected_ckpt_hash, f"Hash mismatch: expected {expected_ckpt_hash}, got {ckpt_hash}"
print("Baseline checkpoint verified 100%.")

# 2. Verify Manifest
manifest_path = Path("reports/generated/combined_split_manifest.json")
if not manifest_path.is_file():
    print(f"ERROR: Manifest not found at {manifest_path}")
    sys.exit(1)
manifest_bytes = manifest_path.read_bytes()
manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
expected_manifest_hash = "f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868"
print(f"\nDataset manifest: {manifest_path} ({manifest_path.stat().st_size} bytes)")
print(f"SHA-256: {manifest_hash}")
assert manifest_hash == expected_manifest_hash, f"Hash mismatch: expected {expected_manifest_hash}, got {manifest_hash}"

manifest_data = json.loads(manifest_bytes.decode("utf-8"))
partition_counts = {}
for r in manifest_data.get("records", []):
    p = r.get("split", "unknown")
    partition_counts[p] = partition_counts.get(p, 0) + 1

print("Partition counts:")
for p, c in sorted(partition_counts.items()):
    print(f"  {p}: {c:,} images")

assert partition_counts.get("train") == 40151, f"Expected 40,151 train, got {partition_counts.get('train')}"
assert partition_counts.get("validation") == 5436, f"Expected 5,436 validation, got {partition_counts.get('validation')}"
assert partition_counts.get("calibration") == 5429, f"Expected 5,429 calibration, got {partition_counts.get('calibration')}"
assert partition_counts.get("test") == 5417, f"Expected 5,417 test, got {partition_counts.get('test')}"
print("Manifest partitions verified 100%.")

# 3. Verify Hardware & Environment
print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CPU Threads: {torch.get_num_threads()}")

# 4. Verify E: Drive Free Space
import shutil
total, used, free = shutil.disk_usage("E:")
print(f"\nE: Drive Storage:")
print(f"  Total: {total / (1024**3):.1f} GB")
print(f"  Used:  {used / (1024**3):.1f} GB")
print(f"  Free:  {free / (1024**3):.1f} GB")
assert free / (1024**3) > 10.0, "Low disk space (< 10 GB free)"

print("\nAll Pre-Execution Checks Passed Successfully!")
