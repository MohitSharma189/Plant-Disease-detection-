"""Restore a previously validated classifier backup after explicit confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch


MODEL_FILES = ("classifier.pt", "class_mapping.json", "classifier_metadata.json")


def validate_model_directory(directory: Path) -> list[str]:
    missing = [filename for filename in MODEL_FILES if not (directory / filename).is_file()]
    if missing:
        raise ValueError(f"Model directory is missing required files: {', '.join(missing)}")
    try:
        class_names = json.loads((directory / "class_mapping.json").read_text(encoding="utf-8"))
        metadata: dict[str, Any] = json.loads(
            (directory / "classifier_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model metadata from {directory}.") from exc
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("class_mapping.json must contain unique, non-empty class names.")
    if metadata.get("classes") is not None and metadata["classes"] != class_names:
        raise ValueError("Checkpoint metadata classes do not match class_mapping.json.")
    try:
        image_size = int(metadata.get("image_size", 224))
        if image_size < 32:
            raise ValueError("image_size must be at least 32.")
        model = torch.jit.load(str(directory / "classifier.pt"), map_location="cpu")
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, image_size, image_size))
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Classifier checkpoint could not be loaded from {directory}.") from exc
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != (1, len(class_names))
        or not torch.isfinite(output).all()
    ):
        raise ValueError("Classifier output does not match the saved class mapping.")
    return class_names


def rollback_model(backup_dir: Path, model_dir: Path, confirm: bool = False) -> Path:
    if not confirm:
        raise ValueError("Rollback requires explicit confirmation.")
    backup_dir = backup_dir.resolve()
    model_dir = model_dir.resolve()
    if backup_dir == model_dir:
        raise ValueError("The backup directory must differ from the production model directory.")
    validate_model_directory(backup_dir)
    validate_model_directory(model_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    current_backup = model_dir / f"baseline_backup_before_rollback_{stamp}"
    current_backup.mkdir(parents=False, exist_ok=False)
    for filename in MODEL_FILES:
        shutil.copy2(model_dir / filename, current_backup / filename)

    staging = Path(tempfile.mkdtemp(prefix=".rollback_staging_", dir=model_dir))
    try:
        for filename in MODEL_FILES:
            shutil.copy2(backup_dir / filename, staging / filename)
        for filename in MODEL_FILES:
            os.replace(staging / filename, model_dir / filename)
    except Exception:
        recovery = Path(tempfile.mkdtemp(prefix=".rollback_recovery_", dir=model_dir))
        try:
            for filename in MODEL_FILES:
                shutil.copy2(current_backup / filename, recovery / filename)
            for filename in MODEL_FILES:
                os.replace(recovery / filename, model_dir / filename)
        finally:
            shutil.rmtree(recovery, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return current_backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--confirm", action="store_true", help="Confirm replacing the selected production model.")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required to replace the production model.")
    try:
        previous = rollback_model(args.backup_dir, args.model_dir, confirm=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"restored_from": str(args.backup_dir), "previous_model_backup": str(previous)}, indent=2))


if __name__ == "__main__":
    main()