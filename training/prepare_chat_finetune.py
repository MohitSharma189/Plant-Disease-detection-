"""Validate and split a chat fine-tuning JSONL dataset.

This prepares data for an external LoRA/QLoRA trainer. It does not fine-tune
model weights itself and never mixes image-classifier labels into chat data.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_ROLES = ("system", "user", "assistant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("training/chat_finetune_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/chat_finetune"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")
        record_id = record.get("id")
        category = record.get("category")
        messages = record.get("messages")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in seen_ids:
            raise ValueError(f"Line {line_number} has a missing or duplicate id.")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Line {line_number} has no category.")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Line {line_number} must contain at least system, user, and assistant messages.")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles[:3] != list(REQUIRED_ROLES):
            raise ValueError(f"Line {line_number} must begin with roles {REQUIRED_ROLES}.")
        if len(roles) != len(messages) or any(
            not isinstance(message.get("content"), str) or not message["content"].strip()
            for message in messages
            if isinstance(message, dict)
        ):
            raise ValueError(f"Line {line_number} contains an invalid message.")
        if any(role not in {"system", "user", "assistant"} for role in roles):
            raise ValueError(f"Line {line_number} contains an unsupported role.")
        seen_ids.add(record_id)
        records.append(record)
    if len(records) < 10:
        raise ValueError("The dataset must contain at least 10 examples.")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.validation_fraction <= 0.4:
        raise SystemExit("--validation-fraction must be between 0.1 and 0.4.")
    records = load_records(args.input)
    random.Random(args.seed).shuffle(records)
    validation_count = max(1, round(len(records) * args.validation_fraction))
    validation = records[:validation_count]
    train = records[validation_count:]
    if not train:
        raise SystemExit("The split produced no training examples.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    summary = {
        "input": str(args.input),
        "seed": args.seed,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "categories": dict(Counter(record["category"] for record in records)),
        "warning": "This is prepared data for an external LoRA/QLoRA trainer; it is not model-weight training.",
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
