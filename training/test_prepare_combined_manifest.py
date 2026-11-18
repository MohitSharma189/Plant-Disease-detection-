import json

import numpy as np
from PIL import Image

from training.prepare_combined_manifest import build_combined_manifest


def _write_image(path, seed):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def test_combined_manifest_adds_only_deduplicated_plantdoc_training_images(tmp_path):
    pv_dir = tmp_path / "processed" / "PlantVillage"
    field_dir = tmp_path / "raw" / "PlantDoc_dataset"
    records = []
    duplicate_source = None
    for label, base_seed in (("class_a", 10), ("class_b", 30)):
        for index, split in enumerate(("train", "validation", "calibration", "test")):
            path = pv_dir / label / f"{index}.png"
            _write_image(path, base_seed + index)
            if label == "class_a" and split == "test":
                duplicate_source = path
            records.append(
                {
                    "path": f"{label}/{index}.png",
                    "label": label,
                    "split": split,
                    "duplicate_group": len(records),
                }
            )

    field_train = field_dir / "train" / "Field class" / "new.png"
    _write_image(field_train, 99)
    field_duplicate = field_dir / "train" / "Field class" / "duplicate.png"
    field_duplicate.parent.mkdir(parents=True, exist_ok=True)
    field_duplicate.write_bytes(duplicate_source.read_bytes())
    field_test = field_dir / "test" / "Field class" / "test.png"
    _write_image(field_test, 100)

    plantvillage_manifest = {
        "seed": 42,
        "class_names": ["class_a", "class_b"],
        "split_fractions": {
            "train": 0.7,
            "validation": 0.1,
            "calibration": 0.1,
            "test": 0.1,
        },
        "records": records,
    }
    plantdoc_manifest = {
        "source": "https://example.test/PlantDoc",
        "commit": "abc123",
        "license": "CC BY 4.0",
        "records": [
            {
                "source_path": "train/Field class/new.png",
                "path": "train/Field class/new.png",
                "split": "train",
                "class_name": "Field class",
            },
            {
                "source_path": "train/Field class/duplicate.png",
                "path": "train/Field class/duplicate.png",
                "split": "train",
                "class_name": "Field class",
            },
            {
                "source_path": "test/Field class/test.png",
                "path": "test/Field class/test.png",
                "split": "test",
                "class_name": "Field class",
            },
        ],
    }

    hash_cache = tmp_path / "combined_hash_cache.json"
    combined = build_combined_manifest(
        pv_dir,
        plantvillage_manifest,
        field_dir,
        plantdoc_manifest,
        {"Field class": "class_a"},
        hash_cache_path=hash_cache,
    )
    cached_combined = build_combined_manifest(
        pv_dir,
        plantvillage_manifest,
        field_dir,
        plantdoc_manifest,
        {"Field class": "class_a"},
        hash_cache_path=hash_cache,
    )

    assert cached_combined == combined
    assert hash_cache.is_file()
    assert combined["source_counts"]["plantdoc_train_added"] == 1
    assert sum(combined["counts_by_split_and_class"]["test"].values()) == 2
    assert all(
        record["split"] != "test" or record.get("source") != "PlantDoc"
        for record in combined["records"]
    )
    added = [record for record in combined["records"] if record.get("source") == "PlantDoc"]
    assert added[0]["label"] == "class_a"
    assert (pv_dir / added[0]["dataset_root"] / added[0]["path"]).is_file()
    assert len(
        combined["excluded_plantdoc_records"][
            "plantdoc_duplicate_or_conflicting_label"
        ]
    ) == 1
    json.dumps(combined)


def test_combined_manifest_excludes_plantdoc_test_leakage(tmp_path):
    pv_dir = tmp_path / "processed" / "PlantVillage"
    field_dir = tmp_path / "raw" / "PlantDoc_dataset"

    # Minimal PlantVillage record
    pv_img = pv_dir / "class_a" / "pv0.png"
    _write_image(pv_img, 1)
    plantvillage_manifest = {
        "seed": 42,
        "class_names": ["class_a"],
        "split_fractions": {"train": 1.0, "validation": 0.0, "calibration": 0.0, "test": 0.0},
        "records": [{"path": "class_a/pv0.png", "label": "class_a", "split": "train", "duplicate_group": 0}],
    }

    # Official PlantDoc test image
    field_test = field_dir / "test" / "Field class" / "official_test.png"
    _write_image(field_test, 555)

    # 1. Exact duplicate of test image in train
    field_train_exact = field_dir / "train" / "Field class" / "exact_leak.png"
    field_train_exact.parent.mkdir(parents=True, exist_ok=True)
    field_train_exact.write_bytes(field_test.read_bytes())

    # 2. Clean train image
    field_train_clean = field_dir / "train" / "Field class" / "clean_train.png"
    _write_image(field_train_clean, 777)

    plantdoc_manifest = {
        "source": "https://example.test/PlantDoc",
        "commit": "test123",
        "license": "CC BY 4.0",
        "records": [
            {
                "source_path": "test/Field class/official_test.png",
                "path": "test/Field class/official_test.png",
                "split": "test",
                "class_name": "Field class",
            },
            {
                "source_path": "train/Field class/exact_leak.png",
                "path": "train/Field class/exact_leak.png",
                "split": "train",
                "class_name": "Field class",
            },
            {
                "source_path": "train/Field class/clean_train.png",
                "path": "train/Field class/clean_train.png",
                "split": "train",
                "class_name": "Field class",
            },
        ],
    }

    combined = build_combined_manifest(
        pv_dir,
        plantvillage_manifest,
        field_dir,
        plantdoc_manifest,
        {"Field class": "class_a"},
    )

    # Verify clean image was added, but exact test leakage was excluded
    assert combined["source_counts"]["plantdoc_train_added"] == 1
    assert len(combined["excluded_plantdoc_records"]["plantdoc_duplicate_of_plantdoc_test"]) == 1
    leak_record = combined["excluded_plantdoc_records"]["plantdoc_duplicate_of_plantdoc_test"][0]
    assert leak_record["path"] == "train/Field class/exact_leak.png"
    assert leak_record["matching_test_path"] == "test/Field class/official_test.png"
