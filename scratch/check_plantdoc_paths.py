import json, os
from pathlib import Path

with open("reports/generated/candidate_screening_manifest.json", "r", encoding="utf-8") as f:
    m = json.load(f)

exist_count = 0
for r in m["records"]:
    p = Path(r["resolved_path"])
    if p.exists():
        exist_count += 1
print(f"Direct exist count: {exist_count}/{len(m['records'])}")

if exist_count < len(m["records"]):
    # check fallback
    print("Checking alternative locations...")
    alt_dirs = [
        Path("data/raw/PlantDoc_dataset/TEST"),
        Path("data/raw/PlantDoc-Dataset/TEST"),
        Path("data/processed/PlantVillage/../../raw/PlantDoc_dataset/TEST"),
    ]
    for d in alt_dirs:
        if d.exists():
            print("Found dir:", d)
            files = list(d.glob("*"))
            print("Files count in dir:", len(files))

