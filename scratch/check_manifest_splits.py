import json
from collections import Counter

with open("reports/generated/combined_split_manifest.json", "r", encoding="utf-8") as f:
    m = json.load(f)

print("Class count:", len(m["class_names"]))
splits = Counter(r.get("split") for r in m["records"])
print("Split counts:", dict(splits))
sample = m["records"][0]
print("Sample record keys:", list(sample.keys()))
print("Sample record:", sample)

