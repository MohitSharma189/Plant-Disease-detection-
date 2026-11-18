import json

with open("reports/generated/candidate_screening_manifest.json", "r", encoding="utf-8") as f:
    m = json.load(f)

print("Plantdoc records:", len(m["records"]))
r = m["records"][0]
print("Keys:", list(r.keys()))
print("resolved_path:", r.get("resolved_path"))
print("leaf_annotations count:", len(r.get("leaf_annotations", [])))
if r.get("leaf_annotations"):
    print("first ann:", r["leaf_annotations"][0])

