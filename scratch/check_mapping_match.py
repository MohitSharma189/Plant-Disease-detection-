import json

with open("reports/generated/candidate_screening_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open("reports/generated/combined_split_manifest.json", "r", encoding="utf-8") as f:
    pv_manifest = json.load(f)
pv_classes = set(pv_manifest["class_names"])

with open("training/plantdoc_label_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

matched = 0
unmatched = []
for r in manifest["records"]:
    anns = r.get("leaf_annotations", [])
    if not anns:
        continue
    c = anns[0]["class"]
    mc = mapping.get(c, c)
    if mc in pv_classes:
        matched += 1
    else:
        unmatched.append((c, mc))

print(f"Matched: {matched}/{len(manifest['records'])}")
if unmatched:
    print("Sample unmatched:", unmatched[:5])

