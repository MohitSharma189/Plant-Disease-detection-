import json
m = json.load(open("reports/generated/combined_split_manifest.json"))
pd = [r for r in m["records"] if "dataset_root" in r]
print("Records with dataset_root:", len(pd))
if pd:
    print("Example:", pd[0])
pv = [r for r in m["records"] if "dataset_root" not in r]
print("Records without dataset_root:", len(pv))
if pv:
    print("Example PV:", pv[0])

