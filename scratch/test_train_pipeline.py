import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from training.train_benchmark_architecture import ManifestDataset, build_model

manifest = json.load(open("reports/generated/combined_split_manifest.json"))
class_names = manifest["class_names"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
records = manifest["records"]
train_records = [record for record in records if record["split"] == "train"]

t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
ds = ManifestDataset(Path("data/processed/PlantVillage").resolve(), train_records[:20], class_to_idx, t)
img, lbl = ds[0]
print("Item 0 loaded successfully:", img.shape, lbl)

# Check PlantDoc records
pd_recs = [r for r in train_records if "dataset_root" in r]
ds_pd = ManifestDataset(Path("data/processed/PlantVillage").resolve(), pd_recs[:10], class_to_idx, t)
img_pd, lbl_pd = ds_pd[0]
print(f"PlantDoc item 0 loaded successfully:", img_pd.shape, lbl_pd)

model, head_names = build_model("mobilenet_v3_large", len(class_names), 0.3)
out = model(img.unsqueeze(0))
print("Model forward pass successful. Logits shape:", out.shape)
print("All dataset and model pipeline tests passed 100%!")
