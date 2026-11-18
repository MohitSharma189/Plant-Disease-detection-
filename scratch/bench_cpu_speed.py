import time
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

print("Testing MobileNetV3-Large throughput on CPU...")
device = torch.device("cpu")
torch.set_num_threads(8)

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 38)
model.to(device)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()

dummy_x = torch.randn(8, 3, 224, 224)
dummy_y = torch.randint(0, 38, (8,))

# Warmup
for _ in range(5):
    opt.zero_grad()
    crit(model(dummy_x), dummy_y).backward()
    opt.step()

t0 = time.time()
n_steps = 20
for _ in range(n_steps):
    opt.zero_grad()
    out = model(dummy_x)
    loss = crit(out, dummy_y)
    loss.backward()
    opt.step()
dt = time.time() - t0
fps = (n_steps * 8) / dt
sec_per_step = dt / n_steps
print(f"Batch size 8: {sec_per_step:.3f} s/step, {fps:.1f} images/s")
est_epoch_min = (5019 * sec_per_step) / 60
print(f"Estimated 1 epoch (40,151 images): {est_epoch_min:.1f} minutes")

