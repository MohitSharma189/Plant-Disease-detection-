import time
import torch
import torchvision.models as models

device = torch.device("cpu")
torch.set_num_threads(8)

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 38)
model.to(device)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()

for bs in [8, 16, 32]:
    dummy_x = torch.randn(bs, 3, 224, 224)
    dummy_y = torch.randint(0, 38, (bs,))
    for _ in range(3):
        opt.zero_grad()
        crit(model(dummy_x), dummy_y).backward()
        opt.step()
    t0 = time.time()
    n = 10
    for _ in range(n):
        opt.zero_grad()
        crit(model(dummy_x), dummy_y).backward()
        opt.step()
    dt = time.time() - t0
    fps = (n * bs) / dt
    sec_step = dt / n
    steps_per_epoch = 40151 // bs
    est_min = (steps_per_epoch * sec_step) / 60
    print(f"BS={bs:2d}: {sec_step:.3f} s/step | {fps:5.1f} img/s | Est epoch: {est_min:5.1f} min")

