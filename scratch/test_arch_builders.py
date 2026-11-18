import torch
import torch.nn as nn
from torchvision import models

def test_archs():
    configs = [
        ("mobilenet_v3_large", models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        ("efficientnet_b0", models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        ("resnet18", models.resnet18, models.ResNet18_Weights.DEFAULT),
        ("shufflenet_v2", models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    ]
    
    dummy = torch.randn(2, 3, 224, 224)
    for name, builder, weights in configs:
        model = builder(weights=weights)
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                # find last linear
                for i in reversed(range(len(model.classifier))):
                    if isinstance(model.classifier[i], nn.Linear):
                        model.classifier[i] = nn.Linear(model.classifier[i].in_features, 38)
                        break
        elif hasattr(model, "fc"):
            if isinstance(model.fc, nn.Linear):
                model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 38))
        
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] Output shape: {out.shape}, Total params: {params:,}")
        assert out.shape == (2, 38)

test_archs()
print("All model architectures successfully validated!")

