import torchvision.models as m
import torch
import torch.nn as nn

print("Checking weights availability...")
# 1. MobileNetV3-Small
m_s = m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)
print("MobileNetV3-Small loaded successfully. Params:", sum(p.numel() for p in m_s.parameters()))

# 2. MobileNetV3-Large
m_l = m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.DEFAULT)
print("MobileNetV3-Large loaded successfully. Params:", sum(p.numel() for p in m_l.parameters()))

# 3. EfficientNet-B0
eff = m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.DEFAULT)
print("EfficientNet-B0 loaded successfully. Params:", sum(p.numel() for p in eff.parameters()))

# 4. ResNet-18
res = m.resnet18(weights=m.ResNet18_Weights.DEFAULT)
print("ResNet-18 loaded successfully. Params:", sum(p.numel() for p in res.parameters()))

# 5. ShuffleNetV2
try:
    shuf = m.shufflenet_v2_x1_0(weights=m.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    print("ShuffleNetV2 loaded successfully. Params:", sum(p.numel() for p in shuf.parameters()))
except Exception as e:
    print("ShuffleNetV2 weights failed or need download:", e)

