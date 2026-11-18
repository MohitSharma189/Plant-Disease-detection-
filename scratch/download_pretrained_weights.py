import torchvision.models as m

print("Downloading pretrained weights for Phase 5 benchmark architectures...")

# 1. MobileNetV3-Large
print("Downloading MobileNetV3-Large weights...")
m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.DEFAULT)
print("MobileNetV3-Large ready.")

# 2. EfficientNet-B0
print("Downloading EfficientNet-B0 weights...")
m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.DEFAULT)
print("EfficientNet-B0 ready.")

# 3. ResNet-18
print("Downloading ResNet-18 weights...")
m.resnet18(weights=m.ResNet18_Weights.DEFAULT)
print("ResNet-18 ready.")

# 4. ShuffleNetV2
try:
    print("Downloading ShuffleNetV2 weights...")
    m.shufflenet_v2_x1_0(weights=m.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    print("ShuffleNetV2 ready.")
except Exception as e:
    print("ShuffleNetV2 download error:", e)

print("All pretrained weights successfully cached!")

