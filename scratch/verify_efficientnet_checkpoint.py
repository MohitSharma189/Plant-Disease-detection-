import hashlib, os

path = r"saved_models/experiments/architecture_benchmark/efficientnet_b0/best_model.pth"
with open(path, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()
size = os.path.getsize(path)
print(f"SHA256: {h}")
print(f"Size:   {size} bytes ({size/1024/1024:.2f} MB)")

