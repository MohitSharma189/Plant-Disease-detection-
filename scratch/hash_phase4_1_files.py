import hashlib
import glob
import os

files = sorted(glob.glob("reports/phase4_1/*.*"))
for p in files:
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    sz = os.path.getsize(p)
    print(f"{os.path.basename(p):<40} {sz:<8} {h}")
