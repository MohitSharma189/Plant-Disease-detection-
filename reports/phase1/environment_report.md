# Environment Setup & Hardware Verification Report (Phase 1)

**Execution Date:** 2026-11-12  
**Virtual Environment Path:** `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\.venv`  
**Execution Architecture:** CPU-only (No CUDA/GPU)

---

## 1. Hardware Specifications

The environment configuration is aligned with the host workstation hardware profile:

| Hardware Component | Workstation Profile | Environment Configuration |
| :--- | :--- | :--- |
| **Processor (CPU)** | AMD Ryzen 7 5800HS @ 3.20 GHz (8 cores, 16 threads) | PyTorch configured with CPU execution (`torch.device("cpu")`), threads set to available cores |
| **RAM** | 16 GB Physical DDR4 (15.4 GB usable) | Conservative memory policy: batch size 8, streaming dataset loaders, no full-RAM caching |
| **Graphics (GPU)** | AMD Radeon(TM) Integrated Graphics (~496 MB VRAM) | **CPU-Only Execution**. No CUDA runtime, CUDA PyTorch, or ROCm installed |
| **Operating System** | Windows 11 Home / Pro 64-bit (x86_64) | PowerShell / CMD subprocess management with long-path support |

---

## 2. Python Runtime & Virtual Environment

| Property | Details |
| :--- | :--- |
| **Python Version** | Python 3.12.5 (`v3.12.5:ff3bc82`, MSC v.1940 64 bit AMD64) |
| **Host Python Path** | `C:\Users\Mohit Sharma\AppData\Local\Programs\Python\Python312\python.exe` |
| **Virtualenv Python Path** | `E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\.venv\Scripts\python.exe` |
| **Pre-existing State** | `.venv` directory was empty (0 files). Preserved and populated with fresh 3.12.5 venv |
| **Pip Version** | Pip 24.2 |

---

## 3. Verified Dependency Manifest

All required project and research dependencies were installed using CPU-only wheels from official repositories:

| Package Category | Package Name | Installed Version | Verification Status |
| :--- | :--- | :--- | :--- |
| **Deep Learning & Tensors** | `torch` | `2.15.0+cpu` | Verified CPU-only (`torch.cuda.is_available() == False`) |
| **Computer Vision** | `torchvision` | `0.29.2+cpu` | Verified (`torchvision.models.mobilenet_v3_small` imports) |
| **Image Processing** | `opencv-python-headless` | `5.0.0.93` | Verified (`cv2.__version__ == '5.0.0'`) |
| **Image I/O** | `Pillow` | `13.0.0` | Verified RGB & Lanczos support |
| **Numerical Computing** | `numpy` | `2.5.4` | Verified |
| **Data Analysis & Tabular** | `pandas` | `3.1.1` | Verified DataFrame & CSV operations |
| **Machine Learning & Metrics**| `scikit-learn` | `1.9.1` | Verified (confusion matrix, PRF metrics, ECE) |
| **Plotting & Visualization** | `matplotlib` | `3.11.2` | Verified headless Agg backend |
| **Statistical Visualization** | `seaborn` | `0.13.2` | Verified |
| **Web API Framework** | `fastapi` | `0.162.4` | Verified |
| **ASGI Web Server** | `uvicorn[standard]` | `0.54.0` | Verified with watchfiles & httptools |
| **Configuration & Secrets** | `pydantic-settings` | `2.16.2` | Verified |
| **Configuration Formats** | `PyYAML` | `6.0.3` | Verified |
| **Testing Harness** | `pytest` | `9.2.0` | Verified (unit tests passing) |
| **Dataset Utilities** | `datasets` | `5.2.0` | Verified |
| **Model Hub Hub Utilities** | `huggingface-hub` | `2.4.0` | Verified |
| **PDF Generation** | `reportlab` | `5.1.0` | Verified |
| **HTTP Client** | `httpx` | `0.28.1` | Verified |

*Full lockfile preserved at:* `Plant Disease Detection/requirements.lock.txt`

---

## 4. PyTorch CPU Execution & Model Verification Test

A programmatic check executed inside `.venv` produced the following runtime log:

```text
Python: 3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)]
PyTorch: 2.15.0+cpu | CUDA available: False | CPU count: 8
Torchvision: 0.29.2+cpu
MobileNetV3-Small initialized successfully: MobileNetV3
OpenCV: 5.0.0
NumPy: 2.5.4
Pandas: 3.1.1
Scikit-learn: 1.9.1
Matplotlib: 3.11.2
Seaborn: 0.13.2
Pillow: 13.0.0
FastAPI: 0.162.4
Uvicorn: 0.54.0
PyYAML: 6.0.3
pytest: 9.2.0
```

### Test Suite Execution
- **Screening & Quality Tests:** 13 passed in 5.96s (`test_screening.py`, `test_measurements.py`, `test_quality.py`, `test_models.py`).
- **Data Preparation Tests:** 3 passed in 10.41s (`test_prepare_splits.py`, `test_prepare_combined_manifest.py`, `test_evaluate_plantdoc.py`).

