# Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps

A CPU-first full-stack research application integrating the verified Phase 5 **MobileNetV3-Small TorchScript** classifier with an optimized two-stage candidate-region foliar screening pipeline.

---

## 1. Research Overview & System Status

This application is the research prototype developed to evaluate lightweight convolutional neural networks for foliar disease screening under computational constraints.

* **Primary Classifier:** MobileNetV3-Small (TorchScript container)
* **Active Checkpoint:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`
* **Verified SHA-256 Digest:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`
* **Training Data:** In-domain PlantVillage holdout corpus ($54,274\text{ images}$ across 38 classes) combined with select in-domain PlantDoc crops.
* **Inference Mode:** Uncalibrated raw softmax scores ($T = 1.0$, labeled `"Uncalibrated Model Score"`).
* **Operational Review Threshold:** $0.60$ ($60\%$ score threshold for whole-leaf human expert escalation).
* **Candidate Screening Threshold:** $0.60$ (`MIN_CANDIDATE_SCORE` for localized foliar crop proposals).
* **Execution Environment:** Windows 11 CPU mode (no GPU, CUDA, or cloud coprocessor required).

> [!CAUTION]
> **Clinical & Field Diagnosis Notice:** This application is an **exploratory screening tool**, not an autonomous diagnostic instrument or clinical plant pathology certificate.
> 1. A low-confidence prediction of "healthy" is **not proof of health**.
> 2. The absence of candidate screening regions is **not proof that a plant is disease-free**.
> 3. Grad-CAM heatmaps represent **model-attention / saliency attribution**, not verified biological lesion masks.

---

## 2. Architecture & Two-Stage Screening Design

```text
               [User Browser / Device]
                │
                ├─► Image Upload / Camera Capture / Client Validation
                │
                ▼ (HTTP POST /screen-plant)
        [FastAPI Backend - Port 8000]
                │
    ┌───────────┴─────────────────────────────┐
    │                                         │
    ▼                                         ▼
[Section A: Whole-Leaf]            [Section B: Candidate-Region]
• Full-image resize (224x224)      • Dynamic Multi-scale Sliding Grid
• 38-class MobileNetV3-Small       • 9 crops (leaf) / 12–24 crops (canopy)
• Score & State Determination      • Foliar saliency & threshold filtering
  (HIGH_CONFIDENCE vs              • Greedy IoU NMS (IoU >= 0.55)
   LOW_CONFIDENCE_UNCERTAIN)       • Grad-CAM saliency attribution
• Healthy Safety Alert Gate        • Bounding-box overlay generation
    │                                         │
    └───────────┬─────────────────────────────┘
                │
                ▼ (Structured JSON Response)
        [React / Vite Frontend - Port 5173]
        • Section A: Whole-Leaf Classification Card
        • Section B: Foliar Candidate Bounding Overlays
        • Guided Follow-Up Q&A & Research Links
```

---

## 3. Technology Stack

* **Frontend:** React 18, Vite 6, Native Canvas & CSS3 (Agricultural calm theme, no external UI component bloat).
* **Backend:** FastAPI, Starlette ASGI, Pydantic v2.
* **Computer Vision & Inference:** PyTorch 2.15.0+cpu, TorchScript JIT, OpenCV 4.10, Pillow 10.4.
* **Testing & Verification:** PyTest, Starlette TestClient, NumPy.

---

## 4. Installation and Local Setup

### Prerequisites
* Windows 10/11 (or Linux / macOS)
* Python 3.11 or 3.12
* Node.js v18+ and npm

### Step 1: Set Up Backend Virtual Environment
From the project root:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

### Step 2: Set Up Frontend
```powershell
cd frontend
npm install
cd ..
```

### Step 3: Configure Environment
Copy the environment template:
```powershell
Copy-Item .env.example .env
```
*(Optional: configure OpenAI-compatible provider credentials in `.env` if multilingual follow-up Q&A is required; core classification and candidate screening operate 100% offline).*

---

## 5. Running the Application

### Start Backend Service (FastAPI)
In Terminal 1 (from project root with `.venv` active):
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Health verification:* Open `http://127.0.0.1:8000/health` (should return `{"status": "ok", "inference_ready": true}`).

### Start Frontend Application (Vite)
In Terminal 2:
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173` in any modern web browser.

---

## 6. Verification and Testing Suite

To re-run the complete verification, validation, and latency benchmark suite:

```powershell
# 1. Run backend unit and regression tests (44 tests)
.\.venv\Scripts\pytest.exe backend/tests

# 2. Run the end-to-end full-stack smoke test
.\.venv\Scripts\python.exe test_final_smoke.py

# 3. Run Phase 6A.4 Full-Stack Scenario Suite (12 scenarios)
.\.venv\Scripts\python.exe test_phase6a_4_fullstack.py

# 4. Run Phase 6A.5 Comprehensive Robustness & Benchmark Suite (50-run timing)
.\.venv\Scripts\python.exe run_phase6a_5_suite.py
```

---

## 7. Performance Highlights (AMD Ryzen 7 5800HS CPU)

* **Pure Model Forward Pass:** **$8.43 \pm 1.16\text{ ms}$** ($p95 = 10.77\text{ ms}$)
* **Whole-Leaf Single Image API (`/predict`):** **$27.84 \pm 0.77\text{ ms}$** ($p95 = 29.41\text{ ms}$)
* **Two-Stage Single-Leaf Screening (`/screen-plant`):** **$186.24 \pm 11.52\text{ ms}$** ($p95 = 197.99\text{ ms}$)
* **Multi-Crop Whole-Plant Screening (`/screen-plant`):** **$200.03 \pm 17.63\text{ ms}$** ($p95 = 244.75\text{ ms}$)

---

## 8. Documentation Index

Detailed engineering and operational documentation is maintained in the `docs/` and `reports/` directories:

* [docs/USER_GUIDE.md](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/docs/USER_GUIDE.md) — Step-by-step instructions for uploading specimens, selecting screening modes, interpreting scores, and navigating uncertainty alerts.
* [docs/TECHNICAL_DOCUMENTATION.md](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/docs/TECHNICAL_DOCUMENTATION.md) — Comprehensive architecture specifications, API schemas, candidate-region proposal algorithms, and NMS mathematics.
* [docs/KNOWN_LIMITATIONS.md](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/docs/KNOWN_LIMITATIONS.md) — Thorough audit of validation gaps, cross-domain degradation, rotation sensitivity, and lighting failure modes.
* [reports/phase6a/](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase6a/) — Full collection of formal verification reports from Phase 6A.1 to Phase 6A.6.
