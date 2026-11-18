# Verdant Field Laboratory: Technical Documentation

**System:** Plant Disease Screening System with MobileNetV3-Small  
**Phase:** 6A Release Specification  
**Architecture:** CPU-First Decoupled Full-Stack Web Application  

---

## 1. System Architecture Overview

The system is architected as an asynchronous, decoupled full-stack application optimized for low-latency CPU inference without specialized acceleration hardware:

```text
[Client: React 18 + Vite 6 (Port 5173)]
   │
   ├─► Client Validation (Format, Size, Natural Dimensions)
   │
   ▼ (HTTP / JSON / Multipart Form-Data)
[Server: FastAPI / Starlette ASGI (Port 8000)]
   │
   ├─► App Middleware (CORS: localhost:5173, 127.0.0.1:5173)
   │
   ├─► Image Processing Module (OpenCV & Pillow)
   │    ├─ JPEG/PNG byte stream decoding
   │    ├─ EXIF orientation normalization
   │    └─ Laplacian variance blur score estimation
   │
   ├─► Model Inference Engine (PyTorch 2.15.0+cpu)
   │    ├─ Checkpoint: MobileNetV3-Small TorchScript
   │    ├─ Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   │    └─ Softmax probability projection (T=1.0)
   │
   ├─► Two-Stage Screening Pipeline
   │    ├─ Multi-scale sliding window proposal generator
   │    ├─ Foliar saliency filter (Excess Green ExG = 2G - R - B)
   │    ├─ Greedy IoU Non-Maximum Suppression (IoU >= 0.55)
   │    ├─ Saliency Grad-CAM feature attribution
   │    └─ Base64 annotated overlay composer
   │
   └─► Knowledge & Advisory Layer
        ├─ Structured agronomic guidance & institutional source mapping
        ├─ Europe PMC / PubMed scientific publication search
        └─ Optional OpenAI-compatible A2A chat proxy
```

---

## 2. Model Integration & Checkpoint Provenance

### 2.1 Model Specifications
* **Architecture:** MobileNetV3-Small (Hardswish / Squeeze-and-Excitation inverted bottleneck blocks).
* **Checkpoint Path:** `saved_models/experiments/mobilenet_v3_small_final/classifier.pt`
* **Verified SHA-256 Digest:** `6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213`
* **Parameters:** $1.56\text{M}$ parameters ($6.06\text{ MB}$ uncompressed TorchScript archive).
* **Classifier Head:** Linear layer projecting 576-dimensional pooled features into 38 class logits.

### 2.2 Preprocessing Transforms
All image tensors passed to MobileNetV3-Small follow the exact pipeline:
$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}_{\text{resized}} / 255.0 - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
Where:
* Size: $224 \times 224\text{ pixels}$ (Bicubic interpolation).
* $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$
* $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$

### 2.3 Image Quality Assessment
Uploaded images are evaluated for focal sharpness using the variance of the discrete Laplacian operator:
$$\text{blur\_score} = \operatorname{Var}\left(\nabla^2 I_{\text{gray}}\right)$$
Images with $\text{blur\_score} < 25.0$ are flagged as `is_blurry = True`, triggering frontend sharpness advisories.

---

## 3. Prediction-State & Safety Logic

The backend maps raw model logits into discrete operational prediction states:

```python
probs = torch.softmax(logits / temperature, dim=1)
confidence, pred_idx = torch.max(probs, dim=1)
predicted_class = class_names[pred_idx]
is_uncertain = confidence < review_threshold  # review_threshold = 0.60

if is_uncertain:
    prediction_state = "LOW_CONFIDENCE_UNCERTAIN"
    prediction_status = "Uncertain Prediction"
    state_tone = "amber"
    needs_review = True
else:
    prediction_state = "HIGH_CONFIDENCE"
    prediction_status = "Model Prediction"
    state_tone = "good"
    needs_review = False
```

### Safety Flags:
* `is_healthy`: Evaluated via class name substring `cond_clean.lower() == "healthy"`.
* When `is_healthy == True` and `is_uncertain == True`, the frontend renders the `.healthy-warning-alert`, preventing users from assuming an inconclusive prediction implies disease absence.

---

## 4. Two-Stage Screening Pipeline

### 4.1 Proposal Generation

| Parameter | Single-Leaf Mode (`single_leaf`) | Whole-Plant Mode (`whole_plant`) |
| :--- | :---: | :---: |
| **Operational Goal** | Symptom verification on close-up leaf | Multi-foliage canopy scan |
| **Whole-Leaf Evaluation** | Executed ($224 \times 224$ full canvas) | **Decoupled (`None`)** |
| **Patch Scale Fraction** | $52\%$ of canvas minimum dimension | $38\%$ of canvas minimum dimension |
| **Grid Stride** | $62\%$ step fraction | $62\%$ step fraction |
| **Typical Crop Count** | 9 proposals | 12 to 24 proposals |
| **Maximum Crop Cap** | 16 proposals | 36 proposals |

### 4.2 Foliar Saliency Pre-Filtering
Proposals are filtered prior to inference by calculating the foliar Excess Green (ExG) index across RGB channels:
$$\text{ExG} = 2G - R - B$$
Windows with negative mean ExG (soil, gravel, sky, trellis hardware) are rejected to save forward execution cycles.

### 4.3 Candidate Filtering & Greedy IoU NMS
1. **Confidence Threshold:** Candidates must achieve $\text{confidence} \ge 0.60$.
2. **Healthy Suppression:** Any candidate predicting a healthy class is pruned.
3. **Class-Wise Greedy NMS:** For overlapping proposals of the same predicted disease class with intersection-over-union:
   $$\text{IoU}(A, B) = \frac{\operatorname{Area}(A \cap B)}{\operatorname{Area}(A \cup B)} \ge 0.55$$
   The lower-confidence bounding box is suppressed.

---

## 5. API Schema Specification

### 5.1 POST `/screen-plant`
* **Content-Type:** `multipart/form-data`
* **Form Parameters:**
  * `file`: Binary image data (JPEG, PNG, WEBP).
  * `mode`: String (`"single_leaf"` or `"whole_plant"`, defaults to `"whole_plant"`).
* **Response (HTTP 200 OK):**
```json
{
  "quality": {
    "width": 1024,
    "height": 768,
    "format": "JPEG",
    "blur_score": 142.5,
    "is_blurry": false
  },
  "screening": {
    "mode": "single_leaf",
    "regions_checked": 9,
    "candidate_count": 2,
    "candidate_score_threshold": 0.6,
    "overall_prediction": {
      "predicted_class": "Apple___Apple_scab",
      "display_label": "Apple - Scab",
      "confidence": 0.9556,
      "score_label": "Uncalibrated Model Score",
      "prediction_state": "HIGH_CONFIDENCE",
      "is_uncertain": false,
      "is_healthy": false,
      "needs_review": false,
      "explanation": "...",
      "recommendation": "..."
    },
    "candidate_regions": [
      {
        "box": {"left": 120, "top": 85, "right": 450, "bottom": 415},
        "predicted_class": "Apple___Apple_scab",
        "display_label": "Apple - Scab",
        "confidence": 0.9868,
        "score_label": "Uncalibrated Model Score",
        "image": "data:image/jpeg;base64,...",
        "explanation": "...",
        "visible_evidence": "...",
        "immediate_actions": ["..."],
        "biological_treatment": "...",
        "chemical_treatment": "...",
        "sources": [{"name": "...", "title": "...", "url": "..."}]
      }
    ],
    "screening_state": "CANDIDATES_IDENTIFIED",
    "screening_status": "2 Candidate Region(s) Identified",
    "safety_disclaimer": "...",
    "overlay": "data:image/jpeg;base64,..."
  }
}
```

### 5.2 Error Response Codes
* `HTTP 413 Payload Too Large`: Upload exceeds $10\text{ MB}$.
* `HTTP 422 Unprocessable Entity`: 0-byte file, corrupt image header, sub-$128\text{px}$ dimension, or invalid `mode` string.
* `HTTP 503 Service Unavailable`: Model checkpoint missing or unloaded in memory.

