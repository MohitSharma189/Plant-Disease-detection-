# Deployment Screening Integration & Parity Validation Report

**Execution Timestamp:** 2026-11-12T10:20:26Z  
**Evaluated Application Service:** `backend.app.services.screening.screen_plant_image`  
**Model Registry:** `backend.app.services.models.ModelRegistry`  
**Active Checkpoint:** `saved_models/experiments/mobilenet_v3_small_final/best_model.pth` (Calibrated $T = 0.5538$, $\tau = 0.2231$)  
**Backup Image Created:** `backend/app/services/screening.py.bak` (Verified present on disk)  

---

## 1. Executive Summary

This validation confirms that the candidate-region sliding-window screening logic in the deployed FastAPI backend operates with **100% mathematical, geometric, and categorical equivalence** to the frozen research pipeline:

- **Candidate Count Agreement:** 100% identical.
- **Candidate Bounding-Box Coordinates:** 100% bit-for-bit identical $(left, top, right, bottom)$.
- **Predicted Class Names:** 100% identical.
- **Confidence Scores:** Maximum observed discrepancy $\le 0.00 \times 10^0$ (exact parity).
- **Review Decision Flag (`needs_review`):** 100% identical logic across all candidates.
- **Average CPU Screening Latency:** 405.32 ms per whole-plant image (well within the sub-500ms target).

---

## 2. Sample-by-Sample Parity Audit

| # | Filename | Dimensions | Deployed Candidates | Offline Candidates | Max Conf. Diff | CPU Latency | Verdict | Top Candidates (Sample) |
| :-: | :--- | :---: | :---: | :---: | :--- | :-: | :-: | :--- |
| 1 | `Black%20rot%20on%20foliage.jpg` | 1060x795 | 5 | 5 | 0.0e+00 | 549.1 ms | **PASS** | Tomato___Late_b (0.98, box=[748, 374, 1050, 676]); Tomato___Late_b (0.93, box=[748, 493, 1050, 795]); Squash___Powder (0.90, box=[561, 0, 863, 302]) |
| 2 | `vaccinium_angustifolium_leaf2.JPG.jpg` | 1440x2160 | 0 | 0 | 0.0e+00 | 416.3 ms | **PASS** | None (Filtered) |
| 3 | `stock-photo-fresh-soy-leaves-isolated-on-white-94266130.jpg` | 329x470 | 1 | 1 | 0.0e+00 | 289.6 ms | **PASS** | Tomato___Spider (0.65, box=[0, 316, 128, 444]) |
| 4 | `dsc_8148cs.jpg` | 1299x1267 | 8 | 8 | 0.0e+00 | 346.1 ms | **PASS** | Potato___Late_b (0.99, box=[596, 786, 1077, 1267]); Potato___Early_ (0.95, box=[298, 596, 779, 1077]); Potato___Early_ (0.90, box=[596, 0, 1077, 481]) |
| 5 | `apples_apple-scab_01_zoom.jpg` | 900x692 | 4 | 4 | 0.0e+00 | 425.5 ms | **PASS** | Apple___Apple_s (0.97, box=[489, 163, 752, 426]); Apple___Apple_s (0.87, box=[326, 163, 589, 426]); Corn_(maize)___ (0.84, box=[163, 326, 426, 589]) |

---

## 3. Screening Configuration & Operational Safeguards

The screening parameters in production are governed by [`reports/phase4/screening_configuration.json`](file:///E:/Plant%20Disease%20Detection-20261111T181021Z-1-001/Plant%20Disease%20Detection/reports/phase4/screening_configuration.json):

1. **Sliding-Window Geometry:**
   - Whole-plant mode side fraction: `0.38` of short side (clamped between 128px and short side).
   - Stride factor: `0.62` of side (clamped $\ge 64$ px).
   - Maximum evaluated windows per image: `36`.
2. **Filtering & Suppression:**
   - Minimum candidate confidence threshold: `0.60`.
   - Automatic suppression of healthy leaf candidates to prioritize disease detection.
   - Class-wise Non-Maximum Suppression (NMS) with IoU threshold `0.55`.
   - Maximum retained candidates per image: `8`.
3. **Calibrated Review Policy:**
   - Temperature scaling ($T = 0.5538$) applied before softmax.
   - Uncalibrated raw scores are never exposed directly to users.
   - Candidates with calibrated confidence below $\tau = 0.2231$ automatically carry `needs_review = True`.
