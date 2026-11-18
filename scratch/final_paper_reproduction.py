"""
Additional text and reproduction scripts for Appendix D and Appendix E.
"""

MATH_LOSS_REGULARIZATION = """\
<b>Mathematical Derivation of Class-Weighted Label-Smoothed Optimization:</b><br/>
To counteract the acute class frequency imbalance present across the 38 PlantVillage categories \
(ranging from 550 samples in Citrus Greening to 15 samples in Potato Healthy), we compute class \
weights w_c via normalized inverse square root frequency:
<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>w_c = sqrt( N / (K * N_c) ), &nbsp;&nbsp; where sum_{c=1}^K w_c = K</b><br/>
This square-root attenuation avoids the extreme gradient variance induced by pure inverse frequency \
weighting (w_c proportional to 1/N_c), which frequently destabilizes early fine-tuning stages. \
Incorporating label smoothing with factor epsilon = 0.10, the modified cross-entropy objective is:
<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>L(theta) = -(1/N) * sum_{i=1}^N w_{y_i} * [ (1 - epsilon) * log(p_{i, y_i}) + (epsilon / K) * sum_{k=1}^K log(p_{i, k}) ]</b><br/>
Furthermore, during fine-tuning (Epochs 2–3), cosine annealing modulates the learning rate:
<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>eta_t = eta_{min} + 0.5 * (eta_{max} - eta_{min}) * [ 1 + cos(pi * t / T_{max}) ]</b><br/>
with eta_{max} = 1e-4 for the classification head and eta_{max} = 1e-5 for the backbone features, \
ensuring graceful convergence into flat local minima associated with superior out-of-distribution robustness.
"""

REPRODUCTION_PROTOCOL = """\
<b>D.1 End-to-End Deterministic Reproduction Commands</b><br/>
The accompanying codebase provides exact command sequences executing the Phase 1–5 research lifecycle under seed 42:
<br/><br/>
<b>Step 1 — Duplicate-Aware Dataset Partitioning & Combined Manifest Generation:</b><br/>
<code>python training/prepare_splits.py --dataset-dir data/processed/PlantVillage --seed 42 --output-manifest reports/generated/combined_split_manifest.json</code><br/>
<i>Verification Check:</i> Generates combined split manifest SHA-256 <code>f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868</code> (56,433 records: 40,151 train, 5,436 val, 5,429 calib, 5,417 test).
<br/><br/>
<b>Step 2 — Definitive MobileNetV3-Small Fine-Tuning:</b><br/>
<code>python training/train_definitive.py --config training/config_final.yaml</code><br/>
<i>Operational Configuration:</i> 3 completed epochs, batch size 8, head warmup 1 epoch, initial lr 0.001, backbone fine-tune lr 1e-5, label smoothing 0.10, seed 42, deterministic PyTorch. Checkpoint SHA-256 <code>6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009</code>.
<br/><br/>
<b>Step 2b — Comparative Multi-Architecture Benchmarking:</b><br/>
<code>python training/train_benchmark_architecture.py --arch [mobilenet_v3_large|efficientnet_b0|resnet18] --config training/config_final.yaml</code><br/>
<i>Verification Check:</i> Generates verified comparative weights in <code>saved_models/experiments/architecture_benchmark/</code>.
<br/><br/>
<b>Step 3 — Post-Hoc Probability Calibration:</b><br/>
<code>python training/evaluate_classifier.py --config training/config_final.yaml</code><br/>
<i>Verification Check:</i> Fits optimal scalar T* = 0.5538 via L-BFGS on 5,429 calibration samples, reducing calibration ECE from 10.25% to 0.58% (holdout test ECE from 10.28% to 0.74%).
<br/><br/>
<b>Step 4 — Independent PlantDoc External Evaluation:</b><br/>
<code>python training/evaluate_plantdoc.py --config training/config_final.yaml</code><br/>
<i>Verification Check:</i> Evaluates 236 natural-background test images, confirming 38.98% Top-1 accuracy, 36.84% Macro-F1, and selective prediction (93.22% coverage, 40.91% selective accuracy at threshold 0.2231).
<br/><br/>
<b>Step 5 — Phase 4.1 Candidate-Region Screening Optimization:</b><br/>
<code>python scratch/run_phase4_1_final_test_evaluation.py</code><br/>
<i>Verification Check:</i> Evaluates 236 images (452 ground-truth leaves), confirming Recall@0.25 = 38.05%, Precision@0.25 = 10.60%, F1 = 16.58%, 6.15 FP/img, and 844.25 ms mean CPU latency.
<br/><br/>
<b>D.2 Environment Verification & Runtime Determinism</b><br/>
Deterministic multi-threading and CPU concurrency controls:
<br/>
<code>export OMP_NUM_THREADS=8</code><br/>
<code>export MKL_NUM_THREADS=8</code><br/>
<code>export PYTHONHASHSEED=42</code><br/>
<code>python -c "import torch; print(f'PyTorch: {torch.__version__}, Deterministic: {torch.are_deterministic_algorithms_enabled()}')"</code>
"""

TRIAGE_GUIDE = """\
<b>E.1 Three-Tier Non-Autonomous Screening Protocol for Extension Workers</b><br/>
To translate model outputs into responsible field support without claiming automated clinical diagnostic \
authority, we define a structured three-tier screening triage protocol:
<br/><br/>
&bull; <b>Tier 1 — High-Confidence Screening Candidate (p_calibrated &ge; 0.70):</b><br/>
The detected leaf region demonstrates visual symptom patterns matching a supported pathology with high statistical \
confidence. The system generates bounding box visualization, class attribution heatmaps, and a non-prescriptive \
symptom summary. An agronomic advisory note explicitly instructs the extension worker to perform physical in-field \
verification of foliar margins and local environmental preconditions before initiating standard diagnostic workflow.
<br/><br/>
&bull; <b>Tier 2 — Low-Confidence Referral (0.223 &le; p_calibrated &lt; 0.70):</b><br/>
The localized region exhibits abnormal foliar coloration or early necrotic lesions, but the posterior probability \
falls below confident discrimination thresholds. The system outputs the top-3 differential diagnoses with calibrated \
likelihoods, flags the sample for manual expert inspection, and suppresses automated decision recommendations.
<br/><br/>
&bull; <b>Tier 3 — Inconclusive or Unsupported Scene (p_calibrated &lt; 0.223 or ExG Coverage &lt; 8%):</b><br/>
The input image contains severe motion blur, off-target foliage, complex overlapping canopies, or unsupported \
plant species. The screening engine suppresses candidate proposals and advises the user to recapture the photograph \
under diffuse natural lighting with a single isolated leaf in view. Under no condition does Tier 3 report a disease label.
"""

FIELD_DECISION_DATA = [
    ["Crop & Target Pathology", "Pathological Foliar Indicators", "Confounding Abiotic Stressors / Lookalikes", "Human-in-the-Loop Screening Triage Protocol"],
    ["Apple Scab (Venturia inaequalis)", "Olive-green to velvety dark brown lesions on adaxial leaf surface", "Sooty mold, chemical spray residue, mechanical abrasive wind injury", "Tier 1: Physical inspection of foliar margins; hand lens check; refer sample to extension pathologist if >15% canopy affected."],
    ["Corn Northern Leaf Blight (Exserohilum turcicum)", "Elongated, elliptical grayish-green or tan cigar-shaped necrotic streaks", "Drought leaf scorch, nitrogen deficiency chlorosis, Stewart's wilt", "Tier 1: Scout lower and mid-canopy for lesion progression along veins; collect representative foliar press for regional extension clinic confirmation."],
    ["Grape Black Rot (Guignardia bidwellii)", "Small circular reddish-brown spots with dark necrotic borders and black pycnidia", "Anthracnose, pesticide phytotoxicity, hail puncturing", "Tier 1: Cross-check surrounding petioles and fruit clusters for pycnidia; record GPS plot coordinates for targeted extension re-survey."],
    ["Potato Early Blight (Alternaria solani)", "Concentric target-board rings surrounded by localized chlorotic halo", "Magnesium deficiency, physiological senescence, ozone flecking", "Tier 1: Differentiate concentric zonation from physiological senescence; recommend physical soil and tissue testing by certified agronomist."],
    ["Potato Late Blight (Phytophthora infestans)", "Water-soaked irregular pale green lesions expanding rapidly to dark brown with white sporulation", "Frost injury, wind desiccation, bacterial soft rot", "Tier 1 (Urgent): Visual inspection for white sporulation under high humidity; flag specimen for priority diagnostic laboratory assay."],
    ["Tomato Yellow Leaf Curl Virus (TYLCV)", "Upward leaf curling, severe marginal chlorosis, stunted terminal leaflets", "Severe aphid infestation, mechanical mite damage, physiological leaf roll", "Tier 1: Scout underside of terminal leaflets for whitefly vectors; refer symptomatic foliage to university plant virology diagnostic facility."],
    ["Low Confidence / Ambiguous Foliage", "Diffused chlorosis, atypical lesion shape, overlapping foliage, solar glare", "Nutrient deficiencies (Fe, Zn), herbicide drift, senescence", "Tier 2: Defer automated classification; withhold diagnosis; refer image and physical leaf press to extension diagnostic service."],
    ["Unsegmented / Extreme Occlusion", "Total leaf coverage < 8% of field of view, extreme motion blur, shadow cast", "Bark, soil, non-target background weeds, artificial mulch", "Tier 3: Suppress candidate inference; prompt user to rephotograph under diffuse natural lighting with a single isolated leaf on a neutral backing card."]
]

DEPLOYMENT_CHECKLIST = """\
<b>E.2 Field Deployment Protocol & Mobile Image Capture Guidelines</b><br/>
To minimize out-of-distribution false positives in practical operational settings, field agricultural extension officers \
must adhere to standardized image acquisition constraints before submitting candidates to the screening pipeline:
<br/><br/>
1. <b>Foliar Isolation & Background Control:</b> Position the target leaf flat without folding or extreme curl. Where \
dense background weed cover or bare soil creates color interference, place a matte neutral surface (or shaded hand) \
directly behind the specimen. The leaf blade must occupy at least 35% of the central viewfinder.
<br/>
2. <b>Diffuse Ambient Illumination:</b> Avoid capturing leaves under direct, harsh perpendicular sunlight, which produces \
specular reflection hotspots that saturate green-channel values and mimic viral chlorosis. Early morning diffuse light, \
overcast conditions, or manual body shading are required.
<br/>
3. <b>Lens Sanitation & Focusing:</b> Clean mobile camera lenses to prevent grease smearing that creates false halo \
artifacts. Maintain a distance of 15–25 cm to ensure fine lesion micro-structures (e.g., fungal pycnidia, concentric rings) \
remain sharp and resolvable above the 224x224 input resolution threshold.
<br/>
4. <b>Multi-Angle Verification:</b> For suspected quarantine-grade pathogens (such as Phytophthora infestans), capture \
at least two complementary views (adaxial top surface and abaxial underside) to confirm characteristic sporulation patterns \
prior to escalating mitigation measures.
"""
