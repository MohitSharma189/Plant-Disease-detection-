"""
Section text for the revised research paper.
All experimental claims reference final_paper_data.py values.
"""

ABSTRACT = """\
Plant diseases account for substantial annual losses in global agricultural production, \
yet automated screening tools trained on controlled laboratory images frequently fail \
under natural field conditions. This paper presents a reproducible experimental \
investigation of MobileNetV3-Small and three comparative convolutional architectures \
(MobileNetV3-Large, EfficientNet-B0, and ResNet18) for plant disease classification, \
uncertainty calibration, and candidate-region screening on CPU hardware. \
A duplicate-aware, cryptographically provenance-tracked dataset of 56,433 images \
(40,151 training, 5,436 validation, 5,429 calibration, and 5,417 independent holdout \
images across 38 disease classes and 14 crop species) was constructed from 54,305 \
source PlantVillage images after excluding 31 duplicate and conflicting records \
(10 cross-label conflicts and 21 intra-class exact duplicates) and integrating 2,159 \
PlantDoc training images. An independent external test benchmark of 236 natural-background \
PlantDoc images (452 ground-truth leaves) was reserved for untouched domain-shift evaluation. \
The definitive MobileNetV3-Small checkpoint achieved 95.02% holdout accuracy and 93.13% \
Macro-F1 on PlantVillage, but only 38.98% Top-1 accuracy and 36.84% Macro-F1 on the \
independent PlantDoc external benchmark, exposing a 56-percentage-point domain gap. \
Post-hoc temperature scaling reduced calibration Expected Calibration Error (ECE) \
from 10.25% to 0.58% on the calibration partition (T* = 0.5538). \
A multi-scale, aspect-ratio-aware candidate-region screening pipeline improved \
localization Recall at IoU 0.25 from 18.14% (Phase 4 baseline) to 38.05% \
(Phase 4.1 optimized), at the cost of increasing end-to-end CPU latency from \
167 ms to 844 ms. All results are reproducible from verified artifacts; \
unresolved calibration discrepancies between independent evaluation runs are \
documented transparently. The system is presented as a research prototype, \
not a clinically validated diagnostic product.\
"""

KEYWORDS = [
    "Plant Disease Classification",
    "MobileNetV3",
    "Domain Generalization",
    "Temperature Scaling",
    "Candidate-Region Screening",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────

INTRO_BACKGROUND = """\
Plant diseases are a primary driver of crop yield reduction worldwide. The Food \
and Agriculture Organization estimates that plant pests and pathogens destroy \
between 20% and 40% of global food production annually (FAO, 2021). Early \
visual identification of disease symptoms enables timely intervention and can \
significantly reduce losses, yet expert agronomists are unevenly distributed \
across smallholder farming regions. Image-based machine learning systems offer \
a scalable alternative; however, transferring laboratory-validated models to \
field conditions requires careful evaluation of their actual capabilities and \
their limitations (Mohanty et al., 2016; Barbedo, 2018).\
"""

INTRO_PROBLEM = """\
The PlantVillage dataset (Hughes & Salathé, 2015) has become the dominant \
benchmark for plant disease classification, providing over 54,000 controlled \
leaf images across 38 disease categories. Many published systems report \
accuracy above 95% on PlantVillage splits; however, the same models often \
perform substantially worse on naturally photographed images from the field \
(Ferentinos, 2018; Ramcharan et al., 2019; Thenmozhi & Reddy, 2019). \
PlantDoc (Singh et al., 2020), a dataset of 2,598 internet-collected plant \
disease images, has been adopted as a cross-domain stress test, but its small \
size and internet-collection methodology introduce additional uncertainty. \
A secondary limitation of many published studies is insufficient provenance \
documentation: checkpoint training splits are rarely published alongside \
results, making it impossible to verify that reported test scores are \
genuinely independent (Barbedo, 2018; Lu et al., 2021).\
"""

INTRO_GAP = """\
Several gaps persist in the literature on lightweight plant disease classifiers \
deployed for field screening. First, most studies evaluate a single architecture \
against PlantVillage and do not report external PlantDoc performance, leaving \
domain generalization unquantified (Karthik et al., 2020; Chen et al., 2021). \
Second, model calibration — the alignment between predicted confidence scores \
and empirical accuracy — is rarely measured or corrected despite its importance \
for uncertainty-aware decision support (Guo et al., 2017; Minderer et al., 2021). \
Third, whole-plant screening, which requires localizing candidate symptomatic \
regions within complex images, is typically addressed with dedicated object \
detectors (Fuentes et al., 2017; Tm et al., 2018) rather than classification \
models, leaving an unaddressed question of how well sliding-window candidate \
generation from a CPU-class classifier performs in practice.\
"""

INTRO_OBJECTIVES = [
    "Construct a fully provenance-tracked, duplicate-aware dataset partition for reproducible evaluation of plant disease classifiers.",
    "Train and evaluate the definitive MobileNetV3-Small baseline with verified checkpoint provenance and independent holdout testing.",
    "Conduct independent external evaluation on PlantDoc to quantify the domain gap between controlled PlantVillage images and natural field photographs.",
    "Apply post-hoc temperature scaling to all evaluated architectures and document calibration performance transparently, including discrepancies across evaluation procedures.",
    "Design and evaluate a multi-scale, aspect-ratio-aware candidate-region screening pipeline for whole-plant images and report localization metrics against ground-truth annotations.",
    "Benchmark MobileNetV3-Large, EfficientNet-B0, and ResNet18 against the MobileNetV3-Small baseline across accuracy, calibration, latency, and external generalization.",
]

INTRO_CONTRIBUTIONS = [
    "A cryptographically provenance-tracked training pipeline recording manifest SHA-256 and per-partition split identifiers in checkpoint metadata.",
    "Independent PlantDoc external evaluation quantifying a 56-percentage-point accuracy drop from PlantVillage holdout (95.02%) to PlantDoc (38.98%).",
    "Post-hoc temperature calibration analysis with transparent reconciliation of discrepancies between Phase 3 (T*=0.5538, ECE 0.58%) and Phase 5 (T*=0.9902, ECE 9.94%) evaluation procedures.",
    "Phase 4.1 optimized multi-scale screening pipeline improving Recall@IoU 0.25 by 109.76% over the Phase 4 baseline, with explicit acknowledgment of elevated false-positive rates.",
    "A systematic four-architecture comparison (MobileNetV3-Small/Large, EfficientNet-B0, ResNet18) covering accuracy, calibration, CPU latency, and cross-domain performance.",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LITERATURE REVIEW
# ─────────────────────────────────────────────────────────────────────────────

LIT_INTRO = """\
This section surveys the principal bodies of research relevant to the present \
work: plant disease image datasets, lightweight CNN architectures, domain \
generalization between curated and natural images, model calibration, and \
candidate-region localization in agricultural images.\
"""

LIT_DATASETS = """\
Hughes and Salathé (2015) released PlantVillage, an open repository of controlled \
leaf photographs spanning 38 disease categories across 14 crop species. \
Mohanty et al. (2016) demonstrated that deep convolutional classifiers trained \
on PlantVillage achieve accuracy above 99% under laboratory conditions but \
cautioned that background variation significantly degrades field performance. \
Barbedo (2018) identified acquisition conditions, symptom variability, and \
dataset composition as primary factors limiting real-world applicability, \
noting that most published results cannot be reproduced because training split \
information is unavailable. \
Singh et al. (2020) introduced PlantDoc, a dataset of 2,598 internet-collected \
images across 13 plant species, providing a cross-domain stress test. Models \
trained only on PlantVillage typically achieve accuracy below 50% on PlantDoc, \
confirming the controlled-to-natural image generalization gap. \
Ferentinos (2018) evaluated multiple deep learning architectures on PlantVillage \
and reported high within-distribution accuracy, while acknowledging that \
generalization to field images requires separate evaluation. \
Ramcharan et al. (2019) deployed cassava disease classifiers on mobile devices \
and found that fine-tuning with locally collected field images was necessary \
for acceptable performance, illustrating the practical limits of laboratory-only \
training.\
"""

LIT_ARCHITECTURES = """\
Howard et al. (2019) introduced MobileNetV3, a family of efficient architectures \
designed for mobile and embedded deployment using hard-swish activations and \
neural architecture search. The Small variant contains approximately 1.56 million \
parameters and achieves competitive ImageNet accuracy with single-digit millisecond \
CPU inference latency, making it suitable for CPU-first field applications. \
Tan and Le (2019) proposed EfficientNet, scaling network depth, width, and \
resolution jointly using a compound coefficient. EfficientNet-B0, the base model, \
achieves 77.1% top-1 ImageNet accuracy with 5.3 million parameters and has been \
widely applied to plant disease recognition (Khan et al., 2022; Tiwari et al., 2021). \
He et al. (2016) introduced ResNet with skip connections, enabling training of \
very deep networks by addressing the vanishing gradient problem. ResNet18 \
(11.2 million parameters) remains a strong baseline for transfer learning \
in agricultural image classification (Lu et al., 2021; Militante & Dionisio, 2019). \
Chen et al. (2021) benchmarked multiple architectures including MobileNet variants \
on plant disease datasets and found that the accuracy advantage of larger models \
diminishes substantially under distribution shift. \
Karthik et al. (2020) compared MobileNetV2, ResNet, and DenseNet for leaf disease \
detection and reported that lightweight architectures trained with appropriate \
augmentation can match heavier models on PlantVillage while running faster on \
edge devices. \
Thenmozhi and Reddy (2019) demonstrated that cropping augmentation and mixup \
training improves generalization for plant disease classifiers trained on \
imbalanced datasets.\
"""

LIT_DOMAIN = """\
Domain generalization between controlled laboratory images and natural field \
photographs remains the central challenge for plant disease AI systems. \
Barbedo (2018) systematically documented how background clutter, lighting \
variation, and partial occlusion degrade classifier confidence without \
triggering model uncertainty warnings. \
Fuentes et al. (2017) applied Faster R-CNN and Single Shot MultiBox Detector \
(SSD) for tomato disease detection in natural images and found that object \
detectors trained on natural images substantially outperform classifiers applied \
to sliding-window crops. \
Tm et al. (2018) evaluated deep learning models for plant disease detection \
under controlled and uncontrolled conditions and reported accuracy drops of \
30–60 percentage points when moving from laboratory to field images. \
Chouhan et al. (2020) reviewed deep learning approaches for plant disease \
recognition and concluded that model evaluation must include cross-dataset \
testing to provide meaningful generalization evidence. \
Lu et al. (2021) proposed a domain adaptation framework for plant disease \
recognition transferring knowledge from PlantVillage to field images and \
demonstrated that fine-tuning on even a small number of field images \
significantly improved external performance. \
Militante and Dionisio (2019) reported that ensemble methods and test-time \
augmentation partially compensate for domain shift but do not eliminate it. \
Nawaz et al. (2022) reviewed fourteen deep learning methods for plant disease \
detection and emphasized that PlantVillage accuracy above 95% does not imply \
readiness for autonomous field deployment without additional domain-specific \
validation.\
"""

LIT_CALIBRATION = """\
Calibration, the agreement between a model's predicted probabilities and \
empirical accuracy, is critical for uncertainty-aware decision support systems. \
Guo et al. (2017) demonstrated that modern deep neural networks are \
overconfident: high-capacity models trained with cross-entropy loss produce \
predicted probabilities that systematically exceed empirical accuracy. \
Temperature scaling, a post-hoc method fitting a single scalar divisor \
to held-out logits, was shown to significantly reduce Expected Calibration \
Error (ECE) while preserving classification accuracy. \
Minderer et al. (2021) extended this analysis to vision transformers and \
modern convolutional models, finding that well-calibrated models are more \
reliable for downstream decision pipelines. \
Nixon et al. (2019) introduced Reliability Diagram analysis and proposed \
improved ECE estimators to address bin-size sensitivity, recommending at \
least 15 bins for datasets with more than 5,000 evaluation examples. \
Kumar et al. (2019) proposed adaptive calibration error (ACE) and showed \
that calibration quality varies significantly across model architectures and \
training configurations, reinforcing the need for per-checkpoint calibration \
measurement rather than architectural assumption. \
In plant disease applications, calibration has received limited attention: \
most published systems report raw softmax confidence without measuring ECE, \
making it impossible to assess whether model confidence is a reliable signal \
for farmer decision support.\
"""

LIT_LOCALIZATION = """\
Localizing symptomatic regions within whole-plant images requires either \
dedicated object detectors or sliding-window classification approaches. \
Fuentes et al. (2017) applied deep object detection architectures including \
Faster R-CNN and Multibox SSD to tomato disease localization and achieved \
mean average precision above 80% on annotated datasets collected under \
controlled conditions. \
Tm et al. (2018) used a sliding-window CNN approach similar to the present \
work and reported that classification confidence thresholds must be tuned \
empirically to manage precision-recall trade-offs for candidate region proposals. \
Atole and Park (2018) demonstrated that multi-scale sliding windows improve \
recall for leaf-level symptom detection compared to single-scale approaches, \
confirming the rationale for multi-scale candidate generation in the present study. \
Johannes et al. (2017) evaluated smartphone-based leaf disease detection \
and found that background segmentation using vegetation indices (similar to \
the Excess Green index used in Phase 4.1) can reduce false positives from \
background patches before classifier evaluation. \
Recent work by Chen et al. (2021) and Nawaz et al. (2022) confirmed that \
training classifiers on cropped leaf images results in poor localization when \
applied to whole-plant images, motivating explicit candidate-region evaluation \
rather than implicitly assuming classification accuracy transfers to detection.\
"""

LIT_GAP_STATEMENT = """\
Taken together, the literature reveals three underaddressed gaps. First, most \
lightweight architecture comparisons focus on PlantVillage accuracy and do not \
report calibration or external PlantDoc performance simultaneously. Second, \
temperature scaling has not been systematically applied and compared across \
multiple architectures in plant disease settings with transparent discrepancy \
reporting. Third, sliding-window candidate-region screening has not been \
evaluated against annotated whole-leaf bounding boxes with explicit IoU \
threshold analysis across multiple pipeline configurations. The present work \
addresses all three gaps while maintaining transparent documentation of \
unresolved limitations.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────

METH_DATASET = """\
The primary training corpus was constructed from the PlantVillage dataset \
(Hughes & Salathé, 2015), which contains 54,305 readable images across 38 \
disease categories and 14 crop species. Following image verification, exact \
and perceptual duplicate detection identified 31 duplicate images: 10 cross-label \
duplicates with conflicting pathology annotations across classes (which were \
completely excluded to prevent label corruption) and 21 intra-class redundant \
exact duplicates, yielding 54,274 eligible PlantVillage images. The dataset was \
partitioned using stratified group sampling with deterministic seed 42 into four \
non-overlapping splits: training (37,992 images, 70.00%), validation (5,436 images, \
10.02%), calibration (5,429 images, 10.00%), and independent holdout test \
(5,417 images, 9.98%). To expose the network to diverse environmental backgrounds \
during training, 2,159 deduplicated training images from the PlantDoc dataset \
(Singh et al., 2020) were added to the training partition only, producing a \
combined training partition of 40,151 images and a total combined split manifest \
of 56,433 images (SHA-256: \\texttt{f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868}). \
Validation, calibration, and holdout test partitions remained strictly composed \
of pure PlantVillage images. An independent external test set of 236 natural-background \
PlantDoc images (containing 452 annotated leaves across 13 cross-mapped classes) was \
reserved exclusively for untouched out-of-distribution evaluation.\
"""

METH_ARCHITECTURES = """\
Four convolutional architectures were evaluated. All models were initialized \
from ImageNet-pretrained weights and fine-tuned on the same split manifest. \
\\textbf{MobileNetV3-Small} (Howard et al., 2019) contains 1.56 million \
parameters with an inverted-residual bottleneck structure, hard-swish \
activations, and squeeze-and-excitation modules. The final classifier head \
was replaced with a linear layer mapping 576 features to 38 output logits. \
\\textbf{MobileNetV3-Large} (Howard et al., 2019) extends the Small variant \
with 4.25 million parameters and wider intermediate channels, increasing \
representational capacity at the cost of inference speed. \
\\textbf{EfficientNet-B0} (Tan & Le, 2019) uses compound scaling to balance \
depth, width, and resolution, with 4.06 million parameters and mobile inverted \
bottleneck convolutions. \
\\textbf{ResNet18} (He et al., 2016) employs residual skip connections across \
18 convolutional layers with 11.2 million parameters, serving as the \
largest and most computationally expensive comparative baseline.\
"""

METH_TRAINING = """\
All models were trained with the following shared configuration. \
Input images were resized to 224\\(\\times\\)224 pixels. Training-time augmentations \
included random resized crops (scale 0.78–1.0, ratio 0.85–1.15), horizontal flips (p=0.5), \
vertical flips (p=0.1), color jitter (brightness 0.22, contrast 0.22, saturation 0.20, hue 0.04), \
random affine transformations, random perspective distortion, Gaussian blur, shadow simulation, \
and random erasing. Validation and test transforms applied only resizing to 224x224 and standard \
ImageNet normalization. Training was executed using training/train_definitive.py with batch size 8 \
over 3 completed epochs on an AMD Ryzen 7 5800HS CPU platform. Head warmup for 1 epoch \
optimized classifier parameters at learning rate 1e-3, while subsequent two epochs \
fine-tuned backbone weights at 1e-5 and head weights at 1e-4 using the AdamW optimizer \
(weight decay 1e-4) with cosine annealing decay. Cross-entropy loss incorporated head \
dropout (0.30) and label smoothing (0.10). Class weights were computed as the normalized \
square root of inverse class frequency to address class imbalance. Validation loss early \
stopping with patience 2 and minimum delta 1e-3 monitored convergence. Deterministic \
algorithms were enforced (\\texttt{torch.use_deterministic_algorithms(True)}) with seed 42 throughout.\
"""

METH_METRICS = """\
Per-class precision, recall, and F1 score were computed as:
\\begin{equation}
P_c = \\frac{\\mathrm{TP}_c}{\\mathrm{TP}_c + \\mathrm{FP}_c}, \\quad
R_c = \\frac{\\mathrm{TP}_c}{\\mathrm{TP}_c + \\mathrm{FN}_c}, \\quad
F1_c = \\frac{2P_cR_c}{P_c + R_c}
\\end{equation}
Macro-averaged metrics represent the unweighted mean across all 38 classes. \
Temperature scaling converts logits \\(z\\) into calibrated probabilities:
\\begin{equation}
p_k = \\mathrm{softmax}(z/T^*)_k
\\end{equation}
where \\(T^*\\) is fitted by minimizing negative log-likelihood on the \
calibration partition using L-BFGS. \
The 15-bin Expected Calibration Error (ECE) is:
\\begin{equation}
\\mathrm{ECE} = \\sum_{b=1}^{15} \\frac{|B_b|}{N} \\left| \\mathrm{acc}(B_b) - \\mathrm{conf}(B_b) \\right|
\\end{equation}
where \\(B_b\\) denotes the set of examples whose maximum predicted probability \
falls in bin \\(b\\), and \\(N\\) is the total number of evaluation examples. \
The multiclass Brier score measures probability calibration as:
\\begin{equation}
\\mathrm{Brier} = \\frac{1}{N}\\sum_{i=1}^{N}\\sum_{k}(p_{ik} - y_{ik})^2
\\end{equation}
Candidate-region localization was evaluated using Intersection over Union (IoU):
\\begin{equation}
\\mathrm{IoU}(a,b) = \\frac{|a \\cap b|}{|a \\cup b|}
\\end{equation}
Localization precision and recall were computed by greedy one-to-one matching \
of predicted candidates to ground-truth leaf bounding boxes at IoU thresholds \
of 0.10, 0.25, and 0.50.\
"""

METH_SCREENING = """\
The candidate-region screening pipeline processes whole-plant images through \
five stages. \\textbf{(1) Multi-scale window generation:} overlapping windows \
at three scale fractions (0.25, 0.40, 0.65 of the image dimension) and three \
aspect ratios (1:1, 4:3, 3:4) are generated with stride factor 0.70. \
\\textbf{(2) Foliar saliency pre-filtering:} each window is evaluated by the \
Excess Green index \\(\\mathrm{ExG} = 2G - R - B\\) (Woebbecke et al., 1995), \
and windows with foliar pixel coverage below 8\\% are discarded without \
invoking the classifier. \
\\textbf{(3) Context padding:} surviving windows are expanded by 12\\% on each \
side to include boundary tissue before resizing to 224\\(\\times\\)224. \
\\textbf{(4) CNN classification:} MobileNetV3-Small produces calibrated \
probabilities for each padded crop; windows with maximum non-healthy \
class probability below 0.55 are discarded. \
\\textbf{(5) Hybrid Non-Maximum Suppression (NMS):} remaining candidates are \
sorted by score, and class-wise suppression at IoU 0.50 followed by cross-class \
suppression at IoU 0.70 reduces duplicate proposals. At most ten candidates \
are retained per image (selected configuration).\
"""

METH_CALIB_DISCREPANCY = """\
Two independent calibration evaluations of the same MobileNetV3-Small checkpoint \
produced materially different results: Phase 3 obtained \\(T^* = 0.5538\\) with \
post-calibration ECE 0.58\\% on the calibration partition (0.74\\% on holdout test), \
while Phase 5 batch calibration obtained \\(T^* = 0.9902\\) with test-split ECE 9.94\\%. \
Both evaluations used the same checkpoint (SHA-256: \\texttt{6190f54c...}) and the \
same split manifest. Investigation of the two codebases reveals distinct methodological \
implementations: Phase 3 (\\texttt{training/evaluate_classifier.py}) parameterized \
log-temperature (\\(\\log T_0 = 0\\)) optimized via L-BFGS (learning rate 0.1, 80 maximum \
iterations, strong Wolfe line search) on the calibration partition, whereas Phase 5 \
(\\texttt{training/run_phase5_evaluation.py}) parameterized raw temperature initialized at \
\\(T_0 = 1.5\\) using L-BFGS (learning rate 0.01, 50 maximum iterations, without line search) \
and reported metrics on the test partition. Because the objective landscape sensitivity to \
these differing optimizer hyperparameters has not been exhaustively proven under controlled ablation, \
this discrepancy is documented transparently as an unresolved methodological variation \
between evaluation pipelines, rather than attributed definitively to optimizer failure. \
Both sets of results are reported in Table 8.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RESULTS AND DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────

RES_PHASE2_TRAINING = """\
The definitive MobileNetV3-Small baseline converged at epoch 3 with best \
validation loss 0.2514, validation accuracy 95.14\\%, and validation \
Macro-F1 93.19\\%. Training completed in 65.51 minutes on a CPU-only \
AMD Ryzen 7 5800HS platform. Early stopping terminated training after three \
epochs, suggesting that the frozen warm-up and subsequent fine-tuning schedule \
was effective: the model gained 1.67 percentage points in validation accuracy \
between epoch 2 and epoch 3 as fine-tuning propagated gradients through the \
backbone. The corresponding checkpoint was assigned SHA-256 \
\\texttt{6190f54c...} and recorded in the split manifest to enable \
deterministic reproduction.\
"""

RES_HOLDOUT = """\
On the independent PlantVillage holdout partition (5,417 images, fully \
isolated from training and validation), MobileNetV3-Small achieved 95.02\\% \
Top-1 accuracy and 93.13\\% Macro-F1. The 0.12 percentage-point difference \
between validation accuracy (95.14\\%) and holdout accuracy (95.02\\%) \
indicates negligible overfitting to the validation split. Weighted F1 (95.14\\%) \
remained close to Macro-F1 (93.13\\%), indicating broadly consistent performance \
across both high-support and low-support classes. Per-class analysis revealed \
that the five lowest-performing classes were Tomato Early Blight (F1 72.6\\%), \
Corn Gray Leaf Spot (F1 74.7\\%), Tomato Septoria Leaf Spot (F1 84.0\\%), \
Tomato Late Blight (F1 84.8\\%), and Tomato Leaf Mold (F1 85.7\\%). These classes \
share visually similar symptom patterns — irregular necrotic lesions with \
indistinct boundaries — which likely contribute to inter-class confusion.\
"""

RES_PLANTDOC = """\
The independent PlantDoc external benchmark (236 images, 27 of 38 classes \
represented) revealed a substantial domain generalization gap. MobileNetV3-Small \
achieved 38.98\\% Top-1 accuracy and 36.84\\% Macro-F1 on PlantDoc, compared \
to 95.02\\% and 93.13\\% respectively on the PlantVillage holdout — a gap of \
56.03 percentage points in accuracy. This gap is consistent with findings \
reported by Singh et al. (2020) and Barbedo (2018), who attributed similar \
drops to differences in background complexity, lighting conditions, image \
resolution, and the presence of multiple disease symptoms per image in natural \
photographs. Eleven of the 38 classes in the taxonomy were not represented in \
the 236-image PlantDoc test partition, limiting per-class analysis coverage. \
Under selective prediction, applying the calibration-derived review threshold \
\\(\\tau = 0.2231\\) (which rejected zero calibration samples on PlantVillage due to \
high baseline confidence) to the 236 PlantDoc test images rejected 16 low-confidence \
samples (6.78\\%), yielding 93.22\\% selective coverage (220/236 images). On this retained \
subset, top-1 selective accuracy rose from 38.98\\% to 40.91\\% (90/220 images), confirming \
that confidence-based referral provides modest risk reduction but cannot compensate for \
the fundamental domain shift. These results demonstrate that laboratory accuracy cannot \
serve as a proxy for field diagnostic reliability.\
"""

RES_CALIBRATION = """\
Phase 3 calibration on the dedicated calibration partition (5,429 images) \
produced temperature \\(T^* = 0.5538\\) with pre-calibration ECE 10.25\\% and \
post-calibration ECE 0.58\\%. Applied to the holdout partition, the same \
temperature reduced ECE from 10.28\\% to 0.74\\%, confirming that the \
calibration generalizes well within the PlantVillage distribution. \
Applied to PlantDoc images, temperature scaling increased ECE from 7.43\\% \
to 21.86\\%, because the calibration partition temperature was fitted on \
PlantVillage data and does not transfer to the domain-shifted distribution. \
This result illustrates that calibration fitted on controlled images must \
not be assumed to hold for natural photographs. \
Phase 5 batch calibration with the same checkpoint produced \\(T^* = 0.9902\\) \
and post-calibration test-split ECE 9.94\\%, a materially different result. \
As documented in Section 3, this discrepancy reflects differing optimizer parameterizations \
(log-temperature with Wolfe line search vs. raw temperature with smaller learning rate) \
and evaluation targets across independent evaluation scripts. Both results are reported \
in Table 8 for complete transparency, with the discrepancy documented as an unresolved \
methodological variation. The 100\\% prediction parity between offline and deployed \
backend models on the 10-image verification sample confirms that the calibrated temperature \
is correctly applied during inference.\
"""

RES_ARCH_COMPARISON = """\
All four architectures achieved holdout accuracy between 95.02\\% and 95.96\\%, \
confirming that the PlantVillage task is well within the capacity of these \
lightweight to medium-weight convolutional models. MobileNetV3-Large achieved \
the highest holdout accuracy (95.96\\%) and Macro-F1 (94.52\\%), followed \
closely by ResNet18 (95.92\\%, 94.30\\%) and EfficientNet-B0 (95.77\\%, 94.26\\%). \
MobileNetV3-Small was the lowest on PlantVillage accuracy but the most \
computationally efficient, with 9.42 ms single-image latency and 6.06 MB \
checkpoint size. \
The external PlantDoc generalization ranking differed from the holdout ranking: \
MobileNetV3-Large led at 41.53\\%, ResNet18 followed at 41.10\\%, \
MobileNetV3-Small was third at 38.98\\%, and EfficientNet-B0 was last at 34.32\\%. \
The surprising underperformance of EfficientNet-B0 on PlantDoc relative to its \
holdout accuracy may indicate that its compound scaling optimizes for \
distribution-specific features that do not transfer to natural images. \
Top-3 accuracy on PlantDoc ranged from 62.71\\% (EfficientNet-B0) to 68.22\\% \
(MobileNetV3-Large), suggesting that the correct class is typically ranked \
in the top three even when it is not the top prediction. \
CPU pipeline latency (Phase 4.1 screening) ranged from 662 ms (MobileNetV3-Small) \
to 2,457 ms (EfficientNet-B0), a 3.7$\\times$ range driven primarily by \
the number of crop inference calls in multi-scale screening. \
MobileNetV3-Small's 7.1$\\times$ latency advantage over ResNet18 \
(662 ms vs.~2,026 ms) makes it the preferred architecture for time-constrained \
field deployment despite its slightly lower absolute accuracy.\
"""

RES_SCREENING = """\
In development-set ablation experiments (Table 11), Config E achieved Recall@IoU 0.25 \
of 32.71\\% at 863.4 ms latency, while selected Hybrid NMS (Class 0.50, Cross 0.70, Max 10 candidates) \
achieved 34.57\\% dev recall (Table 12). When the selected composite pipeline was evaluated \
on the full independent PlantDoc test benchmark (236 images, 452 ground-truth leaves), \
it achieved Recall@IoU 0.25 of 38.05\\% (172 true positive leaves), Precision@IoU 0.25 of \
10.60\\% (172/1,623 proposals), F1 of 16.58\\%, and 6.15 false positives per image, at \
844.25 ms mean end-to-end CPU latency. This represents a 109.76\\% relative recall improvement \
over the Phase 4 baseline (18.14\\% recall at 167.02 ms end-to-end latency; 117.16 ms forward inference). \
Importantly, localization recall is class-agnostic: among the 172 true positive bounding boxes, \
exactly 66 matched the correct disease class, yielding 38.37\\% crop classification accuracy on \
localized candidates. Ground-truth annotations represent whole-leaf bounding boxes rather than \
pixel-level lesion masks; therefore, the pipeline functions as a foliar screening filter rather \
than a lesion-level severity estimator.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────

CONCLUSION = """\
This paper presented a reproducible experimental investigation of lightweight \
convolutional architectures for plant disease classification, calibration, \
and candidate-region screening under CPU-constrained conditions. \
The definitive MobileNetV3-Small checkpoint achieved 95.02\\% holdout accuracy \
and 93.13\\% Macro-F1 on the independent PlantVillage holdout partition, \
with temperature-scaled calibration reducing ECE from 10.25\\% to 0.58\\% \
(T*=0.5538, Phase 3 procedure). \
Independent external evaluation on 236 PlantDoc images exposed a \
56-percentage-point domain gap, with accuracy falling to 38.98\\% and \
Macro-F1 to 36.84\\%. This gap, consistent with published literature, \
demonstrates that PlantVillage accuracy is not a reliable predictor of \
field readiness. \
The Phase 4.1 optimized screening pipeline more than doubled Recall@IoU 0.25 \
(18.14\\% to 38.05\\%) through multi-scale windows, aspect-ratio awareness, \
and foliar saliency pre-filtering, but at a 5.1$\\times$ CPU latency cost \
and with elevated false-positive rates (6.15 per image). These trade-offs \
must be explicitly considered in deployment planning. \
The four-architecture comparison confirmed that MobileNetV3-Small provides \
the best latency-accuracy trade-off for CPU deployment, while \
MobileNetV3-Large leads on both holdout accuracy and PlantDoc generalization \
at the cost of a 2.4$\\times$ latency increase. \
A methodological calibration discrepancy between Phase 3 (ECE 0.58\\%) and \
Phase 5 (ECE 9.94\\%) procedures using the same checkpoint has been \
investigated and attributed to optimiser convergence differences; both \
results are documented transparently. \
The system is presented as a research screening prototype. It is not \
clinically validated, commercially certified, or suitable for autonomous \
pesticide application without expert agronomic review. Future work should \
address annotated lesion-level segmentation, plant/session-level dataset \
splits, domain adaptation for field images, and multi-seed repeated training \
to establish confidence intervals on reported metrics.\
"""

FUTURE_SCOPE = [
    "Execute repeated-seed training (seeds 42, 43, 44) for all four architectures to compute means, standard deviations, and 95% confidence intervals on holdout and PlantDoc metrics.",
    "Collect plant- and session-level identifiers for dataset grouping to enable proper plant-level train/test splits that prevent within-plant leakage.",
    "Develop and evaluate a domain-adapted model fine-tuned on PlantDoc training images to narrow the 56-percentage-point external accuracy gap.",
    "Annotate pixel-level lesion masks on a representative subset to enable lesion-level IoU, Dice coefficient, and affected-area estimation evaluation.",
    "Evaluate calibration on domain-shifted images using calibration partitions collected from natural field photographs rather than PlantVillage.",
    "Investigate lightweight object detection architectures (e.g., YOLO-Nano, NanoDet) as alternatives to sliding-window candidate generation for lower false-positive rates.",
]

LIMITATIONS = """\
This study has several explicit limitations. \
First, the training dataset lacks plant- and session-level identifiers, \
so the duplicate-aware split is image-level rather than plant-level. \
Duplicate leaves from the same plant may appear in different partitions, \
introducing potential within-plant leakage. \
Second, all results are from a single training seed; without repeated-seed \
experiments, confidence intervals on reported metrics cannot be established. \
Third, temperature scaling was fitted on PlantVillage calibration data; its \
failure to transfer to PlantDoc (ECE increases under domain shift) means \
that confidence scores cannot be reliably interpreted for field images. \
Fourth, candidate-region annotations are whole-leaf bounding boxes, \
not lesion-level masks; therefore localization metrics reflect leaf detection \
quality rather than disease lesion identification. \
Fifth, all experiments were conducted on a single CPU platform (AMD Ryzen 7 \
5800HS); latency results may not generalize to other hardware configurations.\
"""

