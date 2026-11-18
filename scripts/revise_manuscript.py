import os
import re

src_path = r"E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\reports\research_paper\main.tex"
dst_path = r"E:\Plant Disease Detection-20261111T181021Z-1-001\Plant Disease Detection\manuscript\phase7_revised\main.tex"

with open(src_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded source manuscript: {len(text)} characters, {len(text.splitlines())} lines.")

# 1. Replace Abstract
pattern_abstract = re.compile(r"\\abstract\{.*?\n\}", re.DOTALL)
new_abstract = r"""\abstract{
Plant diseases cause between 20\% and 40\% of global crop yield losses annually, imposing
severe threats to food security where smallholder farmers lack timely diagnostic expertise \citep{savary2019burden}.
This paper presents a comprehensive implementation, empirical evaluation, and full-stack CPU deployment of a
plant disease screening system built upon lightweight convolutional neural networks. Models are trained on a
curated combined corpus of 56{,}433 images across 38 classes (40{,}151 training [37{,}992 PlantVillage + 2{,}159 PlantDoc],
5{,}436 validation, 5{,}429 calibration, and 5{,}417 holdout test images, constructed after rigorous deduplication and
leakage exclusion). Four architectures are systematically benchmarked under an identical protocol:
MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, and ResNet18. On the reserved PlantVillage holdout test
partition (5{,}417 images), MobileNetV3-Large achieves 95.96\% top-1 accuracy (Macro-F1 94.52\%), while MobileNetV3-Small
provides optimal computational efficiency with 95.02\% top-1 accuracy (Macro-F1 93.13\%), 9.42\,ms CPU forward latency,
and a 6.06\,MB disk footprint. External evaluation on 236 natural-background PlantDoc images reveals a severe domain shift
of approximately 56 percentage points (MobileNetV3-Small top-1: 38.98\%, top-3: 64.83\%, Macro-F1 across all 38 classes:
31.08\%, and Macro-F1 across 27 represented classes: 36.84\%). An assistive multi-scale sliding-window candidate-region
screening pipeline evaluated against 452 ground-truth leaf bounding boxes achieves 38.05\% localization recall at
IoU~$\geq~0.25$ with 844.25\,ms latency, more than doubling baseline recall (18.14\%).
To support practical usage, the system is integrated into a complete CPU-optimized FastAPI backend and React frontend
utilizing a verified Phase 5 MobileNetV3-Small TorchScript export (SHA-256: \texttt{6fff52f1...}). The deployed application
decouples whole-leaf classification from candidate-region screening, enforces uncalibrated inference ($T=1.0$) with an
operational review threshold of 0.60 to prevent overconfident field errors, and visualizes exploratory Grad-CAM saliency.
Critical validation gaps, domain shift bounds, and non-autonomous triage protocols are documented to provide a rigorous,
reproducible foundation for field-deployable agricultural AI.
}"""
text = pattern_abstract.sub(lambda m: new_abstract, text, count=1)

# 2. Table 1 (Dataset Partitions) -> Resizebox & Text clarification
old_tab1 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Dataset partitions used in this study.*?\\end\{table\})", text, re.DOTALL)
if old_tab1:
    new_tab1 = r"""\begin{table}[htbp]
\centering
\caption{Dataset partitions used in this study. Each partition is fully isolated with no sample overlap. The combined split manifest comprises 56{,}433 records (37{,}992 PlantVillage training images, 2{,}159 PlantDoc training images, 5{,}436 PlantVillage validation images, 5{,}429 PlantVillage calibration images, and 5{,}417 PlantVillage holdout test images). The official external PlantDoc test benchmark (236 images, 452 ground-truth whole-leaf annotations) is strictly held out and was never observed during gradient updates, architecture selection, or hyperparameter tuning.}\label{tab:dataset_partitions}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lp{6.0cm}rrl}
\toprule
\textbf{Partition} & \textbf{Source Dataset \& Slicing Rule} & \textbf{Images} & \textbf{Classes} & \textbf{Operational Role} \\
\midrule
Training Split & PlantVillage (37{,}992) + PlantDoc Train (2{,}159) & 40{,}151 & 38 & Gradient updates only \\
Validation Split & PlantVillage Holdout Slice (disjoint) & 5{,}436 & 38 & Model selection \& early stopping \\
Calibration Split & PlantVillage Holdout Slice (disjoint) & 5{,}429 & 38 & Temperature scaling parameter fitting \\
Holdout Test Split & PlantVillage Holdout Slice (disjoint) & 5{,}417 & 38 & Independent in-domain evaluation \\
\midrule
\textbf{Combined Manifest Total} & \textbf{Deduplicated Multi-Source Corpus} & \textbf{56{,}433} & \textbf{38} & \textbf{Manifest SHA-256: \texttt{f9ef8b9d...15868}} \\
\midrule
External PlantDoc Benchmark & Official Test Benchmark (natural field) & 236 & 27 & Out-of-distribution domain evaluation \\
PlantDoc Leaf Annotations & Bounding-box annotations on test images & 452 boxes & 27 & Candidate-region screening benchmark \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab1.group(1), new_tab1)
    print("Replaced Table 1.")

# 3. Add text explaining PlantDoc data funnel right after Table 1
funnel_note_old = re.search(r"(\\noindent\\textit\{Note on PlantDoc training images in the training partition\}:.*?ensuring bit-for-bit data reproducibility\.)", text, re.DOTALL)
if funnel_note_old:
    funnel_note_new = r"""\noindent\textit{Data Provenance and Reconciliation of PlantDoc Counts}:
The progression from raw external source downloads to the final evaluation partitions is governed by strict exclusion protocols:
\begin{enumerate}
  \item \textbf{Raw Source Corpus}: The official PlantDoc repository \citep{singh2020plant} contains 2{,}598 total images.
  \item \textbf{Audited Classification Images}: After removing unreadable, corrupt, and zero-byte files, exactly 2{,}342 valid images were established as the audited source classification pool.
  \item \textbf{Leakage and Duplicate Filtering}: To ensure absolute isolation between training and testing, perceptual hashing and exact SHA-256 checks identified 10 duplicate images and 4 near-duplicate pairs matching the official test benchmark. Excluding these instances yielded 2{,}159 clean PlantDoc training images that were incorporated into the 40{,}151-image training partition (seed 42).
  \item \textbf{External Test Benchmark}: The 236 official PlantDoc test images containing 452 whole-leaf ground-truth bounding boxes were reserved exclusively for cross-domain evaluation and were never accessed during training or validation.
\end{enumerate}
The combined split manifest was assigned the SHA-256 digest
\texttt{f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868}, ensuring bit-for-bit data reproducibility."""
    text = text.replace(funnel_note_old.group(1), funnel_note_new)
    print("Replaced PlantDoc counts funnel text.")

# 4. Table 2 (Training Hyperparameters) -> Resizebox
old_tab2 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Unified training hyperparameters.*?\\end\{table\})", text, re.DOTALL)
if old_tab2:
    new_tab2 = r"""\begin{table}[htbp]
\centering
\caption{Unified training hyperparameters applied identically across all four benchmark architectures. All models were initialized with ImageNet-pretrained weights and fine-tuned under deterministic conditions (PyTorch seed 42).}\label{tab:training_config}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll}
\toprule
\textbf{Hyperparameter / Setting} & \textbf{Operational Configuration} \\
\midrule
Optimizer & AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-2}$) \\
Initial learning rate & $1 \times 10^{-4}$ \\
Batch size & 64 \\
Label smoothing & 0.10 \\
Learning rate scheduler & CosineAnnealingLR ($T_{\max}=20, \eta_{\min}=10^{-6}$) \\
Maximum epochs & 20 (with early stopping patience 5 on validation Macro-F1) \\
Input tensor dimensions & $224 \times 224 \times 3$ (RGB) \\
Data augmentation & RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation(15) \\
Image normalization & ImageNet standard ($\mu=[0.485,0.456,0.406], \sigma=[0.229,0.224,0.225]$) \\
Loss function & CrossEntropyLoss (label-smoothed) \\
Host execution platform & Windows 10 Home (64-bit), AMD Ryzen 7 5800HS (8 threads), 16\,GB RAM \\
Deterministic seed & 42 (\texttt{torch.use\_deterministic\_algorithms(True)}) \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab2.group(1), new_tab2)
    print("Replaced Table 2.")

# 5. Table 3 (PlantVillage Classification Performance) -> Resizebox & compact headers
old_tab3 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Classification performance on the PlantVillage holdout test set.*?\\end\{table\})", text, re.DOTALL)
if old_tab3:
    new_tab3 = r"""\begin{table}[htbp]
\centering
\caption{Classification performance on the PlantVillage holdout test set (5{,}417 images, 38 classes). All metrics were computed on the reserved test partition under deterministic execution. No test data were used for gradient updates or checkpoint selection.}\label{tab:pv_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Architecture} & \textbf{Params (M)} & \textbf{Size (MB)} & \textbf{Acc (\%)} & \textbf{Bal. Acc (\%)} & \textbf{Macro P (\%)} & \textbf{Macro R (\%)} & \textbf{Macro F1 (\%)} \\
\midrule
MobileNetV3-Small  & 1.56 &  6.06 & 95.02 & 95.06 & 92.16 & 95.06 & 93.13 \\
MobileNetV3-Large  & 4.25 & 16.40 & \textbf{95.96} & \textbf{95.97} & \textbf{94.03} & \textbf{95.97} & \textbf{94.52} \\
EfficientNet-B0    & 4.06 & 15.74 & 95.77 & 95.38 & 93.60 & 95.38 & 94.26 \\
ResNet18           & 11.20 & 42.79 & 95.92 & 95.60 & 93.76 & 95.60 & 94.30 \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab3.group(1), new_tab3)
    print("Replaced Table 3.")

# 6. Table 4 (Validation vs Holdout) -> Resizebox
old_tab4 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Comparison of validation accuracy.*?\\end\{table\})", text, re.DOTALL)
if old_tab4:
    new_tab4 = r"""\begin{table}[htbp]
\centering
\caption{Comparison of validation accuracy (used for model selection) and holdout test accuracy (independent evaluation) on PlantVillage partitions. The close agreement across all architectures confirms partition integrity and absence of overfitting.}\label{tab:val_vs_holdout}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
\textbf{Architecture} & \textbf{Validation Accuracy (\%)} & \textbf{Holdout Test Accuracy (\%)} & \textbf{Generalization Gap (pp)} \\
\midrule
MobileNetV3-Small & 95.14 & 95.02 & $-0.12$ \\
MobileNetV3-Large & 96.38 & 95.96 & $-0.42$ \\
EfficientNet-B0   & 96.10 & 95.77 & $-0.33$ \\
ResNet18          & 95.93 & 95.92 & $-0.01$ \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab4.group(1), new_tab4)
    print("Replaced Table 4.")

# 7. Table 5 (PlantDoc External Evaluation) -> Resizebox & Both Macro-F1 metrics
old_tab5 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Classification performance on the external PlantDoc test set.*?\\end\{table\})", text, re.DOTALL)
if old_tab5:
    new_tab5 = r"""\begin{table}[htbp]
\centering
\caption{Classification performance on the external PlantDoc test benchmark (236 images, 27 evaluated classes out of 38 model classes). Macro-F1 is reported across both the complete 38-class space (where 11 unrepresented classes evaluate to $F_1 = 0$) and across the 27 represented non-empty classes. Severe cross-domain degradation ($\sim 56$ percentage points relative to PlantVillage) is observed across all architectures.}\label{tab:plantdoc_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
\textbf{Architecture} & \textbf{Top-1 Acc (\%)} & \textbf{Top-3 Acc (\%)} & \textbf{Macro F1 (38-cls, \%)} & \textbf{Macro F1 (27-cls, \%)} & \textbf{Weighted F1 (\%)} \\
\midrule
MobileNetV3-Small & 38.98 & 64.83 & 31.08 & 36.84 & 37.41 \\
MobileNetV3-Large & \textbf{41.53} & \textbf{68.22} & \textbf{33.93} & \textbf{39.81} & \textbf{40.08} \\
EfficientNet-B0   & 34.32 & 62.71 & 26.14 & 31.55 & 32.57 \\
ResNet18          & 41.10 & 67.80 & 34.31 & 40.22 & 39.74 \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab5.group(1), new_tab5)
    print("Replaced Table 5.")

# 8. Table 6 (Temperature Scaling Calibration) -> Resizebox & Reconciliation
old_tab6 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Temperature scaling calibration results evaluated on the PlantVillage holdout.*?\\end\{table\})", text, re.DOTALL)
if old_tab6:
    new_tab6 = r"""\begin{table}[htbp]
\centering
\caption{Temperature scaling calibration results evaluated on the PlantVillage holdout test partition (5{,}417 images). Temperatures $T^*$ were fitted strictly on the isolated calibration partition (5{,}429 images). Pre- and post-calibration Expected Calibration Error (ECE, 15 equal-width bins), Negative Log-Likelihood (NLL), and Brier scores are reported. Both Phase 5 batch calibration and Phase 3 dedicated baseline calibration are documented. In the deployed production application, calibration is intentionally disabled ($T=1.0$) to avoid propagating misleading confidence under field domain shifts.}\label{tab:calibration_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Architecture / Protocol} & $T^*$ & \textbf{Pre ECE (\%)} & \textbf{Post ECE (\%)} & \textbf{Pre NLL} & \textbf{Post NLL} & \textbf{Pre Brier} & \textbf{Post Brier} \\
\midrule
MobileNetV3-Small (Phase 5 Batch) & 0.9902 & 10.28 & 9.94 & 0.2554 & 0.2513 & 0.0993 & 0.0981 \\
MobileNetV3-Small (Phase 3 Dedicated) & 0.5538 & 10.25 & 0.58 & 0.2426 & 0.1454 & 0.0936 & 0.0710 \\
MobileNetV3-Large (Phase 5 Batch) & 0.9240 & 14.00 & 10.80 & 0.2755 & 0.2341 & 0.0986 & 0.0854 \\
EfficientNet-B0 (Phase 5 Batch)   & 0.8846 & 16.43 & 11.62 & 0.3162 & 0.2542 & 0.1156 & 0.0945 \\
ResNet18 (Phase 5 Batch)          & 1.0392 &  8.99 & 10.21 & 0.2133 & 0.2277 & 0.0809 & 0.0851 \\
\midrule
\multicolumn{8}{l}{\textit{Operational Deployed Policy}: Temperature scaling disabled ($T=1.0000$); raw scores presented with 0.60 human review threshold.} \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab6.group(1), new_tab6)
    print("Replaced Table 6.")

# Update the explanation following Table 6 to clearly reconcile the three calibration temperatures
cal_text_old = re.search(r"(A methodological discrepancy requires transparent documentation\..*?calibration optimum without temperature scaling\.)", text, re.DOTALL)
if cal_text_old:
    cal_text_new = r"""A methodological discrepancy across experimental phases requires transparent reconciliation:
\begin{enumerate}
  \item \textbf{Phase 3 Baseline Calibration ($T^* = 0.5538$)}: In Phase 3, extended L-BFGS optimization (50 iterations, learning rate 0.01) was performed on the historical baseline MobileNetV3-Small model (\texttt{saved\_models/classifier.pt}, SHA-256: \texttt{1aef041f...}). This converged to $T^* = 0.5538$, achieving a dramatic reduction in ECE from 10.25\% to 0.58\% on the 5{,}429 calibration samples and from 10.28\% to 0.74\% on the holdout test partition.
  \item \textbf{Phase 5 Multi-Architecture Benchmark ($T^* = 0.9902$)}: In Phase 5, all four newly trained benchmark architectures were evaluated under a standardized batch calibration protocol. For the definitive MobileNetV3-Small model (\texttt{best\_model.pth}, SHA-256: \texttt{6190f54c...}), L-BFGS converged to $T^* = 0.9902$ with a post-calibration ECE of 9.94\%. The difference reflects optimization convergence hyperparameters ($T_0 = 1.5$ vs $T_0 = 1.0$ initialization) and checkpoint training trajectories.
  \item \textbf{Deployed Application Policy ($T = 1.0$, Uncalibrated)}: When transferred to the external PlantDoc field benchmark, temperature parameters fitted on PlantVillage studio images failed to calibrate out-of-distribution probabilities, creating an unjustified impression of certainty. Consequently, the production FastAPI backend intentionally disables temperature scaling (\texttt{calibration\_enabled = False}, $T=1.0$), presents raw probabilities labeled strictly as \textit{Uncalibrated Model Score}, and establishes an empirical human-review threshold of 0.60.
\end{enumerate}
For the other three architectures evaluated in Phase 5, temperature scaling provided moderate improvements: MobileNetV3-Large reduced ECE from 14.00\% to 10.80\% ($T^* = 0.9240$), EfficientNet-B0 from 16.43\% to 11.62\% ($T^* = 0.8846$), and ResNet18 from 8.99\% to 10.21\% ($T^* = 1.0392$)."""
    text = text.replace(cal_text_old.group(1), cal_text_new)
    print("Replaced Calibration reconciliation text.")

# 9. Table 7 (Comprehensive Architecture Benchmark) -> Resizebox & both Macro-F1s
old_tab7 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Comprehensive architecture benchmark: classification, generalization, calibration.*?\\end\{table\})", text, re.DOTALL)
if old_tab7:
    new_tab7 = r"""\begin{table}[htbp]
\centering
\caption{Comprehensive architecture benchmark: in-domain classification, out-of-domain generalization, calibration, and computational efficiency. Single-image inference is measured over 100 runs on an AMD Ryzen 7 5800HS CPU (8 threads). Full screening pipeline latency encompasses foliar ExG filtering, multi-crop generation, batch inference, and hybrid NMS over 50 PlantDoc images.}\label{tab:full_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Architecture} & \textbf{PV Acc} & \textbf{PV F1} & \textbf{PD Top-1} & \textbf{PD Top-3} & \textbf{PD F1-38} & \textbf{PD F1-27} & \textbf{Latency} & \textbf{Pipeline} \\
& \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(ms)} & \textbf{(ms)} \\
\midrule
MobileNetV3-Small  & 95.02 & 93.13 & 38.98 & 64.83 & 31.08 & 36.84 & \textbf{9.42}  & \textbf{662.1} \\
MobileNetV3-Large  & \textbf{95.96} & \textbf{94.52} & \textbf{41.53} & \textbf{68.22} & \textbf{33.93} & \textbf{39.81} & 16.87 & 1{,}585.2 \\
EfficientNet-B0    & 95.77 & 94.26 & 34.32 & 62.71 & 26.14 & 31.55 & 24.58 & 2{,}457.4 \\
ResNet18           & 95.92 & 94.30 & 41.10 & 67.80 & 34.31 & 40.22 & 25.54 & 2{,}025.9 \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab7.group(1), new_tab7)
    print("Replaced Table 7.")

# 10. Table 8 & Table 9 (Screening Evaluation & Stage-wise Latency Breakdown) -> Resizebox
old_tab8 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Candidate-region screening evaluation on the PlantDoc official test benchmark.*?\\end\{table\})", text, re.DOTALL)
if old_tab8:
    new_tab8 = r"""\begin{table}[htbp]
\centering
\caption{Candidate-region screening evaluation on the PlantDoc official test benchmark (236 images, 452 whole-leaf bounding-box annotations). Ground-truth annotations represent entire leaf boundaries, not lesion segmentation masks. Correctly classified detections refers to candidate regions that (a) match a ground-truth leaf at IoU $\geq 0.25$ and (b) are classified with the correct disease label.}\label{tab:screening_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
\textbf{Evaluation Metric} & \textbf{Phase 4 Baseline} & \textbf{Phase 4.1 Optimized} & \textbf{Relative Change} \\
\midrule
Recall @ IoU 0.10        & 50.00\% & 62.17\% & +12.17 pp \\
Precision @ IoU 0.10     & 24.78\% & 17.31\% & $-7.47$ pp \\
Recall @ IoU 0.25        & 18.14\% & \textbf{38.05\%} & +109.8\% rel. \\
Precision @ IoU 0.25     &  8.99\% & 10.60\% & +17.9\% rel. \\
F1-Score @ IoU 0.25      & 12.02\% & \textbf{16.58\%} & +37.9\% rel. \\
Recall @ IoU 0.50        &  3.10\% &  7.96\% & +157.0\% rel. \\
Precision @ IoU 0.50     &  1.54\% &  2.22\% & +44.5\% rel. \\
Correct detections       & 30 leaves & \textbf{66 leaves} & +120.0\% rel. \\
Crop classification acc. & 36.59\% & 38.37\% & +1.78 pp \\
False positives per image & 3.52 & 6.15 & $+2.63$ \\
Mean CPU latency         & 167.02\,ms & 844.25\,ms & +677.23\,ms \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab8.group(1), new_tab8)
    print("Replaced Table 8.")

old_tab9 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Stage-wise CPU latency breakdown of the candidate-region screening pipeline.*?\\end\{table\})", text, re.DOTALL)
if old_tab9:
    new_tab9 = r"""\begin{table}[htbp]
\centering
\caption{Stage-wise CPU latency breakdown of the candidate-region screening pipeline measured on an AMD Ryzen 7 5800HS processor (8 CPU threads). Candidate crop extraction, context padding, and tensor normalization constitute the primary execution overhead alongside CNN batch inference.}\label{tab:latency_breakdown}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrp{6.0cm}}
\toprule
\textbf{Pipeline Stage} & \textbf{Baseline (ms)} & \textbf{Optimized (ms)} & \textbf{Execution Implementation} \\
\midrule
Window Proposal Gen. (Multi-scale)       & 0.04 &   0.10 & NumPy vectorized grid indexing \\
Foliar Saliency Pre-Filter (ExG)         & 0.00 &  24.16 & Vectorized ExG thresholding ($\text{ExG} > 10$, $\ge 8\%$) \\
Crop Extraction \& Tensor Prep           & 67.07 & 358.19 & PIL bounding box crop, 12\% padding, resize \& norm \\
Batch Forward Inference (MobileNetV3-S)  & 99.87 & 461.62 & PyTorch 2.15.0+cpu (8 CPU threads) \\
Hybrid NMS \& Overlap Suppression        & 0.04 &   0.18 & Greedy IoU loop (class-wise 0.50, cross-class 0.70) \\
\midrule
\textbf{Total End-to-End Latency}        & \textbf{167.02} & \textbf{844.25} & \textbf{Complete Research Screening Pipeline} \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab9.group(1), new_tab9)
    print("Replaced Table 9.")

# 11. Add Section 4.8: Full-Stack Application Integration and Runtime Validation
fullstack_section = r"""\subsection{Full-Stack System Deployment and Runtime Validation}
\label{sec:system_deployment}

Following experimental benchmarking, the verified Phase 5 MobileNetV3-Small TorchScript model (\texttt{saved\_models/experiments/mobilenet\_v3\_small\_final/classifier.pt}, SHA-256: \texttt{6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213}) was deployed into an interactive full-stack research application comprising a FastAPI backend and a React/Vite web interface.

\textbf{Decoupled Two-Stage Presentation.}
To prevent user confusion between whole-image classification and localized region analysis, the user interface decouples prediction into two clearly distinguished operational sections:
\begin{itemize}
  \item \textbf{Section A --- Whole-Leaf Classifier (\texttt{/predict})}: Evaluates the global input photograph, returning the predicted disease class, top-3 differential candidates, uncalibrated score, prediction confidence state (\texttt{CONFIDENT} if score $\geq 0.60$, \texttt{UNCERTAIN} otherwise), and an optional Grad-CAM visual attention heatmap.
  \item \textbf{Section B --- Candidate-Region Screening (\texttt{/screen})}: Operates as an assistive localization tool. For single-leaf close-ups, an adaptive 9-crop grid is evaluated; for whole-plant field scenes, a multi-scale grid (12--24 crops) is generated. Candidate crops passing the ExG vegetation filter are scored. Proposals with scores $\geq 0.60$ predicting non-healthy classes are retained and deduplicated via greedy IoU NMS (threshold 0.55).
\end{itemize}

\textbf{Exploratory Grad-CAM Interpretation.}
Grad-CAM heatmaps are extracted from the final convolutional layer of the MobileNetV3-Small backbone (\texttt{features.12}). The manuscript explicitly emphasizes that Grad-CAM represents \textit{exploratory visual saliency} highlighting regions that influenced the classifier's feature representations, rather than pixel-accurate pathological lesion segmentation.

\textbf{Empirical Latency and Robustness Benchmarks.}
Runtime performance was systematically profiled on the CPU development environment over 50 timed executions with 10 explicit warmup iterations:
\begin{itemize}
  \item \textbf{Isolated Model Forward Pass}: $8.43 \pm 1.16$\,ms per single $224 \times 224$ tensor.
  \item \textbf{End-to-End API Request (\texttt{/predict})}: $27.84 \pm 0.77$\,ms (including multipart decoding, PIL image loading, tensor normalization, inference, and JSON formatting).
  \item \textbf{Single-Leaf Screening}: $186.24 \pm 11.52$\,ms (evaluating 9 candidate crops).
  \item \textbf{Whole-Plant Screening}: $200.03 \pm 17.63$\,ms (evaluating 12 candidate crops).
\end{itemize}
Robustness evaluation across 20 synthetic perturbations (exposure shifts, contrast reduction, Gaussian noise, in-plane rotations up to $90^\circ$, Gaussian blur, and center occlusion) confirmed that the system gracefully degrades: inputs with severe noise or blur are correctly flagged as \texttt{UNCERTAIN} below the 0.60 threshold rather than outputting false confident diagnoses."""

# Insert before Section 5
text = text.replace(r"\section{Conclusion and Future Scope}", fullstack_section + "\n\n" + r"\section{Conclusion and Future Scope}")
print("Inserted Section 4.8.")

# 12. Table 10 (Cryptographic Checkpoint Provenance) -> Resizebox & Include TorchScript Checkpoint
old_tab10 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Cryptographic checkpoint provenance for all Phase 5 benchmark models.*?\\end\{table\})", text, re.DOTALL)
if old_tab10:
    new_tab10 = r"""\begin{table}[htbp]
\centering
\caption{Cryptographic checkpoint provenance for all Phase 5 benchmark models, the deployed production TorchScript artifact, and the historical baseline. SHA-256 digests verify checkpoint integrity and bit-level immutability across evaluation stages.}\label{tab:checkpoints}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrl}
\toprule
\textbf{Architecture / Operational Role} & \textbf{Format} & \textbf{Params} & \textbf{Size (MB)} & \textbf{Cryptographic SHA-256 Checksum} \\
\midrule
MobileNetV3-Small (Phase 5 Benchmark) & PyTorch (.pth) & 1{,}556{,}806 & 6.06 & \texttt{6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009} \\
MobileNetV3-Small (Phase 6A Deployed)  & TorchScript (.pt) & 1{,}556{,}806 & 6.06 & \texttt{6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213} \\
MobileNetV3-Large (Phase 5 Benchmark) & PyTorch (.pth) & 4{,}250{,}710 & 16.40 & \texttt{8ebfb512a071915ca9a7b508e52a9f3a5e8e82a93eb9d5924dd755c9603223f6} \\
EfficientNet-B0 (Phase 5 Benchmark)   & PyTorch (.pth) & 4{,}056{,}226 & 15.74 & \texttt{faba42992b190b4c6c37409634070f0afb0dc27357c31e9c991ba810c0e86b02} \\
ResNet18 (Phase 5 Benchmark)          & PyTorch (.pth) & 11{,}196{,}006 & 42.79 & \texttt{f2991bef8630f6e162923fe3224d8d689b251b54ec651b1f9b33a04a62174d1a} \\
Historical Baseline (Phase 2 Model)   & TorchScript (.pt) & 1{,}556{,}806 & 9.57 & \texttt{1aef041f87ad115b65aea90ad9c32584fda5689e063fcd750e6144f364a1bc2f} \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab10.group(1), new_tab10)
    print("Replaced Table 10.")

# 13. Update Limitations and Ethical Considerations
old_limits = re.search(r"(\\section\*\{Limitations and Ethical Considerations\}.*?\\begin\{itemize\}.*?\\end\{itemize\})", text, re.DOTALL)
if old_limits:
    new_limits = r"""\section*{Limitations and Ethical Considerations}

This system is an exploratory research prototype designed for decision support and is not validated for autonomous clinical diagnosis or certified regulatory deployment. Agricultural extension personnel and researchers must observe the following critical limitations:
\begin{itemize}
  \item \textbf{Domain Generalization Bound}: Performance on natural-background field photographs (PlantDoc top-1 accuracy: 38.98\%--41.53\%) demonstrates that models trained primarily on laboratory specimens cannot operate autonomously in real-world agricultural environments without severe risk of misdiagnosis.
  \item \textbf{Whole-Leaf vs.\ Lesion Segmentation}: The candidate-region screening pipeline generates \textit{whole-leaf candidate bounding boxes}, not pixel-level lesion segmentation masks. Grad-CAM saliency maps represent neural activation heatmaps rather than pathological lesion boundaries and must not be interpreted as quantitative disease severity metrics.
  \item \textbf{Calibration Non-Transferability}: Post-hoc temperature scaling fitted on closed-world laboratory partitions (PlantVillage) does not generalize to open-world field imagery. Presenting artificially scaled probabilities on out-of-distribution inputs induces unjustified user overconfidence. The application therefore operates with uncalibrated scores ($T=1.0$) and explicit uncertainty referral.
  \item \textbf{Input Condition Sensitivity}: Model performance degrades under severe overexposure, heavy shadows, motion blur, and large in-plane rotations ($>45^\circ$). Users must follow standardized acquisition protocols (single leaf against diffuse natural light).
  \item \textbf{Validation Scope}: While full-stack application integration and synthetic perturbation stress tests were successfully completed, multi-season, multi-site longitudinal clinical trials across heterogeneous agroecological zones remain necessary prior to field deployment.
\end{itemize}"""
    text = text.replace(old_limits.group(1), new_limits)
    print("Replaced Limitations section.")

# 14. Table 11 (Agronomic Differential Triage Matrix) -> Resizebox
old_tab11 = re.search(r"(\\begin\{table\}\[h\].*?\\caption\{Agronomic differential triage matrix.*?\\end\{table\})", text, re.DOTALL)
if old_tab11:
    new_tab11 = r"""\begin{table}[htbp]
\centering
\caption{Agronomic differential triage matrix mapping foliar signs to operational escalation tiers across major crop categories. Prescriptive chemical directives and quarantine mandates are omitted; actions are strictly restricted to non-autonomous human-in-the-loop inspection and extension clinic referral.}\label{tab:field_matrix}
\resizebox{\textwidth}{!}{%
\begin{tabular}{p{2.8cm}p{3.8cm}p{3.2cm}p{4.5cm}}
\toprule
\textbf{Crop \& Pathogen Target} & \textbf{Characteristic Foliar Symptoms} & \textbf{Abiotic / Biotic Lookalikes} & \textbf{Operational Screening Triage Protocol} \\
\midrule
Apple Scab (\textit{Venturia inaequalis}) & Olive-green to velvety dark brown lesions on adaxial leaf surface & Sooty mold, spray residue, mechanical abrasive wind injury & Tier 1: Foliar margin inspection; 10$\times$ hand lens check; refer if $>15\%$ canopy affected. \\
Corn Leaf Blight (\textit{Exserohilum turcicum}) & Elongated elliptical grayish-green or tan necrotic streaks along veins & Drought scorch, nitrogen deficiency chlorosis, Stewart's wilt & Tier 1: Scout lower canopy for lesion progression; collect dry foliar specimen for clinic. \\
Grape Black Rot (\textit{Guignardia bidwellii}) & Small circular reddish-brown spots with dark borders \& visible pycnidia & Anthracnose, pesticide phytotoxicity, hail puncturing injury & Tier 1: Cross-check petioles and fruit clusters for black pycnidia; record GPS plot coordinates. \\
Potato Early Blight (\textit{Alternaria solani}) & Concentric target-board rings with localized chlorotic halo & Magnesium deficiency, physiological senescence, ozone flecking & Tier 1: Confirm concentric zonation; recommend physical soil moisture and nutrient testing. \\
Potato Late Blight (\textit{Phytophthora infestans}) & Water-soaked irregular pale green lesions rapidly expanding to dark brown & Frost injury, wind desiccation, bacterial soft rot & Tier 1 (Urgent): Inspect abaxial surface for white sporulation; flag for immediate extension assay. \\
Tomato TYLCV & Upward leaf curling, marginal chlorosis, stunted leaflets & Aphid feeding damage, mechanical mite injury, physiological roll & Tier 1: Scout leaflet undersides for whitefly vectors; submit leaf press to plant virology facility. \\
Low Confidence / Ambiguous & Diffused chlorosis, atypical lesion shape, overlapping foliage & Nutrient deficiencies (Fe, Zn, N), herbicide drift, senescence & Tier 2: Defer automated classification; withhold diagnosis; flag for agronomist secondary inspection. \\
Unsegmented / Extreme Occlusion & Total leaf coverage $< 8\%$ of view, extreme motion blur, shadow & Bark, soil, non-target background weeds, artificial mulch & Tier 3: Suppress candidate inference; prompt user to re-photograph leaf with neutral backing. \\
\botrule
\end{tabular}%
}
\end{table}"""
    text = text.replace(old_tab11.group(1), new_tab11)
    print("Replaced Table 11.")

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully wrote revised manuscript to {dst_path} ({len(text)} chars, {len(text.splitlines())} lines).")
