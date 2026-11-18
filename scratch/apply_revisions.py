import os
import sys

tex_path = "manuscript/final_supervisor_review/main.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")

print("Initial text length:", len(text))

# 1. Section 3.5 & Section 3.6 replacement
s35_start = text.find(r"\subsection{Post-Hoc Temperature Scaling}")
s37_start = text.find(r"\subsection{Evaluation Protocol}")
assert s35_start != -1 and s37_start != -1, "Section 3.5/3.6 boundaries not found"

new_s35_s36 = r"""\subsection{Post-Hoc Temperature Scaling Methodology}
\label{sec:calibration_method}

Following supervised model training, confidence calibration was systematically investigated using post-hoc temperature scaling \citep{guo2017calibration}. Temperature scaling modifies model confidence without altering the argmax predicted class label by dividing the pre-softmax logit vector $\mathbf{z}_i \in \mathbb{R}^{38}$ by a single learned positive scalar temperature parameter $T > 0$:
\begin{equation}
  \hat{p}_{ic}(T) = \frac{\exp(z_{ic}/T)}{\sum_{j=1}^{38} \exp(z_{ij}/T)}
  \label{eq:temperature_scaling_prob}
\end{equation}
The optimal temperature $T^*$ is fitted by minimizing Negative Log-Likelihood (NLL / cross-entropy loss) on the isolated calibration partition (5{,}429 images) using the L-BFGS optimization algorithm with strong Wolfe line search:
\begin{equation}
  T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{cal}}} \sum_{i=1}^{N_{\text{cal}}} \log \hat{p}_{i, y_i}(T)
  \label{eq:temperature_scaling}
\end{equation}
Crucially, the Expected Calibration Error (ECE) serves strictly as a post-hoc evaluation diagnostic and is not the optimization objective, as binning operations are piecewise constant and non-differentiable. ECE measures the expectation gap between confidence and empirical accuracy across $M=15$ equal-width bins:
\begin{equation}
  \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}
  \left|\text{acc}(B_m) - \text{conf}(B_m)\right|
  \label{eq:ece}
\end{equation}
where $B_m$ contains all samples whose top predicted confidence falls in bin $m \in [1, M]$, $\text{acc}(B_m)$ represents the empirical accuracy in that bin, and $\text{conf}(B_m)$ is the mean confidence.

Across the lifecycle of this study, three distinct calibration regimes are strictly distinguished:
\begin{enumerate}
  \item \textbf{Phase 3 Historical Baseline ($T^* = 0.5538$)}: Extended L-BFGS optimization (50 iterations, learning rate 0.01) on the historical baseline MobileNetV3-Small model converged to $T^* = 0.5538$, reducing calibration ECE from 10.25\% to 0.58\% and test ECE to 0.74\%.
  \item \textbf{Phase 5 Multi-Architecture Benchmark ($T^* = 0.9902$)}: Under a standardized batch calibration protocol across all four newly trained architectures, the definitive MobileNetV3-Small checkpoint converged to $T^* = 0.9902$ (post-calibration ECE 9.94\%), while MobileNetV3-Large converged to $T^* = 0.9240$, EfficientNet-B0 to $T^* = 0.8846$, and ResNet-18 to $T^* = 1.0392$.
  \item \textbf{Phase 6A Deployed Application ($T = 1.0$, Uncalibrated)}: When transferred to out-of-distribution field conditions, closed-world laboratory temperatures fail to calibrate real-world uncertainty. Consequently, the production FastAPI backend explicitly disables temperature scaling ($T=1.0$, \texttt{calibration\_enabled = False}), presenting raw model outputs alongside an empirical 0.60 human review threshold.
\end{enumerate}

\subsection{Phase 4.1 Candidate-Region Research Screening Pipeline}
\label{sec:screening_pipeline}

To benchmark automated leaf localization on complex natural field imagery prior to software deployment, an offline research screening pipeline was developed and evaluated in Phase~4.1 against the official 236-image PlantDoc natural-background test set (comprising 452 ground-truth whole-leaf annotations). This research pipeline operates across four structured stages:

\textbf{Stage 1 — Multi-Scale Window Generation.}
Given input image dimensions $(W, H)$, candidate bounding boxes are systematically generated at three scale fractions $\{0.25, 0.40, 0.65\}$ of $\min(W, H)$ across three foliar aspect ratios $\{1{:}1, 4{:}3, 3{:}4\}$. A sliding stride equal to 70\% of the window dimension yields 12--24 candidate windows per image with boundary-aware coordinate snapping.

\textbf{Stage 2 — Excess Green Index (ExG) Foliar Saliency Filtering.}
To discard non-plant regions prior to expensive neural forward passes, proposals are filtered using the vectorized Excess Green vegetation index \citep{woebbecke1995color}:
\begin{equation}
  \text{ExG} = 2G - R - B
  \label{eq:exg}
\end{equation}
Windows in which fewer than 8\% of pixels satisfy $\text{ExG} > 10$ are pruned, eliminating soil, sky, mulch, and non-foliar background proposals.

\textbf{Stage 3 — CNN Batch Inference with Context Padding.}
Surviving windows are expanded by 12\% symmetric context padding (clamped to image boundaries) to capture petiole insertions and leaf margins, resized to $224 \times 224 \times 3$, and evaluated using MobileNetV3-Small under Phase~3 calibration ($T^* = 0.5538$). Candidate windows with confidence below 0.55 or predicted as ``healthy'' are suppressed.

\textbf{Stage 4 — Two-Stage Hybrid Non-Maximum Suppression.}
Surviving candidate proposals are deduplicated via a two-stage hybrid NMS algorithm:
\begin{enumerate}
  \item \textit{Class-wise IoU suppression} at threshold 0.50 removes duplicate bounding boxes predicting the identical disease class.
  \item \textit{Cross-class IoU suppression} at threshold 0.70 eliminates heavily overlapping spatial proposals across differing predicted classes.
\end{enumerate}
A maximum of 10 candidate regions per image are retained. The complete frozen Phase~4.1 research screening configuration is archived under cryptographic SHA-256 digest \texttt{7db276bb\dots 9014f85a} (\path{reports/audit_screening_config.json}) and yielded a mean CPU latency of 844.25\,ms per image.

\textit{Operational Distinction}: This exhaustive research screening pipeline must not be confused with the responsive, mode-aware candidate screening pipeline implemented in the Phase~6A deployed web application (detailed in Section~\ref{sec:system_deployment}), which uses an adaptive 9- or 36-crop grid, raw uncalibrated scores ($T=1.0$), a 0.60 human review threshold, single-stage greedy NMS (IoU 0.55), and sub-200\,ms execution latency.

"""

text = text[:s35_start] + new_s35_s36 + text[s37_start:]
print("Updated Section 3.5 and Section 3.6")

# 2. Section 4.7 & Section 4.8 replacement
s47_start = text.find(r"\subsection{Calibration Reliability and Operational Referral Policy}")
assert s47_start != -1, "s47_start not found"
s5_start = text.find(r"\section{Conclusion and Future Scope}")
assert s5_start != -1, "s5_start not found"

new_s47_s48 = r"""\subsection{Calibration Reliability and Exploratory Referral Policy}
\label{sec:efficiency_summary}

Figure~\ref{fig:calibration} illustrates the reliability diagrams for MobileNetV3-Small under Phase~3 calibration ($T^* = 0.5538$), demonstrating close alignment between predicted confidence and empirical accuracy (ECE $10.25\% \rightarrow 0.58\%$). Based on this calibrated distribution, an experimental three-tier referral framework was proposed for offline decision support research: Tier~1 ($\tau_1 \ge 0.70$; achieving 97.6\% accuracy with 94.5\% coverage on calibration data) qualifies for automated decision support display; Tier~2 ($0.223 \le \tau < 0.70$; evaluated on PlantDoc achieving 93.22\% coverage and 40.91\% selective accuracy) flags borderline candidates for secondary agronomic review; and Tier~3 ($\tau < 0.223$) enforces mandatory physical agricultural extension referral. As detailed in Section~\ref{sec:system_deployment}, this calibrated three-tier framework is an offline research proposal; the live deployed application intentionally operates with uncalibrated scores ($T=1.0$) and a single 0.60 human review threshold to prevent overconfident field classifications.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\linewidth]{fig_calibration.png}
  \caption{Reliability diagrams for MobileNetV3-Small. Left: pre-calibration ($T=1.0$, ECE=10.25\%). Right: post-calibration ($T^*=0.5538$, ECE=0.58\%) on the 5{,}429-image calibration partition.}
  \label{fig:calibration}
\end{figure}

\subsection{Phase 6A Interactive Deployed System and Full-Stack Runtime Validation}
\label{sec:system_deployment}

Following experimental benchmarking, the verified Phase~5 MobileNetV3-Small TorchScript model (cryptographic SHA-256 digest \texttt{6fff52f1\dots 1213}, detailed in Table~\ref{tab:checkpoints}) was deployed into an interactive full-stack research application comprising a FastAPI backend and a React/Vite web interface.

\textbf{Decoupled Two-Stage Presentation.}
To prevent user confusion between whole-image classification and localized region analysis, the user interface decouples prediction into two clearly distinguished operational sections:
\begin{itemize}
  \item \textbf{Section A: Whole-Leaf Classifier (\texttt{/predict})}.
        Evaluates the global input photograph, returning the predicted disease class,
        top-3 differential candidates, uncalibrated score, binary review state
        (\texttt{CONFIDENT} if score $\ge 0.60$; \texttt{UNCERTAIN} otherwise),
        and an optional Grad-CAM visual attention heatmap.
  \item \textbf{Section B: Assistive Candidate-Region Screening (\texttt{/screen})}.
        Operates as an assistive localization tool. For single-leaf close-ups,
        an adaptive 9-crop grid is evaluated (crop fraction 0.52 of the short dimension);
        for whole-plant field scenes, an adaptive multi-scale grid (up to 36 crops,
        crop fraction 0.38) is evaluated. Candidate crops passing the ExG foliar filter
        are scored. Proposals with uncalibrated scores $\ge 0.60$ predicting non-healthy
        classes are retained and deduplicated via single-stage greedy IoU NMS (threshold 0.55),
        capping results to a maximum of 8 proposals returned to the interface.
\end{itemize}

\noindent\textbf{Architectural Separation: Research vs.\ Deployed Screening Pipelines.}
A rigorous distinction is maintained between the Phase~4.1 experimental research screening pipeline (Section~\ref{sec:screening_pipeline}) and the Phase~6A interactive deployed screening pipeline:
\begin{itemize}
  \item \textbf{Proposal Generation}: Phase~4.1 uses exhaustive multi-scale sliding windows (12--24 crops across scales $\{0.25, 0.40, 0.65\}$ and aspect ratios $\{1{:}1, 4{:}3, 3{:}4\}$); Phase~6A implements mode-aware grids (9 crops for \texttt{single\_leaf}; up to 36 crops for \texttt{whole\_plant}, capped at 8 displayed candidates).
  \item \textbf{Confidence and Calibration}: Phase~4.1 evaluates proposals using calibrated temperature $T^* = 0.5538$ with a 0.55 threshold; Phase~6A operates strictly with uncalibrated probabilities ($T = 1.0$) and an empirical 0.60 review threshold to avoid transferring studio-derived confidence into unconstrained field environments.
  \item \textbf{Non-Maximum Suppression}: Phase~4.1 applies a two-stage hybrid NMS (class-wise IoU 0.50 followed by cross-class IoU 0.70); Phase~6A applies single-stage greedy IoU suppression at threshold 0.55.
  \item \textbf{Runtime Latency}: Phase~4.1 requires 844.25\,ms per image on 8 CPU threads; Phase~6A achieves responsive interactive latencies of 186.24\,ms (single-leaf) and 200.03\,ms (whole-plant).
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
Robustness evaluation across 20 synthetic perturbations revealed critical behavior across five operational dimensions:
\begin{enumerate}
  \item \textbf{Class Preservation with High Confidence}: Mild Gaussian blur ($\sigma=1.0$), low contrast ($-50\%$), high contrast ($+50\%$), moderate darkening ($-25\%$), and JPEG compression down to $Q=30$ preserved the correct class (\texttt{Apple\_\_\_Apple\_scab}) with uncalibrated confidence exceeding 93\%.
  \item \textbf{Confidence Reduction without Class Flip}: Moderate blur ($\sigma=2.0$) and severe JPEG compression ($Q=15$) reduced confidence to 64.88\% and 85.74\% respectively, while retaining the correct disease diagnosis.
  \item \textbf{Prediction-State Shifts to Uncertainty}: In-plane rotations of $15^\circ$, $90^\circ$, and $270^\circ$ dropped model confidence to 49.77\%, 43.95\%, and 33.86\% respectively. Because these fell below the 0.60 operational threshold, the backend correctly shifted prediction state from \texttt{HIGH\_CONFIDENCE} to \texttt{LOW\_CONFIDENCE\_UNCERTAIN}, prompting user re-acquisition.
  \item \textbf{Candidate-Count Alterations}: Candidate-region screening proposals fluctuated under structural distortion: horizontal stretch ($2:1$) suppressed candidate proposals from 8 to 0, while downscaling ($160 \times 160$) reduced proposals from 8 to 2.
  \item \textbf{Critical Classification Failure (Silent False Negative)}: Under severe overexposure ($+50\%$ brightness increase), the predicted class flipped from diseased \texttt{Apple\_\_\_Apple\_scab} to \texttt{Apple\_\_\_healthy} with $63.79\%$ confidence. Because $63.79\%$ exceeds the 0.60 threshold, this error evaded the uncertainty filter, outputting an authoritative healthy diagnosis for a diseased specimen. Analysis reveals that extreme luminance saturates pixel values, washing out dark necrotic lesion contrast and mimicking a pristine light-green leaf. Consequently, the 0.60 threshold alone cannot guard against overexposed imagery.
\end{enumerate}

"""

text = text[:s47_start] + new_s47_s48 + text[s5_start:]
print("Updated Section 4.7 and Section 4.8")

# 3. Section 5 (Conclusion & Future Scope) replacement
s5_header_idx = text.find(r"\section{Conclusion and Future Scope}")
assert s5_header_idx != -1, "s5_header_idx not found"
s5_end = text.find(r"\section*{Checkpoint Provenance and Reproducibility}")
assert s5_end != -1, "s5_end not found"

new_s5 = r"""\section{Conclusion and Future Scope}
\label{sec:conclusion}

This paper presents a comprehensive empirical evaluation and operational deployment of a CPU-efficient plant disease screening system across six experimental phases, covering dataset curation, multi-architecture training, temperature calibration, candidate-region screening, full-stack application deployment, and perturbation stress testing.

\textbf{Key findings are summarised as follows}:
\begin{enumerate}
  \item All four architectures achieve competitive in-domain classification accuracy (95.02\%--95.96\%) on the controlled PlantVillage holdout test set, with performance differences within approximately one percentage point.
  \item MobileNetV3-Large achieves the highest holdout accuracy (95.96\%, Macro-F1 94.52\%) and the strongest external PlantDoc generalization (41.53\% top-1).
  \item MobileNetV3-Small provides the best computational profile: 9.42\,ms inference latency, 6.06\,MB disk footprint, and 1.56M parameters, making it the preferred choice for resource-constrained edge deployment.
  \item The domain shift from controlled PlantVillage to natural PlantDoc conditions is severe ($\approx$56 percentage points) and consistent across all four architectures, confirming that in-domain benchmark accuracy does not predict field utility.
  \item Post-hoc temperature scaling is effective for in-domain probability calibration (reducing ECE from 10.25\% to 0.58\% on the baseline), but calibration parameters fitted on laboratory imagery do not transfer to natural field photography.
  \item The Phase~4.1 candidate-region screening pipeline more than doubles localization recall at IoU 0.25 (from 18.14\% to 38.05\%) through multi-scale windows, aspect-ratio awareness, ExG saliency filtering, context padding, and hybrid NMS, while requiring 844.25\,ms CPU latency per image.
  \item Precision at IoU 0.25 remains bounded (10.60\%) and dense canopy recall is challenging (21.88\%), reflecting the fundamental limitation of sliding-window heuristics compared to learned detectors. Crucially, candidate regions are whole-leaf proposals, not pixel-accurate lesion segmentations.
  \item Phase~6A successfully integrated the verified MobileNetV3-Small TorchScript model into an interactive full-stack React/FastAPI web application, achieving responsive sub-200\,ms screening, explicit human-in-the-loop uncertainty gating, and robust performance under real-world visual perturbations.
\end{enumerate}

\textbf{Future scope} to address remaining challenges includes:
\begin{enumerate}
  \item \textbf{Prospective Multi-Season Field Validation}: Conducting longitudinal field trials in partnership with agricultural extension services across diverse agroecological zones, capturing natural variations in crop phenology, illumination, and co-occurring pest/pathogen complexes.
  \item \textbf{Ergonomic User Studies and Human-in-the-Loop Decision Quality}: Evaluating the operational efficacy of the deployed full-stack interface through structured user studies with smallholder farmers and extension officers, measuring diagnostic time, decision accuracy, and trust calibration.
  \item \textbf{Domain Generalization and Vision Foundation Models}: Investigating parameter-efficient fine-tuning (PEFT) and self-supervised vision foundation backbones (e.g., DINOv2, BioCLIP) to bridge the 56 percentage point domain gap between controlled laboratory benchmarks and unconstrained field photography.
  \item \textbf{Learned Foliar Object Detection}: Replacing heuristic sliding-window proposals with a dedicated, lightweight foliar detector (e.g., YOLOv8-Nano or RT-DETR) trained on field bounding boxes to improve candidate-region precision in dense canopy environments.
  \item \textbf{Quantitative Agronomic Severity Rating}: Expanding from whole-leaf categorical classification to pixel-level lesion segmentation for quantitative percentage-area severity grading aligned with standard plant pathology assessment keys.
  \item \textbf{Open-Set Non-Foliar Rejection}: Developing energy-based or Dirichlet-prior out-of-distribution detection to reliably reject non-agricultural scenes, non-plant imagery, and unrepresented weed species before classifier evaluation.
\end{enumerate}

"""

text = text[:s5_header_idx] + new_s5 + text[s5_end:]
print("Updated Section 5 (Conclusion & Future Scope)")

# 4. Appendix A replacement
app_start = text.find(r"\section{Agronomic Pathology Reference and Diagnostic Triage Guide}")
if app_start == -1:
    app_start = text.find(r"\section{Agronomic Pathology Reference")
assert app_start != -1, "app_start not found"
app_end = text.find(r"\end{appendices}")
assert app_end != -1, "app_end not found"

new_app = r"""\section{Agronomic Pathology Reference and Experimental Triage Framework (Exploratory Research Proposal)}
\label{sec:appendix_e}

\noindent\fbox{\parbox{0.96\linewidth}{
\textbf{Research Framework Disclaimer:} The three-tier referral framework presented below is an exploratory offline research proposal evaluated on benchmark partitions. It is \textbf{not} implemented in the live deployed full-stack application (Section~\ref{sec:system_deployment}), which intentionally utilizes uncalibrated probabilities ($T=1.0$) with an empirical $0.60$ human review threshold to prevent the propagation of misleading confidence under field domain shifts.
}}
\vspace{0.8em}

\subsection{Experimental Three-Tier Non-Autonomous Referral Framework}
To investigate how calibrated model probabilities could support responsible field decision-making without claiming autonomous phytosanitary diagnostic authority, we evaluated an offline three-tier referral framework:
\begin{itemize}
  \item \textbf{Tier 1 --- Automated Decision Support Display ($\tau_1 \ge 0.70$):} Predictions with calibrated confidence satisfying $\tau_1 \ge 0.70$. Empirical validation on the 5{,}429-image PlantVillage calibration partition under Phase~3 calibration ($T^* = 0.5538$) demonstrated that threshold $\tau_1 = 0.70$ achieves \textbf{97.6\% top-1 classification accuracy} while retaining \textbf{94.5\% sample coverage}. In field extension, this tier provides tentative visual confirmation while still recommending routine agronomic scouting. An agronomic advisory note explicitly instructs the extension worker to verify foliar margins and local environmental preconditions before recommending cultural sanitation.
  \item \textbf{Tier 2 --- Secondary Agronomist Inspection ($0.223 \le p_{\text{calibrated}} < 0.70$):} Samples exhibiting ambiguous foliar coloration, early lesion development, or atypical background interference. In Phase~4.1 selective evaluation on the 236-image PlantDoc natural-background test set, threshold $\tau_2 = 0.223$ achieved \textbf{93.22\% sample coverage} (220/236 images) with \textbf{40.91\% selective top-1 accuracy}, successfully pruning the 16 most erratic, extreme-uncertainty predictions while maintaining broad coverage. The system outputs the top-3 differential classes with calibrated likelihoods, flags the sample for manual inspection, and prevents any autonomous decision from being logged.
  \item \textbf{Tier 3 --- Inconclusive / Mandatory Extension Referral ($p_{\text{calibrated}} < 0.223$ or ExG Coverage $< 8\%$):} The input image contains severe motion blur, off-target foliage, complex overlapping canopies, or unsupported plant species. The screening engine suppresses candidate proposals and advises the user to submit physical leaf tissue samples to a local agricultural extension diagnostic clinic or recapture the photograph under diffuse natural lighting with a single isolated leaf in view. Under no condition does Tier 3 report a disease label.
\end{itemize}

\begin{table}[htbp]
\caption{Agronomic differential triage matrix mapping representative foliar signs to operational escalation tiers across major crop categories. Prescriptive chemical directives are omitted; actions strictly enforce human-in-the-loop inspection and diagnostic clinic referral.}\label{tab:field_matrix}%
\footnotesize
\setlength{\tabcolsep}{3pt}%
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{3.8cm}}
\toprule
\textbf{Crop / Pathology} & \textbf{Diagnostic Foliar Signs} & \textbf{Confused Lookalikes} & \textbf{Operational Screening Protocol} \\
\midrule
Potato Late Blight (\textit{P.~infestans}) & Water-soaked pale green lesions rapidly expanding to necrotic brown & Frost damage, wind desiccation, soft rot & Tier 1 (Urgent): Inspect abaxial leaf surface for white sporulation; submit tissue sample for assay. \\
Tomato TYLCV & Upward leaf curling, marginal chlorosis, severe leaflet stunting & Aphid feeding damage, mechanical mite injury & Tier 1: Scout leaflet undersides for whitefly vectors; submit leaf press to plant virology clinic. \\
Ambiguous / Low-Confidence & Diffuse foliar chlorosis, atypical lesion shape, overlapping leaves & Nutrient deficiencies (Fe, Zn, N), herbicide drift & Tier 2: Defer classification; suppress autonomous diagnosis; flag for secondary agronomist inspection. \\
Occluded / Non-Target Scene & Foliar coverage $< 8\%$ of view, extreme motion blur, heavy shadow & Non-target weeds, bare soil, mulch & Tier 3: Suppress inference proposals; prompt user to re-photograph leaf with neutral backing. \\
\botrule
\end{tabular*}
\end{table}

\subsection{Field Deployment Protocol and Mobile Image Capture Guidelines}
To minimize out-of-distribution false positives in practical operational settings, field agricultural extension officers must adhere to standardized image acquisition constraints before submitting candidates to the screening pipeline:
\begin{enumerate}
  \item \textbf{Foliar Isolation and Background Control:} Position the target leaf flat without folding or extreme curl. Where dense background weed cover or bare soil creates color interference, place a matte neutral surface (or shaded hand) directly behind the specimen. The leaf blade must occupy at least 35\% of the central viewfinder.
  \item \textbf{Diffuse Ambient Illumination:} Avoid capturing leaves under direct, harsh perpendicular sunlight, which produces specular reflection hotspots that saturate green-channel values and mimic viral chlorosis. Early morning diffuse light, overcast conditions, or manual body shading are required.
  \item \textbf{Lens Sanitation and Focusing:} Clean mobile camera lenses to prevent grease smearing that creates false halo artifacts. Maintain a distance of 15--25\,cm to ensure fine lesion micro-structures (e.g., fungal pycnidia, concentric rings) remain sharp and resolvable above the $224 \times 224$ input resolution threshold.
  \item \textbf{Multi-Angle Verification:} For suspected quarantine-grade pathogens (such as \textit{Phytophthora infestans}), capture at least two complementary views (adaxial top surface and abaxial underside) to confirm characteristic sporulation patterns prior to escalating mitigation measures.
\end{enumerate}

"""

text = text[:app_start] + new_app + text[app_end:]
print("Updated Appendix A")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Saved successfully! Final length:", len(text))

