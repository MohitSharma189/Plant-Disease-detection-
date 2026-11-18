# Research Paper Completion Report
## Plant Disease Screening System with MobileNetV3-Small

**Date:** 2026-11-13  
**Status:** DRAFT COMPLETE — Awaiting Author Review and LaTeX Compilation  
**Output Directory:** `reports/research_paper/`

---

## 1. Sections Completed

| Section | Title | Status | Page Estimate |
|---|---|---|---|
| Title / Abstract / Keywords | Paper header + 230-word abstract | ✅ Complete | 1 |
| §1 | Introduction | ✅ Complete | ~1.5 |
| §1.1 | Research Gap | ✅ Complete | 0.5 |
| §1.2 | Research Objectives | ✅ Complete | 0.5 |
| §2 | Literature Review | ✅ Complete | ~2.5 |
| §2.1 | Deep Learning for Plant Disease | ✅ Complete | 0.5 |
| §2.2 | CNN Architectures for Lightweight Deployment | ✅ Complete | 0.5 |
| §2.3 | PlantVillage and PlantDoc Benchmarks | ✅ Complete | 0.5 |
| §2.4 | Domain Generalization | ✅ Complete | 0.5 |
| §2.5 | Model Calibration | ✅ Complete | 0.5 |
| §2.6 | Candidate-Region Detection | ✅ Complete | 0.5 |
| §3 | Methodology | ✅ Complete | ~4 |
| §3.1 | Dataset (tabular form) | ✅ Complete | 1 |
| §3.2 | Model Architectures (4 models with formulas) | ✅ Complete | 2 |
| §3.3 | System Pipeline / Flowchart | ✅ Complete | 0.5 |
| §3.4 | Temperature Scaling | ✅ Complete | 0.5 |
| §3.5 | Screening Pipeline (with pseudocode-equivalent) | ✅ Complete | 1 |
| §3.6 | Evaluation Protocol | ✅ Complete | 0.5 |
| §4 | Results and Discussion | ✅ Complete | ~5 |
| §4.1 | PlantVillage Classification Results (Table) | ✅ Complete | 1 |
| §4.2 | PlantDoc External Evaluation (Table) | ✅ Complete | 1 |
| §4.3 | Calibration Results (Table + discrepancy note) | ✅ Complete | 1 |
| §4.4 | Error Analysis + Confusion Matrices | ✅ Complete | 1 |
| §4.5 | Comparative Architecture Analysis (Table + Figure) | ✅ Complete | 1 |
| §4.6 | Screening Pipeline Evaluation (Table) | ✅ Complete | 1 |
| §4.7 | Efficiency Summary + Reliability Diagram | ✅ Complete | 0.5 |
| §5 | Conclusion and Future Scope | ✅ Complete | ~1 |
| Checkpoint Provenance | Reproducibility section with SHA-256 table | ✅ Complete | 0.5 |
| Limitations | Ethical statement | ✅ Complete | 0.5 |
| References | 38 APA-formatted entries in references.bib | ✅ Complete | ~2 |

**Estimated total page count: 18–22 pages** (Springer single-column layout).  
Actual page count requires LaTeX compilation.

---

## 2. All Five Teacher Objections Addressed

| # | Objection | Resolution Location |
|---|---|---|
| 1 | Reproducible training + checkpoint provenance | §3.1, §3.2, §3.3, Checkpoint Provenance table |
| 2 | Independent external PlantDoc evaluation | §4.2 (Table 3), §4.4 |
| 3 | Offline calibration vs. deployed app discrepancy | §4.3 (transparent documented discrepancy), Limitations |
| 4 | Candidate-region screening + whole-plant processing | §3.5, §4.6 (Table 5) |
| 5 | Multi-architecture comparison + stronger evidence | §4.1, §4.5 (Tables 1–4), Figure 3 |

---

## 3. Verified Experimental Sources Used

| Data Source | Used In | Verified Via |
|---|---|---|
| Phase 1 — Dataset integrity, dedup audit | §3.1 (Table 1) | `reports/phase1/phase1_completion_report.md` |
| Phase 2 — MobileNetV3-Small training history | §3.2, §3.3 | `reports/phase2/final/PHASE2_COMPLETION_REPORT.md` |
| Phase 3 — PlantVillage holdout, PlantDoc eval, T=0.5538 calibration | §4.1, §4.2, §4.3, §4.7 | `reports/phase3/PHASE3_COMPLETION_REPORT.md` |
| Phase 4 — Baseline screening pipeline | §4.6 (Table 5) | `reports/phase4/PHASE4_COMPLETION_REPORT.md` |
| Phase 4.1 — Optimized screening pipeline | §3.5, §4.6 | `reports/phase4_1/PHASE4_1_COMPLETION_REPORT.md` |
| Phase 5 — All 4 architectures, calibration, latency | §4.1–4.5, Tables 1–4 | `reports/phase5/PHASE5_COMPLETION_REPORT.md` |

---

## 4. References Included

**Total bibliography entries:** 38 genuine peer-reviewed citations in `references.bib`

| Category | Count | Key Works |
|---|---|---|
| Plant disease recognition | 8 | Mohanty et al. (2016), Hughes & Salathé (2015), Ferentinos (2018) |
| MobileNet architectures | 3 | Howard et al. (2017, 2019) |
| EfficientNet | 2 | Tan & Le (2019), Atila et al. (2021) |
| ResNet | 2 | He et al. (2016), Jiang et al. (2019) |
| PlantDoc / domain shift | 4 | Singh et al. (2020), D'Souza et al. (2023), Toda & Okura (2019) |
| Calibration | 3 | Guo et al. (2017), Minderer et al. (2021), Niculescu-Mizil & Caruana (2005) |
| Object detection | 3 | Ren et al. (2015), Redmon et al. (2016), Neubeck & Van Gool (2006) |
| Transfer learning | 3 | Deng et al. (2009), Chen et al. (2020), Too et al. (2019) |
| Agricultural context | 2 | Savary et al. (2019), Pantazi et al. (2019) |
| Explainability | 2 | Selvaraju et al. (2017), Zhang et al. (2019) |
| ExG vegetation index | 1 | Woebbecke et al. (1995) |
| Data augmentation | 2 | Shorten & Khoshgoftaar (2019), Kwak et al. (2023) |
| PyTorch | 1 | Paszke et al. (2019) |
| Survey | 1 | Kamilaris & Prenafeta-Boldú (2018) |
| Squeeze-and-Excitation | 1 | Hu et al. (2018) |

---

## 5. Figures and Tables Included

### Figures
| ID | Filename | Source | Description |
|---|---|---|---|
| Fig. 1 | *(pipeline flowchart)* | Placeholder text | System pipeline overview |
| Fig. 2 | `figures/fig_confmat_mvs.png` | Phase 5 | MobileNetV3-Small confusion matrix |
| Fig. 2b | `figures/fig_confmat_mvl.png` | Phase 5 | MobileNetV3-Large confusion matrix |
| Fig. 3 | `figures/fig_arch_comparison.png` | Phase 5 | 6-panel architecture comparison chart |
| Fig. 4 | `figures/fig_calibration.png` | Phase 3 | Reliability diagrams pre/post calibration |

### Tables
| ID | Caption | Source |
|---|---|---|
| Table 1 | Dataset partitions | Phase 1–3 reports |
| Table 2 | Training hyperparameters | Phase 2–5 config |
| Table 3 | PlantVillage holdout results | Phase 5 CSVs |
| Table 3b | Validation vs holdout accuracy | Phase 5 CSVs |
| Table 4 | PlantDoc external results | Phase 3 + Phase 5 CSVs |
| Table 5 | Temperature calibration results | Phase 3 + Phase 5 calibration CSV |
| Table 6 | Full architecture comparison | Phase 5 architecture_comparison.csv |
| Table 7 | Screening evaluation Phase 4 vs 4.1 | Phase 4 + Phase 4.1 reports |
| Table 8 | Checkpoint provenance | Phase 2 + Phase 5 provenance JSON |

---

## 6. LaTeX Compilation Instructions

### For Overleaf (Recommended)
1. Upload the entire `reports/research_paper/` folder to Overleaf.
2. Set the main file to `main.tex`.
3. Overleaf automatically downloads the Springer Nature template.
4. Compile with **pdfLaTeX**.

### For Local Compilation (requires Springer template files)
1. Download `sn-jnl.cls`, `sn-apacite.bst`, and supporting files from:
   https://www.springernature.com/gp/authors/campaigns/latex-author-support
2. Place these files in the same directory as `main.tex`.
3. Run: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

### Required Springer Template Files (not bundled — download separately)
- `sn-jnl.cls` — Springer Nature journal document class
- `sn-apacite.bst` — APA citation style for Springer

> **Note**: These files are proprietary to Springer Nature and cannot be redistributed.
> Download from the official Springer Nature LaTeX support page.

---

## 7. Remaining Limitations and Items Requiring Author Confirmation

| Item | Status | Required Action |
|---|---|---|
| Author name, affiliation, email | Placeholder values used | **Replace with actual author information** |
| Corresponding author email | `mohit@example.com` (placeholder) | **Update to actual institutional email** |
| Springer template class files | Not included (proprietary) | **Download from Springer Nature LaTeX support** |
| System pipeline flowchart (Fig. 1) | Text placeholder (actual diagram not generated) | **Create TikZ flowchart or add actual figure** |
| Phase 3 calibration discrepancy (T=0.5538 vs T=0.9902) | Transparently documented in §4.3 | **Author should verify the Phase 3 calibration script parameters match the report** |
| PlantDoc per-class metrics | Referenced as supplementary CSV | **Attach `plantdoc_classification_report.csv` as supplementary material** |
| References indexing status | Noted as "not verified" in instructions | **Verify Scopus/SCI/SCIE indexing for each cited paper before submission** |

---

## 8. Research Integrity Compliance

- ✅ No experimental results fabricated or invented.
- ✅ No DOI numbers invented.
- ✅ PlantDoc results clearly distinguished from PlantVillage results.
- ✅ Domain shift transparently documented (56 pp accuracy drop).
- ✅ Calibration discrepancy between Phase 3 and Phase 5 documented rather than concealed.
- ✅ No claim of lesion segmentation from whole-leaf bounding boxes.
- ✅ No claim of clinical deployment validation.
- ✅ ShuffleNetV2 acknowledged as intentionally skipped (no fabricated results).
- ✅ No repeated-seed confidence intervals claimed without evidence.
- ✅ Original paper and all Phase artifacts preserved without modification.

---

*End of Research Paper Completion Report.*

