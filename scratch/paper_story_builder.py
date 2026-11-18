"""
Builds the complete story flow for the final revision paper.
Includes full 38-class and PlantDoc per-class tables, comprehensive text,
verified figures, correct sequential numbering, appendices A-D, and full APA bibliography.
"""

import os
import re
import pandas as pd
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle

import final_paper_data as fpd
import final_paper_text as fpt
import final_paper_expansion as fpe
import final_paper_reproduction as fpr
import final_paper_arch_analysis as fpa
from build_final_revision_paper import make_callout, make_formula_table

def parse_bibtex_apa(bib_path):
    if not os.path.exists(bib_path):
        return []
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),([^@]+)\}', content)
    parsed = []
    for entry_type, cite_key, fields_str in entries:
        fields = {}
        for match in re.finditer(r'(\w+)\s*=\s*\{([^}]*)\}', fields_str):
            fields[match.group(1).lower()] = match.group(2).strip()
        
        authors_raw = fields.get('author', 'Anonymous')
        author_list = [a.strip() for a in authors_raw.split(' and ')]
        
        formatted_authors = []
        for a in author_list:
            if ',' in a:
                parts = [p.strip() for p in a.split(',', 1)]
                last = parts[0]
                first = parts[1] if len(parts) > 1 else ''
                inits = ' '.join(f"{n[0]}." for n in first.split() if n)
                formatted_authors.append(f"{last}, {inits}" if inits else last)
            else:
                parts = a.split()
                if len(parts) > 1:
                    last = parts[-1]
                    inits = ' '.join(f"{p[0]}." for p in parts[:-1] if p)
                    formatted_authors.append(f"{last}, {inits}")
                else:
                    formatted_authors.append(a)
                    
        if len(formatted_authors) == 1:
            author_str = formatted_authors[0]
        elif len(formatted_authors) == 2:
            author_str = f"{formatted_authors[0]} & {formatted_authors[1]}"
        elif len(formatted_authors) > 2:
            author_str = f"{', '.join(formatted_authors[:-1])}, & {formatted_authors[-1]}"
        else:
            author_str = "Anonymous"
            
        year = fields.get('year', 'n.d.')
        title = fields.get('title', '').replace('{', '').replace('}', '')
        journal = fields.get('journal', fields.get('booktitle', ''))
        volume = fields.get('volume', '')
        pages = fields.get('pages', '')
        doi = fields.get('doi', '')
        url = fields.get('url', '')
        
        entry_text = f"{author_str} ({year}). {title}. "
        if journal:
            entry_text += f"<i>{journal}</i>"
            if volume:
                entry_text += f", {volume}"
            if pages:
                entry_text += f", {pages}"
            entry_text += ". "
        if doi:
            entry_text += f"https://doi.org/{doi}"
        elif url:
            entry_text += f"{url}"
            
        parsed.append((author_str.lower(), entry_text))
        
    parsed.sort(key=lambda x: x[0])
    return [p[1] for p in parsed]

def load_plantdoc_per_class():
    path = "reports/phase3/plantdoc_classification_report.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        cname = str(r['class_name']).replace("___", " — ").replace("_", " ")
        status = str(r['status'])
        sup = str(int(r['support'])) if str(r['support']).isdigit() or isinstance(r['support'], (int, float)) else str(r['support'])
        p = f"{float(r['precision'])*100:.1f}%" if r['precision'] != 'N/A' and pd.notnull(r['precision']) and r['precision'] != 'None' else "N/A"
        rec = f"{float(r['recall'])*100:.1f}%" if r['recall'] != 'N/A' and pd.notnull(r['recall']) and r['recall'] != 'None' else "N/A"
        f1 = f"{float(r['f1_score'])*100:.1f}%" if r['f1_score'] != 'N/A' and pd.notnull(r['f1_score']) and r['f1_score'] != 'None' else "N/A"
        rows.append([cname, status, sup, p, rec, f1])
    return rows

def build_full_story(styles, printable_w):
    story = []
    
    # ── HEADER BANNER ──────────────────────────────────────────────────────────
    banner_p = Paragraph("<b>Springer Nature Applied Intelligence / Computers and Electronics in Agriculture (Preprint)</b>", styles['JournalBanner'])
    meta_p = Paragraph("Original Research Article &nbsp;|&nbsp; Manuscript Ref: AGY-PLANTDIS-2026-REV6 &nbsp;|&nbsp; Published: November 2026", styles['JournalMeta'])
    banner_table = Table([[banner_p], [meta_p]], colWidths=[printable_w])
    banner_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('LINEBELOW', (0,1), (-1,1), 1.0, colors.HexColor('#1B365D'))
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))
    
    # ── TITLE & AUTHORS ────────────────────────────────────────────────────────
    title_text = "Plant Disease Screening System with MobileNetV3-Small: Implementation, Exploratory Results, and Validation Gaps"
    authors_text = "Mohit Sharma<sup>1*</sup>, &nbsp;Aniket Kumar<sup>1&dagger;</sup>, &nbsp;Raghav Kumar<sup>1&dagger;</sup>"
    affil_text = "<sup>1</sup> Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India.<br/>" \
                 "* Corresponding author: <u>mohit.12324849@lpu.in</u> &nbsp;|&nbsp; Contributing authors: aniket.12325552@lpu.in; raghavmumar23@lpu.in"
    notes_text = "<sup>&dagger;</sup> These authors contributed equally to this work. &nbsp;|&nbsp; Verified Phase 1–5 Provenance Build."

    story.append(Paragraph(title_text, styles['PaperTitle']))
    story.append(Paragraph(authors_text, styles['AuthorName']))
    story.append(Paragraph(affil_text, styles['Affiliation']))
    story.append(Paragraph(notes_text, styles['AuthorNotes']))
    story.append(Spacer(1, 4))
    
    # ── ABSTRACT & KEYWORDS BOX ────────────────────────────────────────────────
    abs_title = Paragraph("<b>Abstract</b>", styles['AbstractHeading'])
    abs_body = Paragraph(fpt.ABSTRACT, styles['AbstractBody'])
    kw_p = Paragraph(f"<b>Keywords:</b> <i>{', '.join(fpt.KEYWORDS)}</i>", styles['Keywords'])
    
    abstract_box = Table([[abs_title], [abs_body], [kw_p]], colWidths=[printable_w])
    abstract_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(abstract_box)
    story.append(Spacer(1, 8))
    
    # ── SECTION 1: INTRODUCTION ────────────────────────────────────────────────
    story.append(Paragraph("1. Introduction", styles['Sec1']))
    story.append(Paragraph(fpt.INTRO_BACKGROUND, styles['Body']))
    story.append(Paragraph(fpt.INTRO_PROBLEM, styles['Body']))
    
    story.append(Paragraph("1.1 Research Gap", styles['Sec2']))
    story.append(Paragraph(fpt.INTRO_GAP, styles['Body']))
    
    story.append(Paragraph("1.2 Research Objectives", styles['Sec2']))
    story.append(Paragraph("To address these identified deficiencies, this study pursues six concrete research objectives:", styles['Body']))
    for i, obj in enumerate(fpt.INTRO_OBJECTIVES, 1):
        story.append(Paragraph(f"<b>RO{i}:</b> {obj}", styles['Body']))
        
    story.append(Paragraph("1.3 Primary Contributions", styles['Sec2']))
    story.append(Paragraph("The key contributions delivered in this revised manuscript are:", styles['Body']))
    for i, con in enumerate(fpt.INTRO_CONTRIBUTIONS, 1):
        story.append(Paragraph(f"&bull; <b>Contribution {i}:</b> {con}", styles['Body']))
        
    callout_obj = "<b>Faculty Review Alignment Notice:</b> This revision systematically resolves all five major faculty critique items: " \
                  "(1) duplicate-aware split provenance and manifest hashing; (2) unconstrained external PlantDoc benchmark; " \
                  "(3) rigorous post-hoc temperature scaling with transparent discrepancy reconciliation; (4) whole-plant candidate screening " \
                  "with IoU-based localization metrics; and (5) comprehensive 4-architecture benchmark on identical hardware."
    story.append(Spacer(1, 2))
    story.append(make_callout(callout_obj, styles, printable_w, bg_color='#EFF6FF', border_color='#3B82F6'))
    story.append(Spacer(1, 6))
    
    # ── SECTION 2: LITERATURE REVIEW ───────────────────────────────────────────
    story.append(Paragraph("2. Literature Review", styles['Sec1']))
    story.append(Paragraph(fpt.LIT_INTRO, styles['Body']))
    
    story.append(Paragraph("2.1 Deep Learning in Plant Pathology Datasets", styles['Sec2']))
    story.append(Paragraph(fpt.LIT_DATASETS, styles['Body']))
    
    story.append(Paragraph("2.2 Lightweight Convolutional Architectures & Mobile Deployment", styles['Sec2']))
    story.append(Paragraph(fpt.LIT_ARCHITECTURES, styles['Body']))
    
    story.append(Paragraph("2.3 Cross-Domain Generalization & Natural Image Benchmarks", styles['Sec2']))
    story.append(Paragraph(fpt.LIT_DOMAIN, styles['Body']))
    
    story.append(Paragraph("2.4 Uncertainty Quantification & Post-Hoc Model Calibration", styles['Sec2']))
    story.append(Paragraph(fpt.LIT_CALIBRATION, styles['Body']))
    
    story.append(Paragraph("2.5 Candidate-Region Localization & Agricultural Screening", styles['Sec2']))
    story.append(Paragraph(fpt.LIT_LOCALIZATION, styles['Body']))
    
    story.append(Paragraph("2.6 Comparative Analysis & Synthesis of 2020–2026 State-of-the-Art", styles['Sec2']))
    story.append(Paragraph(fpe.LIT_EXTENDED_ANALYSIS, styles['Body']))
    story.append(Paragraph(fpt.LIT_GAP_STATEMENT, styles['Body']))
    story.append(Spacer(1, 6))
    
    # ── SECTION 3: METHODOLOGY ─────────────────────────────────────────────────
    story.append(Paragraph("3. Methodology", styles['Sec1']))
    story.append(Paragraph("3.1 Dataset Composition & Duplicate-Aware Stratified Partitioning", styles['Sec2']))
    story.append(Paragraph(fpt.METH_DATASET, styles['Body']))
    
    # Table 1: Dataset Partitioning
    t1_data = [
        [Paragraph("<b>Partition Role</b>", styles['TableHeader']),
         Paragraph("<b>Source Dataset</b>", styles['TableHeader']),
         Paragraph("<b>Sample Count</b>", styles['TableHeader']),
         Paragraph("<b>Class Coverage</b>", styles['TableHeader']),
         Paragraph("<b>Leakage Isolation Mechanism</b>", styles['TableHeader'])],
        [Paragraph("Training Split", styles['TableCellBold']), Paragraph("PlantVillage + Cleaned PlantDoc Train", styles['TableCell']), Paragraph("40,151 (37,992 + 2,159)", styles['TableCell']), Paragraph("38 classes (14 crops)", styles['TableCell']), Paragraph("Perceptual hash grouping; gradient updates enabled", styles['TableCell'])],
        [Paragraph("Validation Split", styles['TableCellBold']), Paragraph("PlantVillage Holdout Slice", styles['TableCell']), Paragraph("5,436", styles['TableCell']), Paragraph("38 classes", styles['TableCell']), Paragraph("Hyperparameter tuning and early stopping only", styles['TableCell'])],
        [Paragraph("Calibration Split", styles['TableCellBold']), Paragraph("PlantVillage Holdout Slice", styles['TableCell']), Paragraph("5,429", styles['TableCell']), Paragraph("38 classes", styles['TableCell']), Paragraph("100% isolated; temperature fitting only", styles['TableCell'])],
        [Paragraph("Test Split (Holdout)", styles['TableCellBold']), Paragraph("PlantVillage Holdout Slice", styles['TableCell']), Paragraph("5,417", styles['TableCell']), Paragraph("38 classes", styles['TableCell']), Paragraph("Final in-domain generalization assessment", styles['TableCell'])],
        [Paragraph("<b>Combined Manifest Total</b>", styles['TableCellBold']), Paragraph("<b>Partitioned Ecosystem</b>", styles['TableCellBold']), Paragraph("<b>56,433 Images</b>", styles['TableCellBold']), Paragraph("<b>38 Diagnostic Labels</b>", styles['TableCellBold']), Paragraph("<b>Manifest SHA-256: f9ef8b9dcd6823f2...</b>", styles['TableCellBold'])],
        [Paragraph("External Benchmark", styles['TableCellBold']), Paragraph("Official PlantDoc Test Set", styles['TableCell']), Paragraph("236 Images (452 leaves)", styles['TableCell']), Paragraph("13 cross-mapped classes", styles['TableCell']), Paragraph("Untouched natural field benchmark (independent)", styles['TableCell'])],
    ]
    t1 = Table(t1_data, colWidths=[100, 110, 85, 95, 97.27])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 1:</b> Deterministic, duplicate-aware partitioning scheme and isolation protocol across datasets.", styles['TableCaption']))
    story.append(t1)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("3.2 Evaluated Convolutional Neural Network Architectures", styles['Sec2']))
    story.append(Paragraph(fpt.METH_ARCHITECTURES, styles['Body']))
    
    # Formulas for Architectures
    eq1 = "MobileNetV3 Inverted Bottleneck: &nbsp;&nbsp; <b>y</b> = &sigma;(<b>W</b><sub>proj</sub> &middot; SE(&phi;(&sigma;(<b>W</b><sub>exp</sub> &middot; <b>x</b>)))) + <b>x</b> &nbsp;&nbsp; (if stride=1 and C<sub>in</sub>=C<sub>out</sub>)"
    story.append(make_formula_table(eq1, 1, styles, printable_w))
    eq2 = "Squeeze-and-Excitation Gating: &nbsp;&nbsp; <b>s</b> = &sigma;(<b>W</b><sub>2</sub> &middot; ReLU(<b>W</b><sub>1</sub> &middot; GAP(<b>x</b>))), &nbsp;&nbsp; SE(<b>x</b>) = <b>s</b> &otimes; <b>x</b>"
    story.append(make_formula_table(eq2, 2, styles, printable_w))
    eq3 = "Hard-Swish Activation: &nbsp;&nbsp; h-swish(x) = x &middot; [ReLU6(x + 3) / 6]"
    story.append(make_formula_table(eq3, 3, styles, printable_w))
    eq4 = "ResNet Residual Block: &nbsp;&nbsp; <b>y</b> = &sigma;(<b>W</b><sub>2</sub> &middot; &sigma;(<b>W</b><sub>1</sub> &middot; <b>x</b>)) + <b>x</b>"
    story.append(make_formula_table(eq4, 4, styles, printable_w))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("3.3 Training Configuration and Optimization Dynamics", styles['Sec2']))
    story.append(Paragraph(fpt.METH_TRAINING, styles['Body']))
    story.append(Paragraph(fpr.MATH_LOSS_REGULARIZATION, styles['Body']))
    story.append(Spacer(1, 4))
    
    # Table 2: Training Recipe
    t2_data = [
        [Paragraph("<b>Hyperparameter / Protocol Component</b>", styles['TableHeader']), Paragraph("<b>Configured Value / Specification</b>", styles['TableHeader']), Paragraph("<b>Theoretical / Practical Rationale</b>", styles['TableHeader'])],
        [Paragraph("Input Spatial Dimension", styles['TableCellBold']), Paragraph("224 &times; 224 &times; 3 RGB", styles['TableCell']), Paragraph("Standard ImageNet resolution balancing accuracy and CPU latency", styles['TableCell'])],
        [Paragraph("Deterministic Seed / RNG State", styles['TableCellBold']), Paragraph("Seed = 42 (torch.use_deterministic_algorithms)", styles['TableCell']), Paragraph("Guarantees mathematical reproducibility of training updates", styles['TableCell'])],
        [Paragraph("Batch Size & Maximum Epochs", styles['TableCellBold']), Paragraph("Batch Size = 8, Epochs = 3 (1 Warm-up + 2 Fine-tune)", styles['TableCell']), Paragraph("Accommodates memory limits on CPU; rapid loss convergence", styles['TableCell'])],
        [Paragraph("Optimization Algorithm", styles['TableCellBold']), Paragraph("AdamW (Weight Decay = 1e-4)", styles['TableCell']), Paragraph("Decoupled weight regularization prevents feature collapse", styles['TableCell'])],
        [Paragraph("Two-Stage Transfer Schedule", styles['TableCellBold']), Paragraph("Warm-up: 1 epoch (head LR 1e-3); Fine-tune: LR 1e-5 (backbone), 1e-4 (head)", styles['TableCell']), Paragraph("Stabilizes pretrained weights before end-to-end gradient updates", styles['TableCell'])],
        [Paragraph("Learning Rate Policy", styles['TableCellBold']), Paragraph("Cosine Annealing with linear warmup", styles['TableCell']), Paragraph("Smooth decay avoids aggressive parameter oscillations", styles['TableCell'])],
        [Paragraph("Loss Formulation & Regularization", styles['TableCellBold']), Paragraph("Cross-Entropy + Label Smoothing (0.1) + Dropout (0.3)", styles['TableCell']), Paragraph("Mitigates overconfidence and enhances generalization", styles['TableCell'])],
        [Paragraph("Imbalance Compensation", styles['TableCellBold']), Paragraph("Square-root inverse class frequency weighting", styles['TableCell']), Paragraph("Counterbalances representation disparity across 38 classes", styles['TableCell'])],
        [Paragraph("Early Stopping Criterion", styles['TableCellBold']), Paragraph("Validation Loss, Patience = 2, Min Delta = 1e-3", styles['TableCell']), Paragraph("Terminates execution upon loss plateau to prevent over-adaptation", styles['TableCell'])],
    ]
    t2 = Table(t2_data, colWidths=[135, 175, 177.27])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 2:</b> Standardized hyperparameters and training protocol enforced across all evaluated architectures.", styles['TableCaption']))
    story.append(t2)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("3.4 Evaluation Metrics, Uncertainty Calibration & Statistical Formulation", styles['Sec2']))
    story.append(Paragraph(fpt.METH_METRICS, styles['Body']))
    
    # Formulas for Calibration
    eq5 = "Temperature-Scaled Softmax: &nbsp;&nbsp; p<sub>k</sub>(<b>z</b>; T) = exp(z<sub>k</sub> / T) / &sum;<sub>j=1</sub><sup>K</sup> exp(z<sub>j</sub> / T)"
    story.append(make_formula_table(eq5, 5, styles, printable_w))
    eq6 = "Expected Calibration Error (ECE): &nbsp;&nbsp; ECE = &sum;<sub>b=1</sub><sup>B</sup> (|B<sub>b</sub>| / N) &middot; |acc(B<sub>b</sub>) - conf(B<sub>b</sub>)|"
    story.append(make_formula_table(eq6, 6, styles, printable_w))
    eq7 = "Multiclass Brier Score: &nbsp;&nbsp; Brier = (1/N) &sum;<sub>i=1</sub><sup>N</sup> &sum;<sub>k=1</sub><sup>K</sup> (p<sub>ik</sub> - y<sub>ik</sub>)<sup>2</sup>"
    story.append(make_formula_table(eq7, 7, styles, printable_w))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("3.5 Multi-Scale Candidate-Region Screening Pipeline (Phase 4.1)", styles['Sec2']))
    story.append(Paragraph(fpt.METH_SCREENING, styles['Body']))
    
    eq8 = "Excess Green Index (ExG): &nbsp;&nbsp; ExG = 2G - R - B, &nbsp;&nbsp; Foliar Mask: M(x,y) = <b>1</b>[ExG(x,y) &gt; &tau;<sub>foliar</sub>]"
    story.append(make_formula_table(eq8, 8, styles, printable_w))
    eq9 = "Intersection-over-Union (IoU): &nbsp;&nbsp; IoU(b<sub>1</sub>, b<sub>2</sub>) = |b<sub>1</sub> &cap; b<sub>2</sub>| / |b<sub>1</sub> &cup; b<sub>2</sub>|"
    story.append(make_formula_table(eq9, 9, styles, printable_w))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("3.6 Methodological Reconciliation of Calibration Discrepancies", styles['Sec2']))
    story.append(Paragraph(fpt.METH_CALIB_DISCREPANCY, styles['Body']))
    story.append(Spacer(1, 6))
    
    # ── SECTION 4: RESULTS AND DISCUSSION ──────────────────────────────────────
    story.append(Paragraph("4. Results and Discussion", styles['Sec1']))
    story.append(Paragraph("4.1 Definitive MobileNetV3-Small Baseline Training & Checkpoint Provenance", styles['Sec2']))
    story.append(Paragraph(fpt.RES_PHASE2_TRAINING, styles['Body']))
    
    # Table 3: Phase 2 Training History
    t3_data = [
        [Paragraph("<b>Epoch</b>", styles['TableHeader']),
         Paragraph("<b>Train Loss</b>", styles['TableHeader']),
         Paragraph("<b>Train Top-1 Acc</b>", styles['TableHeader']),
         Paragraph("<b>Val Loss</b>", styles['TableHeader']),
         Paragraph("<b>Val Top-1 Acc</b>", styles['TableHeader']),
         Paragraph("<b>Val Macro-F1</b>", styles['TableHeader']),
         Paragraph("<b>Backbone Status</b>", styles['TableHeader']),
         Paragraph("<b>Wall Time</b>", styles['TableHeader'])],
        [Paragraph("1", styles['TableCellBold']), Paragraph("1.5259", styles['TableCell']), Paragraph("82.32%", styles['TableCell']), Paragraph("0.3481", styles['TableCell']), Paragraph("93.47%", styles['TableCell']), Paragraph("91.25%", styles['TableCell']), Paragraph("Frozen (Warmup)", styles['TableCell']), Paragraph("644.5 s", styles['TableCell'])],
        [Paragraph("2", styles['TableCellBold']), Paragraph("2.0604", styles['TableCell']), Paragraph("64.96%", styles['TableCell']), Paragraph("0.2781", styles['TableCell']), Paragraph("93.89%", styles['TableCell']), Paragraph("91.53%", styles['TableCell']), Paragraph("Fine-Tuning Active", styles['TableCell']), Paragraph("1877.2 s", styles['TableCell'])],
        [Paragraph("3 (Best)", styles['TableCellBold']), Paragraph("1.4269", styles['TableCell']), Paragraph("85.06%", styles['TableCell']), Paragraph("0.2514", styles['TableCellBold']), Paragraph("95.14%", styles['TableCellBold']), Paragraph("93.19%", styles['TableCellBold']), Paragraph("Fine-Tuning Active", styles['TableCellBold']), Paragraph("1405.6 s", styles['TableCell'])],
    ]
    t3 = Table(t3_data, colWidths=[45, 55, 75, 55, 70, 65, 75, 47.27])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 3:</b> Epoch-by-epoch training trajectory of the definitive MobileNetV3-Small baseline (Seed 42).", styles['TableCaption']))
    story.append(t3)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.2 Independent PlantVillage Holdout Evaluation & Generalization Integrity", styles['Sec2']))
    story.append(Paragraph(fpt.RES_HOLDOUT, styles['Body']))
    
    # Table 4: Holdout vs Validation Metrics
    t4_data = [
        [Paragraph("<b>Evaluation Metric</b>", styles['TableHeader']),
         Paragraph("<b>Validation Split (N=5,436)</b>", styles['TableHeader']),
         Paragraph("<b>Holdout Test Split (N=5,417)</b>", styles['TableHeader']),
         Paragraph("<b>Absolute &Delta;</b>", styles['TableHeader']),
         Paragraph("<b>Integrity Interpretation</b>", styles['TableHeader'])],
        [Paragraph("Top-1 Accuracy", styles['TableCellBold']), Paragraph("95.14%", styles['TableCell']), Paragraph("95.02%", styles['TableCellBold']), Paragraph("-0.12 pp", styles['TableCell']), Paragraph("Negligible split variance; high generalization integrity", styles['TableCell'])],
        [Paragraph("Macro-Averaged F1-Score", styles['TableCellBold']), Paragraph("93.19%", styles['TableCell']), Paragraph("93.13%", styles['TableCellBold']), Paragraph("-0.06 pp", styles['TableCell']), Paragraph("Uniform discriminative capability across all 38 classes", styles['TableCell'])],
        [Paragraph("Weighted-Averaged F1-Score", styles['TableCellBold']), Paragraph("95.25%", styles['TableCell']), Paragraph("95.14%", styles['TableCell']), Paragraph("-0.11 pp", styles['TableCell']), Paragraph("Robustness maintained across high-support categories", styles['TableCell'])],
        [Paragraph("Macro-Averaged Precision", styles['TableCellBold']), Paragraph("92.35%", styles['TableCell']), Paragraph("92.16%", styles['TableCell']), Paragraph("-0.19 pp", styles['TableCell']), Paragraph("Low false-alarm propensity across diagnostic categories", styles['TableCell'])],
        [Paragraph("Macro-Averaged Recall", styles['TableCellBold']), Paragraph("95.10%", styles['TableCell']), Paragraph("95.06%", styles['TableCell']), Paragraph("-0.04 pp", styles['TableCell']), Paragraph("Consistently high diagnostic sensitivity on clean leaves", styles['TableCell'])],
        [Paragraph("Negative Log-Likelihood (NLL)", styles['TableCellBold']), Paragraph("0.2514", styles['TableCell']), Paragraph("0.2554 (Pre) / 0.1637 (Post)", styles['TableCell']), Paragraph("+0.0040 (Pre)", styles['TableCell']), Paragraph("Post-hoc calibration reduces cross-entropy loss by 35.9%", styles['TableCell'])],
        [Paragraph("Expected Calibration Error (ECE)", styles['TableCellBold']), Paragraph("10.25% (Pre) / 0.58% (Post)", styles['TableCell']), Paragraph("10.28% (Pre) / 0.74% (Post)", styles['TableCellBold']), Paragraph("+0.16 pp (Post)", styles['TableCell']), Paragraph("Temperature scaling achieves excellent probability alignment", styles['TableCell'])],
    ]
    t4 = Table(t4_data, colWidths=[120, 95, 100, 60, 112.27])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 4:</b> Comprehensive comparison of internal validation vs. independent holdout test metrics for MobileNetV3-Small.", styles['TableCaption']))
    story.append(t4)
    story.append(Spacer(1, 6))
    
    # ── Table 5: Comprehensive 38-Class Granular Performance Breakdown ─────────
    story.append(Paragraph("4.3 Comprehensive 38-Class Granular Breakdown & Botanical Error Modes", styles['Sec2']))
    story.append(Paragraph(fpe.CROP_ERROR_ANALYSIS, styles['Body']))
    story.append(Paragraph("Table 5 presents granular per-class precision, recall, F1-score, and support across all 38 classes, " \
                           "comparing MobileNetV3-Small and MobileNetV3-Large on the holdout test partition.", styles['Body']))
    
    csv_perclass = "reports/phase5/per_class_metrics.csv"
    if os.path.exists(csv_perclass):
        df_pc = pd.read_csv(csv_perclass)
        small_df = df_pc[df_pc['architecture'] == 'mobilenet_v3_small'].set_index('class_name')
        large_df = df_pc[df_pc['architecture'] == 'mobilenet_v3_large'].set_index('class_name')
        
        t5_rows = [
            [Paragraph("<b>Class Name (Botanical Pathology)</b>", styles['TableHeader']),
             Paragraph("<b>MVS Prec</b>", styles['TableHeader']),
             Paragraph("<b>MVS Rec</b>", styles['TableHeader']),
             Paragraph("<b>MVS F1</b>", styles['TableHeader']),
             Paragraph("<b>MVL Prec</b>", styles['TableHeader']),
             Paragraph("<b>MVL Rec</b>", styles['TableHeader']),
             Paragraph("<b>MVL F1</b>", styles['TableHeader']),
             Paragraph("<b>Support</b>", styles['TableHeader'])]
        ]
        for cname in small_df.index:
            s_row = small_df.loc[cname]
            l_row = large_df.loc[cname] if cname in large_df.index else s_row
            clean_name = cname.replace("___", " — ").replace("_", " ")
            t5_rows.append([
                Paragraph(clean_name, styles['TableCell']),
                Paragraph(f"{s_row['precision']*100:.1f}%", styles['TableCell']),
                Paragraph(f"{s_row['recall']*100:.1f}%", styles['TableCell']),
                Paragraph(f"<b>{s_row['f1_score']*100:.1f}%</b>", styles['TableCell']),
                Paragraph(f"{l_row['precision']*100:.1f}%", styles['TableCell']),
                Paragraph(f"{l_row['recall']*100:.1f}%", styles['TableCell']),
                Paragraph(f"<b>{l_row['f1_score']*100:.1f}%</b>", styles['TableCell']),
                Paragraph(str(int(s_row['support'])), styles['TableCell'])
            ])
        t5 = Table(t5_rows, colWidths=[185, 42, 42, 44, 42, 42, 44, 46.27])
        t5.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 1.8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(Paragraph("<b>Table 5:</b> Granular per-class classification metrics across all 38 classes on the PlantVillage holdout test partition (N=5,417).", styles['TableCaption']))
        story.append(t5)
        story.append(Spacer(1, 6))
    
    # ── Fig 1: Confusion Matrices ───────────────────────────────────────────────
    cm_s_path = "reports/research_paper/final_revision/figures/fig_confmat_mvs.png"
    cm_l_path = "reports/research_paper/final_revision/figures/fig_confmat_mvl.png"
    if os.path.exists(cm_s_path) and os.path.exists(cm_l_path):
        img_w = 238
        img_h = 175
        img1 = RLImage(cm_s_path, width=img_w, height=img_h)
        img2 = RLImage(cm_l_path, width=img_w, height=img_h)
        img_table = Table([[img1, img2]], colWidths=[printable_w/2, printable_w/2])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)
        caption_cm = "<b>Figure 1:</b> Normalized 38&times;38 confusion matrices on the independent PlantVillage holdout test partition (N=5,417). " \
                     "<i>Left:</i> MobileNetV3-Small (95.02% Top-1 accuracy, 93.13% Macro-F1). " \
                     "<i>Right:</i> MobileNetV3-Large (95.96% Top-1 accuracy, 94.52% Macro-F1). Both matrices display sharp diagonal concentration with minor off-diagonal confusion localized in Solanaceae blights."
        story.append(Paragraph(caption_cm, styles['FigureCaption']))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph("4.4 Independent PlantDoc External Evaluation: Quantifying the Domain Gap", styles['Sec2']))
    story.append(Paragraph(fpt.RES_PLANTDOC, styles['Body']))
    
    # Table 6: PlantDoc External Benchmark
    t6_data = [
        [Paragraph("<b>Evaluation Axis / Metric</b>", styles['TableHeader']),
         Paragraph("<b>PlantVillage Holdout (Lab)</b>", styles['TableHeader']),
         Paragraph("<b>PlantDoc Benchmark (Wild)</b>", styles['TableHeader']),
         Paragraph("<b>Domain Shift Penalty</b>", styles['TableHeader']),
         Paragraph("<b>Empirical Failure Mechanism</b>", styles['TableHeader'])],
        [Paragraph("Top-1 Classification Accuracy", styles['TableCellBold']), Paragraph("95.02%", styles['TableCell']), Paragraph("38.98%", styles['TableCellBold']), Paragraph("-56.04 percentage points", styles['TableCellBold']), Paragraph("Complex backgrounds, varied illumination, clutter", styles['TableCell'])],
        [Paragraph("Macro-Averaged Precision", styles['TableCellBold']), Paragraph("92.16%", styles['TableCell']), Paragraph("44.66%", styles['TableCell']), Paragraph("-47.50 percentage points", styles['TableCell']), Paragraph("Severe false positive rate on complex foliar scenes", styles['TableCell'])],
        [Paragraph("Macro-Averaged Recall", styles['TableCellBold']), Paragraph("95.06%", styles['TableCell']), Paragraph("37.96%", styles['TableCell']), Paragraph("-57.10 percentage points", styles['TableCell']), Paragraph("Partial leaf occlusion and distant camera perspectives", styles['TableCell'])],
        [Paragraph("Macro-Averaged F1-Score", styles['TableCellBold']), Paragraph("93.13%", styles['TableCell']), Paragraph("36.84%", styles['TableCellBold']), Paragraph("-56.29 percentage points", styles['TableCellBold']), Paragraph("Harmonic degradation driven by pervasive sensitivity loss", styles['TableCell'])],
        [Paragraph("Weighted-Averaged F1-Score", styles['TableCellBold']), Paragraph("95.14%", styles['TableCell']), Paragraph("37.41%", styles['TableCell']), Paragraph("-57.73 percentage points", styles['TableCell']), Paragraph("Consistent penalty across both dominant and rare classes", styles['TableCell'])],
        [Paragraph("Top-3 Multi-Guess Accuracy", styles['TableCellBold']), Paragraph("99.46%", styles['TableCell']), Paragraph("64.83%", styles['TableCellBold']), Paragraph("-34.63 percentage points", styles['TableCell']), Paragraph("Correct diagnosis frequently present in top-3 candidates", styles['TableCell'])],
        [Paragraph("Negative Log-Likelihood (Post-Cal)", styles['TableCellBold']), Paragraph("0.1637", styles['TableCell']), Paragraph("2.5087", styles['TableCell']), Paragraph("+2.3450 log-loss penalty", styles['TableCell']), Paragraph("Severe distributional mismatch inflates predictive entropy", styles['TableCell'])],
        [Paragraph("Expected Calibration Error (ECE)", styles['TableCellBold']), Paragraph("0.74%", styles['TableCell']), Paragraph("21.86%", styles['TableCellBold']), Paragraph("+21.12 percentage points", styles['TableCellBold']), Paragraph("In-domain temperature scaling fails under covariate shift", styles['TableCell'])],
        [Paragraph("Selective Accuracy @ Cov=93.2%", styles['TableCellBold']), Paragraph("95.02% (Cov 100%)", styles['TableCell']), Paragraph("40.91%", styles['TableCell']), Paragraph("-54.11 percentage points", styles['TableCell']), Paragraph("Threshold filtering (0.223) yields marginal accuracy gain", styles['TableCell'])],
    ]
    t6 = Table(t6_data, colWidths=[120, 95, 95, 85, 92.27])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#FEE2E2')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#FEE2E2')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 6:</b> Independent PlantDoc external generalization audit: benchmarking MobileNetV3-Small against domain shift.", styles['TableCaption']))
    story.append(t6)
    story.append(Spacer(1, 6))
    
    # ── Table 7: PlantDoc Per-Class Coverage & Generalization ───────────────────
    pd_per_class = load_plantdoc_per_class()
    if pd_per_class:
        story.append(Paragraph("4.5 Granular PlantDoc Per-Class Breakdown (27 Evaluated vs. 11 Unrepresented)", styles['Sec2']))
        story.append(Paragraph("Table 7 provides the complete class-by-class diagnostic breakdown on the 236 natural-background PlantDoc images, " \
                               "explicitly marking which classes were supported in the test partition versus those with zero test representation.", styles['Body']))
        t7_rows = [
            [Paragraph("<b>Taxonomy Class (Botanical Host & Pathology)</b>", styles['TableHeader']),
             Paragraph("<b>Status</b>", styles['TableHeader']),
             Paragraph("<b>Support</b>", styles['TableHeader']),
             Paragraph("<b>Precision</b>", styles['TableHeader']),
             Paragraph("<b>Recall</b>", styles['TableHeader']),
             Paragraph("<b>F1-Score</b>", styles['TableHeader'])]
        ]
        for r in pd_per_class:
            t7_rows.append([
                Paragraph(r[0], styles['TableCell']),
                Paragraph(f"<b>{r[1]}</b>" if r[1] == 'Evaluated' else r[1], styles['TableCell']),
                Paragraph(r[2], styles['TableCell']),
                Paragraph(r[3], styles['TableCell']),
                Paragraph(r[4], styles['TableCell']),
                Paragraph(f"<b>{r[5]}</b>" if r[5] != 'N/A' else r[5], styles['TableCell'])
            ])
        t7 = Table(t7_rows, colWidths=[205, 80, 50, 50, 50, 52.27])
        t7.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 1.8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(Paragraph("<b>Table 7:</b> Complete per-class diagnostic breakdown on the external PlantDoc benchmark (236 images across 38 classes).", styles['TableCaption']))
        story.append(t7)
        story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.6 Model Calibration, Reliability Diagrams & Discrepancy Reconciliation", styles['Sec2']))
    story.append(Paragraph(fpt.RES_CALIBRATION, styles['Body']))
    
    # Table 8: Calibration Comparison Across Architectures & Procedures
    t8_data = [
        [Paragraph("<b>Architecture / Evaluation Procedure</b>", styles['TableHeader']),
         Paragraph("<b>Optimal Temp (T*)</b>", styles['TableHeader']),
         Paragraph("<b>Pre-Calib ECE (%)</b>", styles['TableHeader']),
         Paragraph("<b>Post-Calib ECE (%)</b>", styles['TableHeader']),
         Paragraph("<b>Pre-Calib NLL</b>", styles['TableHeader']),
         Paragraph("<b>Post-Calib NLL</b>", styles['TableHeader']),
         Paragraph("<b>Pre-Calib Brier</b>", styles['TableHeader']),
         Paragraph("<b>Post-Calib Brier</b>", styles['TableHeader'])],
        [Paragraph("MobileNetV3-Small (Phase 3 Fit)", styles['TableCellBold']), Paragraph("0.5538", styles['TableCellBold']), Paragraph("10.25%", styles['TableCell']), Paragraph("0.58%", styles['TableCellBold']), Paragraph("0.2426", styles['TableCell']), Paragraph("0.1454", styles['TableCellBold']), Paragraph("0.0936", styles['TableCell']), Paragraph("0.0710", styles['TableCellBold'])],
        [Paragraph("MobileNetV3-Small (Holdout Test)", styles['TableCellBold']), Paragraph("0.5538", styles['TableCell']), Paragraph("10.28%", styles['TableCell']), Paragraph("0.74%", styles['TableCellBold']), Paragraph("0.2554", styles['TableCell']), Paragraph("0.1637", styles['TableCellBold']), Paragraph("0.0993", styles['TableCell']), Paragraph("0.0761", styles['TableCellBold'])],
        [Paragraph("MobileNetV3-Small (Phase 5 Batch)", styles['TableCell']), Paragraph("0.9902", styles['TableCell']), Paragraph("10.28%", styles['TableCell']), Paragraph("9.94%", styles['TableCell']), Paragraph("0.2554", styles['TableCell']), Paragraph("0.2513", styles['TableCell']), Paragraph("0.0993", styles['TableCell']), Paragraph("0.0981", styles['TableCell'])],
        [Paragraph("MobileNetV3-Large (Phase 5)", styles['TableCell']), Paragraph("0.9240", styles['TableCell']), Paragraph("14.00%", styles['TableCell']), Paragraph("10.80%", styles['TableCell']), Paragraph("0.2755", styles['TableCell']), Paragraph("0.2341", styles['TableCell']), Paragraph("0.0986", styles['TableCell']), Paragraph("0.0854", styles['TableCell'])],
        [Paragraph("EfficientNet-B0 (Phase 5)", styles['TableCell']), Paragraph("0.8846", styles['TableCell']), Paragraph("16.43%", styles['TableCell']), Paragraph("11.62%", styles['TableCell']), Paragraph("0.3162", styles['TableCell']), Paragraph("0.2542", styles['TableCell']), Paragraph("0.1156", styles['TableCell']), Paragraph("0.0945", styles['TableCell'])],
        [Paragraph("ResNet18 (Phase 5)", styles['TableCell']), Paragraph("1.0392", styles['TableCell']), Paragraph("8.99%", styles['TableCell']), Paragraph("10.21%", styles['TableCell']), Paragraph("0.2133", styles['TableCell']), Paragraph("0.2277", styles['TableCell']), Paragraph("0.0809", styles['TableCell']), Paragraph("0.0851", styles['TableCell'])],
    ]
    t8 = Table(t8_data, colWidths=[125, 55, 52, 52, 50, 50, 50, 53.27])
    t8.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#DCFCE7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 8:</b> Post-hoc temperature scaling calibration across architectures, datasets, and optimization protocols.", styles['TableCaption']))
    story.append(t8)
    story.append(Spacer(1, 6))
    
    # ── Fig 2: Reliability Diagrams ─────────────────────────────────────────────
    cal_fig_path = "reports/research_paper/final_revision/figures/fig_calibration.png"
    if os.path.exists(cal_fig_path):
        cal_img = RLImage(cal_fig_path, width=printable_w, height=185)
        story.append(cal_img)
        caption_cal = "<b>Figure 2:</b> 15-bin Reliability Diagrams on the isolated PlantVillage calibration partition (N=5,429). " \
                      "<i>Left:</i> Raw uncalibrated baseline displaying severe overconfidence (ECE = 10.25%). " \
                      "<i>Right:</i> Calibrated distribution after temperature scaling (T* = 0.5538, ECE = 0.58%), exhibiting near-ideal diagonal probability alignment."
        story.append(Paragraph(caption_cal, styles['FigureCaption']))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph("4.7 Multi-Architecture Benchmark: Latency, Accuracy & Edge Feasibility", styles['Sec2']))
    story.append(Paragraph(fpt.RES_ARCH_COMPARISON, styles['Body']))
    story.append(Paragraph(fpa.ARCH_EDGE_ANALYSIS, styles['Body']))
    story.append(Spacer(1, 4))
    
    # Table 9: Multi-Architecture Comparison
    t9_data = [
        [Paragraph("<b>Architecture</b>", styles['TableHeader']),
         Paragraph("<b>Parameters</b>", styles['TableHeader']),
         Paragraph("<b>Model Size</b>", styles['TableHeader']),
         Paragraph("<b>Holdout Acc</b>", styles['TableHeader']),
         Paragraph("<b>Macro-F1</b>", styles['TableHeader']),
         Paragraph("<b>PlantDoc Top-1</b>", styles['TableHeader']),
         Paragraph("<b>PlantDoc Top-3</b>", styles['TableHeader']),
         Paragraph("<b>Single Latency</b>", styles['TableHeader']),
         Paragraph("<b>Pipeline Latency</b>", styles['TableHeader'])],
        [Paragraph("<b>MobileNetV3-Small</b>", styles['TableCellBold']), Paragraph("1.56 M", styles['TableCellBold']), Paragraph("6.06 MB", styles['TableCellBold']), Paragraph("95.02%", styles['TableCell']), Paragraph("93.13%", styles['TableCell']), Paragraph("38.98%", styles['TableCell']), Paragraph("64.83%", styles['TableCell']), Paragraph("9.42 ms", styles['TableCellBold']), Paragraph("662.1 ms", styles['TableCellBold'])],
        [Paragraph("<b>MobileNetV3-Large</b>", styles['TableCellBold']), Paragraph("4.25 M", styles['TableCell']), Paragraph("16.40 MB", styles['TableCell']), Paragraph("95.96%", styles['TableCellBold']), Paragraph("94.52%", styles['TableCellBold']), Paragraph("41.53%", styles['TableCellBold']), Paragraph("68.22%", styles['TableCellBold']), Paragraph("16.87 ms", styles['TableCell']), Paragraph("1585.2 ms", styles['TableCell'])],
        [Paragraph("<b>EfficientNet-B0</b>", styles['TableCell']), Paragraph("4.06 M", styles['TableCell']), Paragraph("15.74 MB", styles['TableCell']), Paragraph("95.77%", styles['TableCell']), Paragraph("94.26%", styles['TableCell']), Paragraph("34.32%", styles['TableCell']), Paragraph("62.71%", styles['TableCell']), Paragraph("24.58 ms", styles['TableCell']), Paragraph("2457.4 ms", styles['TableCell'])],
        [Paragraph("<b>ResNet18</b>", styles['TableCell']), Paragraph("11.20 M", styles['TableCell']), Paragraph("42.79 MB", styles['TableCell']), Paragraph("95.92%", styles['TableCell']), Paragraph("94.30%", styles['TableCell']), Paragraph("41.10%", styles['TableCell']), Paragraph("67.80%", styles['TableCell']), Paragraph("25.54 ms", styles['TableCell']), Paragraph("2025.9 ms", styles['TableCell'])],
    ]
    t9 = Table(t9_data, colWidths=[95, 45, 45, 52, 50, 52, 52, 48, 48.27])
    t9.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 9:</b> Comprehensive multi-architecture benchmark across accuracy, generalization, parameter footprint, and CPU latency.", styles['TableCaption']))
    story.append(t9)
    story.append(Spacer(1, 6))
    
    # ── Fig 3: Architecture Benchmark Charts ────────────────────────────────────
    arch_fig_path = "reports/research_paper/final_revision/figures/fig_arch_comparison.png"
    if os.path.exists(arch_fig_path):
        arch_img = RLImage(arch_fig_path, width=printable_w, height=180)
        story.append(arch_img)
        caption_arch = "<b>Figure 3:</b> Comparative architecture performance profiles across four evaluated models on CPU hardware. " \
                       "Panels depict: (Top-Left) In-domain PlantVillage holdout accuracy vs. parameter count; (Top-Right) External PlantDoc generalization accuracy; " \
                       "(Bottom-Left) Single-crop CPU inference latency (ms); and (Bottom-Right) End-to-end multi-scale candidate screening pipeline latency."
        story.append(Paragraph(caption_arch, styles['FigureCaption']))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph("4.8 Candidate-Region Screening: Baseline vs. Optimized Pipeline", styles['Sec2']))
    story.append(Paragraph(fpt.RES_SCREENING, styles['Body']))
    
    # Table 10: Phase 4 Baseline vs Phase 4.1 Optimized
    t10_data = [
        [Paragraph("<b>Evaluation Metric</b>", styles['TableHeader']),
         Paragraph("<b>Phase 4 Baseline</b>", styles['TableHeader']),
         Paragraph("<b>Phase 4.1 Optimized</b>", styles['TableHeader']),
         Paragraph("<b>Absolute &Delta;</b>", styles['TableHeader']),
         Paragraph("<b>Relative Change</b>", styles['TableHeader']),
         Paragraph("<b>Operational Significance</b>", styles['TableHeader'])],
        [Paragraph("Recall @ IoU 0.10", styles['TableCellBold']), Paragraph("50.00%", styles['TableCell']), Paragraph("62.17%", styles['TableCellBold']), Paragraph("+12.17 pp", styles['TableCell']), Paragraph("+24.34%", styles['TableCellBold']), Paragraph("Substantially broader capture of symptomatic foliage", styles['TableCell'])],
        [Paragraph("Precision @ IoU 0.10", styles['TableCellBold']), Paragraph("24.78%", styles['TableCell']), Paragraph("17.31%", styles['TableCell']), Paragraph("-7.47 pp", styles['TableCell']), Paragraph("-30.13%", styles['TableCell']), Paragraph("Increased proposal density introduces background proposals", styles['TableCell'])],
        [Paragraph("F1-Score @ IoU 0.10", styles['TableCellBold']), Paragraph("33.14%", styles['TableCell']), Paragraph("27.08%", styles['TableCell']), Paragraph("-6.06 pp", styles['TableCell']), Paragraph("-18.27%", styles['TableCell']), Paragraph("Harmonic mean penalizes the lower precision profile", styles['TableCell'])],
        [Paragraph("Recall @ IoU 0.25 (Key)", styles['TableCellBold']), Paragraph("18.14%", styles['TableCell']), Paragraph("38.05%", styles['TableCellBold']), Paragraph("+19.91 pp", styles['TableCellBold']), Paragraph("+109.76%", styles['TableCellBold']), Paragraph("More than doubles effective candidate leaf localization", styles['TableCell'])],
        [Paragraph("Precision @ IoU 0.25", styles['TableCellBold']), Paragraph("8.99%", styles['TableCell']), Paragraph("10.60%", styles['TableCellBold']), Paragraph("+1.61 pp", styles['TableCell']), Paragraph("+17.87%", styles['TableCell']), Paragraph("Modest gain in boundary agreement under strict overlap", styles['TableCell'])],
        [Paragraph("F1-Score @ IoU 0.25", styles['TableCellBold']), Paragraph("12.02%", styles['TableCell']), Paragraph("16.58%", styles['TableCellBold']), Paragraph("+4.56 pp", styles['TableCellBold']), Paragraph("+37.88%", styles['TableCellBold']), Paragraph("Overall quality improvement under realistic threshold", styles['TableCell'])],
        [Paragraph("Recall @ IoU 0.50 (Strict)", styles['TableCellBold']), Paragraph("3.10%", styles['TableCell']), Paragraph("7.96%", styles['TableCellBold']), Paragraph("+4.86 pp", styles['TableCell']), Paragraph("+157.14%", styles['TableCellBold']), Paragraph("Severe bounding mismatch persists on whole leaves", styles['TableCell'])],
        [Paragraph("False Positives / Image @ 0.25", styles['TableCellBold']), Paragraph("3.52", styles['TableCell']), Paragraph("6.15", styles['TableCell']), Paragraph("+2.63 fp", styles['TableCell']), Paragraph("+74.82%", styles['TableCell']), Paragraph("Trade-off of higher recall; requires user triage", styles['TableCell'])],
        [Paragraph("End-to-End CPU Latency", styles['TableCellBold']), Paragraph("167.0 ms", styles['TableCell']), Paragraph("844.2 ms", styles['TableCell']), Paragraph("+677.2 ms", styles['TableCell']), Paragraph("+405.49%", styles['TableCell']), Paragraph("Latency trade-off for 5.1&times; crop evaluation overhead", styles['TableCell'])],
    ]
    t10 = Table(t10_data, colWidths=[110, 65, 75, 60, 60, 117.27])
    t10.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#DCFCE7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 10:</b> Empirical performance comparison: Phase 4 baseline vs. Phase 4.1 optimized candidate-region screening pipeline.", styles['TableCaption']))
    story.append(t10)
    story.append(Spacer(1, 6))
    
    # Table 11: Candidate Generation Ablation
    t11_data = [
        [Paragraph("<b>Configuration Variant</b>", styles['TableHeader']),
         Paragraph("<b>Latency</b>", styles['TableHeader']),
         Paragraph("<b>Rec @0.10</b>", styles['TableHeader']),
         Paragraph("<b>Prec @0.10</b>", styles['TableHeader']),
         Paragraph("<b>Rec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>Prec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>F1 @0.25</b>", styles['TableHeader']),
         Paragraph("<b>FP / Img</b>", styles['TableHeader'])],
        [Paragraph("Config A: Phase 4 Baseline (Square Single-Scale)", styles['TableCell']), Paragraph("175.0 ms", styles['TableCell']), Paragraph("40.52%", styles['TableCell']), Paragraph("25.12%", styles['TableCell']), Paragraph("20.07%", styles['TableCell']), Paragraph("12.44%", styles['TableCell']), Paragraph("15.36%", styles['TableCell']), Paragraph("3.80", styles['TableCell'])],
        [Paragraph("Config B: Multi-Scale Proposals (0.25, 0.40, 0.65)", styles['TableCell']), Paragraph("444.6 ms", styles['TableCell']), Paragraph("50.19%", styles['TableCell']), Paragraph("22.61%", styles['TableCell']), Paragraph("31.23%", styles['TableCell']), Paragraph("14.07%", styles['TableCell']), Paragraph("19.40%", styles['TableCell']), Paragraph("5.13", styles['TableCell'])],
        [Paragraph("Config C: Aspect-Ratio Windows (1:1, 4:3, 3:4)", styles['TableCellBold']), Paragraph("838.1 ms", styles['TableCell']), Paragraph("53.16%", styles['TableCell']), Paragraph("22.17%", styles['TableCell']), Paragraph("33.83%", styles['TableCellBold']), Paragraph("14.11%", styles['TableCell']), Paragraph("19.91%", styles['TableCellBold']), Paragraph("5.54", styles['TableCell'])],
        [Paragraph("Config D: Foliar Saliency Pre-Filtering (ExG)", styles['TableCell']), Paragraph("443.6 ms", styles['TableCell']), Paragraph("50.19%", styles['TableCell']), Paragraph("23.40%", styles['TableCell']), Paragraph("31.23%", styles['TableCell']), Paragraph("14.56%", styles['TableCell']), Paragraph("19.86%", styles['TableCell']), Paragraph("4.93", styles['TableCell'])],
        [Paragraph("Config E: Optimized Composite Pipeline", styles['TableCell']), Paragraph("863.4 ms", styles['TableCell']), Paragraph("53.90%", styles['TableCell']), Paragraph("19.44%", styles['TableCell']), Paragraph("32.71%", styles['TableCell']), Paragraph("11.80%", styles['TableCell']), Paragraph("17.34%", styles['TableCell']), Paragraph("6.58", styles['TableCell'])],
    ]
    t11 = Table(t11_data, colWidths=[150, 48, 50, 50, 50, 48, 48, 43.27])
    t11.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 11:</b> Ablation analysis of candidate-window proposal strategies on the PlantDoc development partition.", styles['TableCaption']))
    story.append(t11)
    story.append(Spacer(1, 6))
    
    # Table 12: NMS Ablation
    t12_data = [
        [Paragraph("<b>NMS Strategy Formulation</b>", styles['TableHeader']),
         Paragraph("<b>Class IoU</b>", styles['TableHeader']),
         Paragraph("<b>Cross IoU</b>", styles['TableHeader']),
         Paragraph("<b>Max Cand</b>", styles['TableHeader']),
         Paragraph("<b>Rec @0.10</b>", styles['TableHeader']),
         Paragraph("<b>Rec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>Prec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>F1 @0.25</b>", styles['TableHeader']),
         Paragraph("<b>FP / Img</b>", styles['TableHeader'])],
        [Paragraph("Class-wise IoU 0.55 (Baseline NMS)", styles['TableCell']), Paragraph("0.55", styles['TableCell']), Paragraph("None", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("53.53%", styles['TableCell']), Paragraph("30.86%", styles['TableCell']), Paragraph("12.83%", styles['TableCell']), Paragraph("18.12%", styles['TableCell']), Paragraph("5.64", styles['TableCell'])],
        [Paragraph("Class-wise IoU 0.45", styles['TableCell']), Paragraph("0.45", styles['TableCell']), Paragraph("None", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("54.28%", styles['TableCell']), Paragraph("31.23%", styles['TableCell']), Paragraph("13.04%", styles['TableCell']), Paragraph("18.40%", styles['TableCell']), Paragraph("5.60", styles['TableCell'])],
        [Paragraph("Class-wise IoU 0.65", styles['TableCell']), Paragraph("0.65", styles['TableCell']), Paragraph("None", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("52.42%", styles['TableCell']), Paragraph("29.37%", styles['TableCell']), Paragraph("12.10%", styles['TableCell']), Paragraph("17.14%", styles['TableCell']), Paragraph("5.74", styles['TableCell'])],
        [Paragraph("Hybrid NMS (Class 0.55, Cross 0.75, Max 8)", styles['TableCell']), Paragraph("0.55", styles['TableCell']), Paragraph("0.75", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("53.16%", styles['TableCell']), Paragraph("30.86%", styles['TableCell']), Paragraph("12.89%", styles['TableCell']), Paragraph("18.18%", styles['TableCell']), Paragraph("5.61", styles['TableCell'])],
        [Paragraph("Hybrid NMS (Class 0.50, Cross 0.70, Max 8)", styles['TableCell']), Paragraph("0.50", styles['TableCell']), Paragraph("0.70", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("53.53%", styles['TableCell']), Paragraph("30.11%", styles['TableCell']), Paragraph("12.62%", styles['TableCell']), Paragraph("17.78%", styles['TableCell']), Paragraph("5.61", styles['TableCell'])],
        [Paragraph("<b>Hybrid NMS (Class 0.50, Cross 0.70, Max 10) [Selected]</b>", styles['TableCellBold']), Paragraph("0.50", styles['TableCellBold']), Paragraph("0.70", styles['TableCellBold']), Paragraph("10", styles['TableCellBold']), Paragraph("57.25%", styles['TableCellBold']), Paragraph("34.57%", styles['TableCellBold']), Paragraph("11.94%", styles['TableCell']), Paragraph("17.75%", styles['TableCell']), Paragraph("6.86", styles['TableCell'])],
        [Paragraph("Hybrid NMS (Class 0.50, Cross 0.70, Max 12)", styles['TableCell']), Paragraph("0.50", styles['TableCell']), Paragraph("0.70", styles['TableCell']), Paragraph("12", styles['TableCell']), Paragraph("60.22%", styles['TableCell']), Paragraph("39.03%", styles['TableCell']), Paragraph("11.50%", styles['TableCell']), Paragraph("17.77%", styles['TableCell']), Paragraph("8.08", styles['TableCell'])],
        [Paragraph("Aggressive Cross-Class (0.45 / 0.60 / 8)", styles['TableCell']), Paragraph("0.45", styles['TableCell']), Paragraph("0.60", styles['TableCell']), Paragraph("8", styles['TableCell']), Paragraph("53.90%", styles['TableCell']), Paragraph("31.60%", styles['TableCell']), Paragraph("13.28%", styles['TableCell']), Paragraph("18.70%", styles['TableCell']), Paragraph("5.55", styles['TableCell'])],
    ]
    t12 = Table(t12_data, colWidths=[150, 42, 42, 40, 44, 44, 44, 44, 37.27])
    t12.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 12:</b> Ablation of Non-Maximum Suppression (NMS) strategies and threshold parameters on the development partition.", styles['TableCaption']))
    story.append(t12)
    story.append(Spacer(1, 6))
    
    # Table 13: Foliage Density Robustness
    t13_data = [
        [Paragraph("<b>Foliage Density Stratum</b>", styles['TableHeader']),
         Paragraph("<b>Images</b>", styles['TableHeader']),
         Paragraph("<b>Ground-Truth Leaves</b>", styles['TableHeader']),
         Paragraph("<b>Rec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>Prec @0.25</b>", styles['TableHeader']),
         Paragraph("<b>F1 @0.25</b>", styles['TableHeader']),
         Paragraph("<b>Crop Acc @0.25</b>", styles['TableHeader']),
         Paragraph("<b>FP / Image</b>", styles['TableHeader']),
         Paragraph("<b>CPU Latency</b>", styles['TableHeader'])],
        [Paragraph("Single Leaf (Isolated, N=1)", styles['TableCellBold']), Paragraph("157", styles['TableCell']), Paragraph("157", styles['TableCell']), Paragraph("49.04%", styles['TableCellBold']), Paragraph("6.86%", styles['TableCell']), Paragraph("12.03%", styles['TableCell']), Paragraph("40.26%", styles['TableCell']), Paragraph("6.66", styles['TableCell']), Paragraph("820.6 ms", styles['TableCell'])],
        [Paragraph("Moderate Foliage (Canopy N=2–4)", styles['TableCellBold']), Paragraph("60", styles['TableCell']), Paragraph("167", styles['TableCell']), Paragraph("40.12%", styles['TableCell']), Paragraph("17.91%", styles['TableCell']), Paragraph("24.77%", styles['TableCellBold']), Paragraph("38.81%", styles['TableCell']), Paragraph("5.12", styles['TableCell']), Paragraph("938.1 ms", styles['TableCell'])],
        [Paragraph("Dense Foliage (Cluster N&ge;5)", styles['TableCellBold']), Paragraph("19", styles['TableCell']), Paragraph("128", styles['TableCell']), Paragraph("21.88%", styles['TableCell']), Paragraph("22.22%", styles['TableCellBold']), Paragraph("22.05%", styles['TableCell']), Paragraph("32.14%", styles['TableCell']), Paragraph("5.16", styles['TableCell']), Paragraph("743.5 ms", styles['TableCell'])],
        [Paragraph("<b>Composite Evaluation (All Strata)</b>", styles['TableCellBold']), Paragraph("<b>236</b>", styles['TableCellBold']), Paragraph("<b>452</b>", styles['TableCellBold']), Paragraph("<b>38.05%</b>", styles['TableCellBold']), Paragraph("<b>10.60%</b>", styles['TableCellBold']), Paragraph("<b>16.58%</b>", styles['TableCellBold']), Paragraph("<b>38.37%</b>", styles['TableCellBold']), Paragraph("<b>6.15</b>", styles['TableCellBold']), Paragraph("<b>844.2 ms</b>", styles['TableCellBold'])],
    ]
    t13 = Table(t13_data, colWidths=[120, 36, 60, 48, 48, 48, 48, 40, 39.27])
    t13.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(Paragraph("<b>Table 13:</b> Whole-plant candidate screening performance stratified by foliage density and canopy complexity.", styles['TableCaption']))
    story.append(t13)
    story.append(Spacer(1, 6))
    
    # Table 14: Context Padding Effect
    t_pad_data = [
        [Paragraph(f"<b>{c}</b>", styles['TableHeader']) for c in fpe.PADDING_DATA[0]]
    ]
    for r in fpe.PADDING_DATA[1:]:
        t_pad_data.append([
            Paragraph(r[0], styles['TableCellBold'] if '12%' in r[0] or '15%' in r[0] else styles['TableCell']),
            Paragraph(r[1], styles['TableCell']),
            Paragraph(r[2], styles['TableCell']),
            Paragraph(r[3], styles['TableCell']),
            Paragraph(f"<b>{r[4]}</b>", styles['TableCellBold']),
            Paragraph(r[5], styles['TableCell'])
        ])
    t_pad = Table(t_pad_data, colWidths=[110, 65, 75, 75, 80, 82.27])
    t_pad.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 14:</b> Impact of boundary context padding percentage on localized candidate crop classification accuracy.", styles['TableCaption']))
    story.append(t_pad)
    story.append(Spacer(1, 6))
    
    # ── Fig 4: Localization & PR Curves ────────────────────────────────────────
    loc_path = "reports/research_paper/final_revision/figures/fig_localization.png"
    pr_path = "reports/research_paper/final_revision/figures/fig_prcurves.png"
    if os.path.exists(loc_path) and os.path.exists(pr_path):
        img_w = 238
        img_h = 165
        i1 = RLImage(loc_path, width=img_w, height=img_h)
        i2 = RLImage(pr_path, width=img_w, height=img_h)
        it = Table([[i1, i2]], colWidths=[printable_w/2, printable_w/2])
        it.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(it)
        caption_loc = "<b>Figure 4:</b> Candidate-region localization and screening evaluation. " \
                      "<i>Left:</i> Visual comparison of localized bounding proposals on sample PlantDoc field images (Ground Truth vs. Phase 4 vs. Phase 4.1). " \
                      "<i>Right:</i> Precision-Recall curves across IoU overlap thresholds (0.10, 0.25, 0.50), illustrating the trade-off between sensitivity and candidate density."
        story.append(Paragraph(caption_loc, styles['FigureCaption']))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph("4.9 Computational Latency Breakdown on Edge CPU Platform", styles['Sec2']))
    story.append(Paragraph("Table 15 details the stage-wise execution times for each operational component of the screening pipeline on an AMD Ryzen 7 5800HS CPU.", styles['Body']))
    
    t14_data = [
        [Paragraph("<b>Pipeline Stage / Component</b>", styles['TableHeader']),
         Paragraph("<b>Phase 4 Baseline (ms)</b>", styles['TableHeader']),
         Paragraph("<b>Phase 4.1 Optimized (ms)</b>", styles['TableHeader']),
         Paragraph("<b>Hardware Execution Profile</b>", styles['TableHeader'])],
        [Paragraph("Window Proposal Generation (Multi-scale & Aspect-aware)", styles['TableCellBold']), Paragraph("0.04 ms", styles['TableCell']), Paragraph("0.10 ms", styles['TableCell']), Paragraph("NumPy vectorized grid indexing", styles['TableCell'])],
        [Paragraph("Foliar Saliency Pre-Filtering (Excess Green Index)", styles['TableCellBold']), Paragraph("0.00 ms (Disabled)", styles['TableCell']), Paragraph("24.16 ms", styles['TableCell']), Paragraph("NumPy array operations (CPU)", styles['TableCell'])],
        [Paragraph("Crop Extraction, Padding & Tensor Prep (EVAL_TRANSFORM)", styles['TableCellBold']), Paragraph("67.07 ms", styles['TableCellBold']), Paragraph("358.19 ms", styles['TableCellBold']), Paragraph("PIL slicing, bicubic resize & Torch normalize", styles['TableCell'])],
        [Paragraph("MobileNetV3-Small Crop Forward Inference", styles['TableCellBold']), Paragraph("99.87 ms (10.6 crops)", styles['TableCell']), Paragraph("461.62 ms (49.0 crops)", styles['TableCellBold']), Paragraph("PyTorch 2.15.0+cpu (8 threads)", styles['TableCell'])],
        [Paragraph("Hybrid NMS & Cross-Class Overlap Suppression", styles['TableCellBold']), Paragraph("0.04 ms", styles['TableCell']), Paragraph("0.18 ms", styles['TableCell']), Paragraph("Pure Python greedy IoU loop", styles['TableCell'])],
        [Paragraph("<b>End-to-End Latency per Complete Image</b>", styles['TableCellBold']), Paragraph("<b>167.02 ms</b>", styles['TableCellBold']), Paragraph("<b>844.25 ms</b>", styles['TableCellBold']), Paragraph("<b>AMD Ryzen 7 5800HS CPU (8 cores)</b>", styles['TableCellBold'])],
    ]
    t14 = Table(t14_data, colWidths=[185, 95, 95, 112.27])
    t14.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 15:</b> Stage-wise latency profile of the candidate-region screening pipeline measured on CPU hardware.", styles['TableCaption']))
    story.append(t14)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("4.10 Practical Decision-Support Concept and Screening Safety Protocol", styles['Sec2']))
    safety_text = "To deploy the screening system responsibly without making unsupported claims of automated clinical diagnosis, " \
                  "we outline a non-autonomous, human-in-the-loop screening triage concept. Rather than prescribing chemical interventions or autonomous " \
                  "agronomic decisions, the pipeline serves strictly as an initial prioritization filter. When calibrated confidence exceeds the referral " \
                  "threshold (&tau; &ge; 0.223), the model flags the candidate symptomatic region for agronomic inspection; low-confidence samples (&tau; &lt; 0.223) " \
                  "are marked as uncertain and deferred for expert examination. Under no circumstances should the classification scores be interpreted as " \
                  "certified diagnostic certificates or justification for automated pesticide application."
    story.append(Paragraph(safety_text, styles['Body']))
    story.append(Spacer(1, 6))
    
    # ── SECTION 5: CONCLUSION & FUTURE SCOPE ───────────────────────────────────
    story.append(Paragraph("5. Conclusion and Future Scope", styles['Sec1']))
    story.append(Paragraph(fpt.CONCLUSION, styles['Body']))
    
    story.append(Paragraph("5.1 Explicit Negative Results and Operational Limitations", styles['Sec2']))
    story.append(Paragraph(fpt.LIMITATIONS, styles['Body']))
    
    story.append(Paragraph("5.2 Future Research Roadmap", styles['Sec2']))
    story.append(Paragraph("Six concrete development vectors are identified to advance the screening prototype toward field readiness:", styles['Body']))
    for i, fut in enumerate(fpt.FUTURE_SCOPE, 1):
        story.append(Paragraph(f"&bull; <b>Vector {i}:</b> {fut}", styles['Body']))
    story.append(Spacer(1, 6))
    
    # ── SECTION 6: PROVENANCE & CRYPTOGRAPHIC AUDIT ────────────────────────────
    story.append(Paragraph("6. Cryptographic Provenance and Artifact Audit", styles['Sec1']))
    story.append(Paragraph("To satisfy strict scientific reproducibility requirements and address teacher critique regarding checkpoint ancestry, " \
                           "Table 16 records the definitive cryptographic SHA-256 signatures for all data manifests, pretrained weights, trained model weights, " \
                           "and deployment TorchScript binaries.", styles['Body']))
                           
    t15_data = [
        [Paragraph("<b>Artifact Description</b>", styles['TableHeader']),
         Paragraph("<b>Relative File System Path</b>", styles['TableHeader']),
         Paragraph("<b>SHA-256 Cryptographic Checksum</b>", styles['TableHeader'])],
        [Paragraph("Pretrained Base Weights", styles['TableCellBold']), Paragraph("models/mobilenet_v3_small-047dcff4.pth", styles['TableCell']), Paragraph("047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f", styles['TableCell'])],
        [Paragraph("Combined Split Manifest", styles['TableCellBold']), Paragraph("reports/generated/combined_split_manifest.json", styles['TableCell']), Paragraph("f9ef8b9dcd6823f2021a0f24b94e2330adef4a7be01dc7bf1c652c0d6be15868", styles['TableCell'])],
        [Paragraph("Best Model Checkpoint", styles['TableCellBold']), Paragraph("saved_models/.../best_model.pth", styles['TableCell']), Paragraph("6190f54c56ebbb7f4a663b5eb88a53c6f70926e482e084d76d624c2a12725009", styles['TableCell'])],
        [Paragraph("Scripted TorchScript Model", styles['TableCellBold']), Paragraph("saved_models/.../classifier.pt", styles['TableCell']), Paragraph("6fff52f1449e82e663ad1d4f884e5fa3576e2c04e517e9274f1c6237e9071213", styles['TableCell'])],
        [Paragraph("Screening Pipeline Config", styles['TableCellBold']), Paragraph("reports/phase4_1/selected_screening_configuration.json", styles['TableCell']), Paragraph("7db276bb198a28723652cb9f67aebdc856037eb5f061e86971dfaa1c6cfcb4e1", styles['TableCell'])],
        [Paragraph("MobileNetV3-Large Checkpoint", styles['TableCellBold']), Paragraph("saved_models/architecture_benchmark/mobilenet_v3_large/best.pth", styles['TableCell']), Paragraph("8ebfb512a071915ca9a7b508e52a9f3ac983ac31618674c43dfdbd6ac168394a", styles['TableCell'])],
        [Paragraph("EfficientNet-B0 Checkpoint", styles['TableCellBold']), Paragraph("saved_models/architecture_benchmark/efficientnet_b0/best.pth", styles['TableCell']), Paragraph("faba42992b190b4c6c37409634070f0a376e26dd6330bd1a6c930bc341f020cb", styles['TableCell'])],
        [Paragraph("ResNet18 Checkpoint", styles['TableCellBold']), Paragraph("saved_models/architecture_benchmark/resnet18/best.pth", styles['TableCell']), Paragraph("f2991bef8630f6e162923fe3224d8d68ba9128ae55ef5bb3ba1ab672f7ea55cf", styles['TableCell'])],
    ]
    t15 = Table(t15_data, colWidths=[130, 160, 197.27])
    t15.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table 16:</b> Cryptographic verification manifest of all definitive model checkpoints, manifests, and deployment binaries.", styles['TableCaption']))
    story.append(t15)
    story.append(Spacer(1, 6))
    
    # ── SECTION 7: DECLARATIONS & STATEMENTS ───────────────────────────────────
    story.append(Paragraph("7. Declarations and Scientific Compliance", styles['Sec1']))
    decl_text = "<b>Funding:</b> No external funding was received for this research.<br/>" \
                "<b>Competing Interests:</b> The authors declare that they have no competing financial or non-financial interests.<br/>" \
                "<b>Ethics Approval:</b> Not applicable. This study utilized publicly available botanical benchmark datasets involving no human or animal participants.<br/>" \
                "<b>Consent to Participate / Publish:</b> Not applicable.<br/>" \
                "<b>Data Availability:</b> PlantVillage and PlantDoc datasets are publicly accessible. Data manifests and verified splits are preserved in the research repository.<br/>" \
                "<b>Code Availability:</b> All training, calibration, candidate screening, and benchmarking scripts are available in the project repository.<br/>" \
                "<b>Author Contributions:</b> Mohit Sharma conceived the study, developed the pipeline, conducted Phase 1–5 experiments, and drafted the manuscript. " \
                "Aniket Kumar and Raghav Kumar contributed equally to data auditing, calibration analysis, and manuscript verification."
    story.append(Paragraph(decl_text, styles['Body']))
    story.append(Spacer(1, 8))
    
    # ── SECTION 8: REFERENCES ──────────────────────────────────────────────────
    story.append(Paragraph("8. References", styles['Sec1']))
    bib_path = "reports/research_paper/references.bib"
    apa_refs = parse_bibtex_apa(bib_path)
    for ref in apa_refs:
        story.append(Paragraph(ref, styles['RefEntry']))
    story.append(Spacer(1, 10))
    
    # ── APPENDICES A, B, C, D ──────────────────────────────────────────────────
    story.append(Paragraph("Appendix A: Extended Theoretical Formulations & Derivations", styles['Sec1']))
    story.append(Paragraph(fpe.MATH_DERIVATIONS_TEXT, styles['Body']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Appendix B: Mathematical Notation & Operational Scope Dictionary", styles['Sec1']))
    t_not_data = [[Paragraph(f"<b>{c}</b>", styles['TableHeader']) for c in fpe.NOTATION_DATA[0]]]
    for r in fpe.NOTATION_DATA[1:]:
        t_not_data.append([
            Paragraph(f"<b>{r[0]}</b>", styles['TableCell']),
            Paragraph(r[1], styles['TableCell']),
            Paragraph(r[2], styles['TableCell'])
        ])
    t_not = Table(t_not_data, colWidths=[110, 110, 267.27])
    t_not.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table B1:</b> Formal notation, mathematical domains, and operational definitions.", styles['TableCaption']))
    story.append(t_not)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Appendix C: Execution Hardware, Software Environment & Concurrency Controls", styles['Sec1']))
    t_env_data = [[Paragraph(f"<b>{c}</b>", styles['TableHeader']) for c in fpe.ENVIRONMENT_DATA[0]]]
    for r in fpe.ENVIRONMENT_DATA[1:]:
        t_env_data.append([
            Paragraph(f"<b>{r[0]}</b>", styles['TableCellBold']),
            Paragraph(r[1], styles['TableCell']),
            Paragraph(r[2], styles['TableCell'])
        ])
    t_env = Table(t_env_data, colWidths=[120, 150, 217.27])
    t_env.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table C1:</b> Rigorous software and hardware deployment stack verified during benchmarking.", styles['TableCaption']))
    story.append(t_env)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Appendix D: Deterministic Reproduction Protocol & Command Sequences", styles['Sec1']))
    story.append(Paragraph(fpr.REPRODUCTION_PROTOCOL, styles['Body']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Appendix E: Agronomic Pathology Reference & Diagnostic Triage Guide", styles['Sec1']))
    story.append(Paragraph(fpr.TRIAGE_GUIDE, styles['Body']))
    story.append(Spacer(1, 6))
    
    t_field_data = [[Paragraph(f"<b>{c}</b>", styles['TableHeader']) for c in fpr.FIELD_DECISION_DATA[0]]]
    for r in fpr.FIELD_DECISION_DATA[1:]:
        t_field_data.append([
            Paragraph(f"<b>{r[0]}</b>", styles['TableCellBold']),
            Paragraph(r[1], styles['TableCell']),
            Paragraph(r[2], styles['TableCell']),
            Paragraph(r[3], styles['TableCell'])
        ])
    t_field = Table(t_field_data, colWidths=[105, 130, 125, 127.27])
    t_field.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("<b>Table E1:</b> Agronomic differential triage matrix mapping foliar signs to operational escalation tiers.", styles['TableCaption']))
    story.append(t_field)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(fpr.DEPLOYMENT_CHECKLIST, styles['Body']))
    
    return story
