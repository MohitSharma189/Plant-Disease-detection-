# scratch/expand_paper.py
# Loads all additional data tables and text for building the 18+ page manuscript

import pandas as pd
import os

def load_plantdoc_per_class():
    path = "reports/phase3/plantdoc_classification_report.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = [["Class Name", "Status", "Support", "Precision", "Recall", "F1-Score"]]
    for _, r in df.iterrows():
        cname = str(r['class_name']).replace("___", " — ").replace("_", " ")
        status = str(r['status'])
        sup = str(int(r['support'])) if str(r['support']).isdigit() or isinstance(r['support'], (int, float)) else str(r['support'])
        p = f"{float(r['precision'])*100:.1f}%" if r['precision'] != 'N/A' and pd.notnull(r['precision']) else "N/A"
        rec = f"{float(r['recall'])*100:.1f}%" if r['recall'] != 'N/A' and pd.notnull(r['recall']) else "N/A"
        f1 = f"{float(r['f1_score'])*100:.1f}%" if r['f1_score'] != 'N/A' and pd.notnull(r['f1_score']) else "N/A"
        rows.append([cname, status, sup, p, rec, f1])
    return rows

def load_candidate_ablation():
    path = "reports/phase4_1/candidate_generation_comparison.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = [["Configuration Variant", "Latency (ms)", "Recall @0.10", "Recall @0.25", "Prec @0.25", "F1 @0.25", "FP/Img"]]
    for _, r in df.iterrows():
        cname = str(r['config_name'])
        lat = f"{float(r['mean_latency_ms']):.1f}"
        rec10 = f"{float(r['recall_0.10'])*100:.1f}%"
        rec25 = f"{float(r['recall_0.25'])*100:.1f}%"
        p25 = f"{float(r['precision_0.25'])*100:.1f}%"
        f1_25 = f"{float(r['f1_0.25'])*100:.1f}%"
        fp = f"{float(r['fp_per_img_0.25']):.2f}"
        rows.append([cname, lat, rec10, rec25, p25, f1_25, fp])
    return rows

def load_nms_ablation():
    path = "reports/phase4_1/nms_comparison.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = [["NMS Strategy", "Class IoU", "Cross IoU", "Max", "Rec @0.10", "Rec @0.25", "Prec @0.25", "F1 @0.25", "FP/Img"]]
    for _, r in df.iterrows():
        name = str(r['nms_strategy']).replace('"', '')
        c_iou = str(r['class_iou'])
        cr_iou = str(r['cross_iou']) if pd.notnull(r['cross_iou']) else "None"
        max_c = str(int(r['max_candidates']))
        r10 = f"{float(r['recall_iou10'])*100:.1f}%"
        r25 = f"{float(r['recall_iou25'])*100:.1f}%"
        p25 = f"{float(r['precision_iou25'])*100:.1f}%"
        f1_25 = f"{float(r['f1_iou25'])*100:.1f}%"
        fp = f"{float(r['fp_per_img_iou25']):.2f}"
        rows.append([name, c_iou, cr_iou, max_c, r10, r25, p25, f1_25, fp])
    return rows

def load_foliage_density():
    path = "reports/phase4_1/whole_plant_robustness.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = [["Foliage Density Stratum", "Images", "GT Leaves", "Localized (IoU>=0.25)", "Recall @0.25", "Prec @0.25", "F1 @0.25", "Latency (ms)"]]
    for _, r in df.iterrows():
        strat = str(r['foliage_density_stratum'])
        imgs = str(int(r['image_count']))
        gt = str(int(r['ground_truth_leaves']))
        loc = str(int(r['localized_leaves_iou25']))
        rec = f"{float(r['localization_recall_iou25'])*100:.1f}%"
        prec = f"{float(r['localization_precision_iou25'])*100:.1f}%"
        f1 = f"{float(r['localization_f1_iou25'])*100:.1f}%"
        lat = f"{float(r['mean_latency_ms']):.1f}"
        rows.append([strat, imgs, gt, loc, rec, prec, f1, lat])
    return rows

print("Data loaders defined successfully.")

