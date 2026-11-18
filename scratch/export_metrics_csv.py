import json, csv
from pathlib import Path

with open('reports/phase2/final/training_metrics.json', encoding='utf-8') as f:
    d = json.load(f)

for dir_path in [Path('reports/phase2/final'), Path('saved_models/experiments/mobilenet_v3_small_final')]:
    csv_path = dir_path / 'training_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as cf:
        w = csv.writer(cf)
        w.writerow(['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'validation_macro_f1', 'fine_tuned_backbone', 'seconds'])
        for h in d['history']:
            w.writerow([
                h['epoch'],
                f"{h['train_loss']:.4f}",
                f"{h['train_accuracy']:.4f}",
                f"{h['validation_loss']:.4f}",
                f"{h['validation_accuracy']:.4f}",
                f"{h['validation_macro_f1']:.4f}",
                h['fine_tuned_backbone'],
                f"{h['seconds']:.1f}"
            ])
print('Exported training_metrics.csv successfully.')

