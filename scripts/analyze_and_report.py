import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'outputs'; MET = BASE/'metrics'; REP = BASE/'report'
REP.mkdir(exist_ok=True, parents=True)

sup = json.loads((MET/'supervised_metrics.json').read_text(encoding='utf-8'))
uns = json.loads((MET/'unsupervised.json').read_text(encoding='utf-8'))

md = f"""# Task 4 Report (Auto-generated)

## Dataset
- Synthetic shapes dataset with 4 classes: **red_circle**, **green_square**, **blue_triangle**, **yellow_star**
- Per class: 200 images → train 140 / val 30 / test 30 (total 800)

## Supervised (SVM on ResNet18 features)
- Accuracy: **{sup['accuracy']:.4f}**
- Macro-F1: **{sup['macro_f1']:.4f}**
- See `outputs/cm_supervised.png`

## Unsupervised (k-Means on features)
- Adjusted Rand Index (ARI): **{uns['adjusted_rand']:.4f}**
- Normalized Mutual Information (NMI): **{uns['nmi']:.4f}**
- See PCA plots: `outputs/pca_true.png`, `outputs/pca_kmeans.png`

## Notes
- The dataset is deliberately simple; supervised method should be strong.
- To increase difficulty/generalization, add stronger augmentation, color jittering, random backgrounds, or move to real-world images.
"""

(REP/'REPORT.md').write_text(md, encoding='utf-8')
print('Saved', REP/'REPORT.md')
