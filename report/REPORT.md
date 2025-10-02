# Task 4 Report (Auto-generated)

## Dataset
- Synthetic shapes dataset with 4 classes: **red_circle**, **green_square**, **blue_triangle**, **yellow_star**
- Per class: 200 images → train 140 / val 30 / test 30 (total 800)

## Supervised (SVM on ResNet18 features)
- Accuracy: **1.0000**
- Macro-F1: **1.0000**
- See `outputs/cm_supervised.png`

## Unsupervised (k-Means on features)
- Adjusted Rand Index (ARI): **1.0000**
- Normalized Mutual Information (NMI): **1.0000**
- See PCA plots: `outputs/pca_true.png`, `outputs/pca_kmeans.png`

## Notes
- The dataset is deliberately simple; supervised method should be strong.
- To increase difficulty/generalization, add stronger augmentation, color jittering, random backgrounds, or move to real-world images.
