# Task 4 — Complete CV Project (Supervised + Unsupervised)  

This repository contains a **ready-to-run** solution for Task 4 with:
- A **synthetic image dataset** (4 classes, balanced, 70/15/15 splits).
- **Supervised** baseline: *SVM on ResNet-18 features*.
- **Unsupervised** baseline: *k-Means on the same features*.
- Standard metrics (**accuracy**, **precision**, **recall**, **F1**), **confusion matrix**, and **PCA** visualizations.
- An auto-generated short **report**.

> Everything is self-contained. Just install deps and run the pipeline.

---

## 1) Project Structure

```
task4_full_project/
├─ dataset/
│  ├─ train/  red_circle/ green_square/ blue_triangle/ yellow_star/
│  ├─ val/    red_circle/ green_square/ blue_triangle/ yellow_star/
│  ├─ test/   red_circle/ green_square/ blue_triangle/ yellow_star/
│  └─ infer/  *.jpg (unlabeled, optional)
├─ scripts/
│  ├─ train_supervised.py        # SVM on ResNet-18 features + confusion matrix
│  ├─ unsupervised_kmeans.py     # k-Means (ARI, NMI) + PCA plots
│  └─ analyze_and_report.py      # assembles metrics → report/REPORT.md
├─ notebooks/
│  └─ task4_all.ipynb            # runs the whole pipeline (Colab/Jupyter)
├─ outputs/                      # figures (created after run)
├─ metrics/                      # JSON metrics (created after run)
├─ report/
│  └─ REPORT.md                  # short auto-generated summary (after run)
├─ requirements.txt
├─ run_all.bat                   # one-click (Windows CMD)
└─ run_all.ps1                   # one-click (PowerShell)
```

**Dataset (included):**  
- **Classes (4):** `red_circle`, `green_square`, `blue_triangle`, `yellow_star`  
- **Per class:** 100 images → **train 500 / val 500 / test 500**  
- **Total labeled:** 500

---

## 2) How to Run (Windows)

### One-click
```bat
cmd /c run_all.bat
```

### Manual (same pipeline)
```bat
.\.venv\Scripts\python.exe scripts\\train_supervised.py
.\.venv\Scripts\python.exe scripts\\unsupervised_kmeans.py
.\.venv\Scripts\python.exe scripts\\analyze_and_report.py
```

> Dependencies are installed into `.venv` automatically by the runner. If you prefer manual setup:
> ```bat
> py -m venv .venv
> .\.venv\Scripts\python.exe -m pip install --upgrade pip
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> ```

---

## 3) Methods

### Supervised (with teacher)
- **Model:** Linear SVM on **ResNet-18** embeddings (penultimate layer).
- **Implementation:** `scripts/train_supervised.py`  
  Saves:
  - `metrics/supervised_metrics.json` — per-class precision/recall/F1 + macro/weighted + accuracy  
  - `outputs/cm_supervised.png` — confusion matrix

### Unsupervised (no labels)
- **Model:** **k-Means** on the same ResNet-18 embeddings (fair comparison).  
- **Implementation:** `scripts/unsupervised_kmeans.py`  
  Saves:
  - `metrics/unsupervised.json` — **ARI** (Adjusted Rand), **NMI** (Normalized Mutual Information)  
  - `outputs/pca_true.png`, `outputs/pca_kmeans.png` — PCA plots (true labels vs clusters)

> Both methods use identical features to compare supervised vs unsupervised fairly.

---

## 4) Preprocessing & Augmentation

- **Resize:** 224×224  
- **Normalize:** ImageNet stats (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)  
- **Augment (implicit in data generation):** light random rotation, blur, color jitter, varied background noise.  
- **Val/Test:** no augmentation (clean).

---

## 5) Results & Where to Find Them

After running:
- **Supervised metrics:** `metrics/supervised_metrics.json`  
  *(contains accuracy, macro/weighted precision/recall/F1 + per-class scores)*
- **Unsupervised metrics:** `metrics/unsupervised.json`  
  *(contains ARI & NMI)*
- **Visualizations:**
  - `outputs/cm_supervised.png` — confusion matrix (supervised)  
  - `outputs/pca_true.png` and `outputs/pca_kmeans.png` — PCA scatter plots (unsupervised)
- **Auto report:** `report/REPORT.md` — compact summary of both results.

> On this synthetic dataset, the supervised baseline is expected to be near-perfect, and k-Means typically achieves high ARI/NMI.

---

## 6) Reproducibility & Notes

- **Environment:** `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `Pillow`, `tqdm`, `seaborn` (see `requirements.txt`).  
- **Windows-safe loaders:** `num_workers=0`, conditional pin-memory.  
- **Troubleshooting (Windows):**
  - Ensure the **Python launcher** `py` exists: `py --version`  
  - If pip error: `py -m ensurepip --upgrade && py -m pip install --upgrade pip`
- **Extending the task:**
  - Replace `dataset/` with your own class-per-folder dataset (same splits).  
  - Re-run the pipeline; metrics and plots will be regenerated.  
  - Try deeper fine-tuning or alternative backbones if moving to real images.

---

## 7) Deliverables Checklist

- ✅ **Dataset** (zipped or in repo) with `train/val/test` class folders  
- ✅ **Notebook**: `notebooks/task4_all.ipynb`  
- ✅ **Supervised + Unsupervised** baselines with metrics  
- ✅ **Visualizations**: confusion matrix + PCA  
- ✅ **Short Report**: `report/REPORT.md`

> This README doubles as the project overview required by the assignment.
