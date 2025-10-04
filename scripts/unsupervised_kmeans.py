import json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE/'dataset'; OUT = BASE/'outputs'; MET = BASE/'metrics'
OUT.mkdir(exist_ok=True, parents=True); MET.mkdir(exist_ok=True, parents=True)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(str(DATA/'test'), transform=tf)
dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
classes = test_ds.classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet18Features(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            m = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.backbone(x); x = self.pool(x)
        return torch.flatten(x,1)

featnet = ResNet18Features().to(device).eval()


Xs, Ys = [], []
with torch.no_grad():
    for xb, yb in dl:
        xb = xb.to(device)
        fb = featnet(xb).cpu().numpy()
        Xs.append(fb); Ys.append(yb.numpy())
X = np.concatenate(Xs,0); y = np.concatenate(Ys,0)


kmeans = KMeans(n_clusters=len(classes), n_init=10, random_state=42)
labels = kmeans.fit_predict(X)


ari = adjusted_rand_score(y, labels)
nmi = normalized_mutual_info_score(y, labels)


xy = PCA(n_components=2, random_state=42).fit_transform(X)



fig1, ax1 = plt.subplots(figsize=(6,5))
for i, cls in enumerate(classes):
    m = (y == i)
    ax1.scatter(xy[m,0], xy[m,1], s=12, alpha=0.85, label=f"{cls} (n={int(m.sum())})")
ax1.set_title('PCA — TRUE labels')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.legend(title='Class (support)', fontsize=8)
fig1.tight_layout()
fig1.savefig(OUT/'pca_true.png', dpi=150)
plt.close(fig1)



K = len(classes); C = len(classes)
cm = np.zeros((K, C), dtype=int) 
cluster_summary = []

fig2, ax2 = plt.subplots(figsize=(6,5))
for k in range(K):
    mk = (labels == k)
    counts = np.bincount(y[mk], minlength=C)
    cm[k] = counts
    maj = int(np.argmax(counts))
    maj_count = int(counts[maj])
    total = int(mk.sum())

    ax2.scatter(xy[mk,0], xy[mk,1], s=12, alpha=0.85,
                label=f"Cluster {k} → {classes[maj]} ({maj_count}/{total})")

    cluster_summary.append({
        "cluster": k,
        "size": total,
        "majority_class": classes[maj],
        "majority_count": maj_count,
        "class_counts": {classes[i]: int(counts[i]) for i in range(C)}
    })

ax2.set_title('PCA — KMeans clusters (legend shows majority class)')
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')

ax2.text(0.02, 0.02, f"ARI={ari:.3f}\nNMI={nmi:.3f}",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
ax2.legend(title='Cluster → majority class (hits/size)', fontsize=8)
fig2.tight_layout()
fig2.savefig(OUT/'pca_kmeans.png', dpi=150)
plt.close(fig2)

with open(MET/'unsupervised.json','w',encoding='utf-8') as f:
    json.dump({
        'adjusted_rand': float(ari),
        'nmi': float(nmi),
        'clusters': cluster_summary
    }, f, indent=2, ensure_ascii=False)

print('Saved unsupervised metrics, legends, and PCA plots.')
