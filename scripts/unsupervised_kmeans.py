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
plt.figure(figsize=(5,4)); plt.scatter(xy[:,0], xy[:,1], c=y, s=12, alpha=0.85)
plt.title('PCA — TRUE labels'); plt.tight_layout(); plt.savefig(OUT/'pca_true.png', dpi=150); plt.close()
plt.figure(figsize=(5,4)); plt.scatter(xy[:,0], xy[:,1], c=labels, s=12, alpha=0.85)
plt.title('PCA — KMeans clusters'); plt.tight_layout(); plt.savefig(OUT/'pca_kmeans.png', dpi=150); plt.close()

with open(MET/'unsupervised.json','w',encoding='utf-8') as f:
    json.dump({'adjusted_rand':float(ari), 'nmi':float(nmi)}, f, indent=2)

print('Saved unsupervised metrics and PCA plots.')
