import os, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA = BASE/'dataset'; OUT = BASE/'outputs'; MET = BASE/'metrics'
OUT.mkdir(exist_ok=True, parents=True); MET.mkdir(exist_ok=True, parents=True)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(str(DATA/'train'), transform=tf)
test_ds  = datasets.ImageFolder(str(DATA/'test'),  transform=tf)
classes  = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

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

def extract(dl):
    Xs, Ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            fb = featnet(xb).cpu().numpy()
            Xs.append(fb); Ys.append(yb.numpy())
    return np.concatenate(Xs,0), np.concatenate(Ys,0)

Xtr, ytr = extract(train_loader)
Xte, yte = extract(test_loader)

svm = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', LinearSVC(C=1.0, max_iter=5000))])
svm.fit(Xtr, ytr)
yp = svm.predict(Xte)

rep = classification_report(yte, yp, target_names=classes, output_dict=True, zero_division=0)
cm = confusion_matrix(yte, yp)
acc = accuracy_score(yte, yp)
mf1 = f1_score(yte, yp, average='macro')

with open(MET/'supervised_metrics.json','w',encoding='utf-8') as f:
    json.dump({'report':rep, 'accuracy':acc, 'macro_f1':mf1}, f, indent=2)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix — Supervised SVM')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig(OUT/'cm_supervised.png', dpi=150); plt.close()
print('Saved supervised metrics and CM.')
