"""## Cell 1 — Imports"""

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split



"""## 2 - Config"""

class CFG:
    data_dir = "/kaggle/input/competitions/iith-deep-learning-2026-hackathon"
    img_size = 96
    batch_size = 64
    epochs = 45
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 2025

"""## 3 -SEED"""

def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(CFG.seed)

"""## Cell 4 — Dataset"""

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

"""## Cell 5 — Transform"""

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

"""## Cell 6 — Load and Split"""

train_dir = os.path.join(CFG.data_dir, "train", "train")

paths, labels = [], []

for label in ["0", "1"]:
    folder = os.path.join(train_dir, label)
    for img in os.listdir(folder):
        paths.append(os.path.join(folder, img))
        labels.append(int(label))

from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

"""## Cell 7.1 — Loaders"""

train_ds = ImageDataset(train_paths, train_labels, train_tfms)
val_ds = ImageDataset(val_paths, val_labels, val_tfms)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

"""# 8 ResNet"""

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c or stride != 1 else nn.Identity()

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 1, 1)

        self.layer1 = Block(32, 64, 2)
        self.layer2 = Block(64, 128, 2)
        self.layer3 = Block(128, 192, 2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(192, 1)
                    )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

"""## Cell 8 — Train Setup"""

model = Net().to(CFG.device)
class SmoothBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

criterion = SmoothBCE(0.05)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epochs)

"""## Cell 9 — Training Loop"""

best_acc = 0
best_loss = 1e9

for epoch in range(CFG.epochs):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(CFG.device), y.float().to(CFG.device)

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    preds, gt = [], []
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(CFG.device), y.float().to(CFG.device)

            out = model(x).squeeze()
            loss = criterion(out, y)

            val_loss += loss.item()

            prob = torch.sigmoid(out).cpu().numpy()
            preds.extend(prob)
            gt.extend(y.cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    acc = np.mean(np.array(preds_bin) == np.array(gt))

    print(f"Epoch {epoch+1} | Acc {acc:.4f} | Loss {val_loss:.4f} | Class1 {np.mean(preds_bin):.2f}")

    scheduler.step()

    # checkpoint 1: best accuracy
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_acc.pt")

    # checkpoint 2: best loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_loss.pt")

# checkpoint 3: last
torch.save(model.state_dict(), "last.pt")

"""## Inference"""

test_dir = os.path.join(CFG.data_dir, "test", "test")
test_imgs = sorted(os.listdir(test_dir))

model.load_state_dict(torch.load("best_acc.pt"))
model.eval()

preds = []

for img_name in tqdm(test_imgs):
    path = os.path.join(test_dir, img_name)
    img = Image.open(path).convert("RGB")

    imgs = [
        val_tfms(img),
        val_tfms(img.transpose(Image.FLIP_LEFT_RIGHT))
    ]

    imgs = torch.stack(imgs).to(CFG.device)

    with torch.no_grad():
        out = torch.sigmoid(model(imgs)).mean().item()

    preds.append(1 if out > 0.5 else 0)

df = pd.DataFrame({
    "ID": test_imgs,
    "Label": preds
})

df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

"""## Cell 10 — Evaluate each checkpoint (on validation)"""

import numpy as np
import torch

def evaluate_checkpoint(model, path, val_loader):
    model.load_state_dict(torch.load(path))
    model.eval()

    preds, gt = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(CFG.device)
            out = torch.sigmoid(model(x)).cpu().numpy().flatten()

            preds.extend(out)
            gt.extend(y.numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    acc = np.mean(np.array(preds_bin) == np.array(gt))
    class1_ratio = np.mean(preds_bin)

    print(f"{path} → Acc: {acc:.4f}, Class1: {class1_ratio:.2f}")

"""## Cell 11 - Calling Evaluate"""

evaluate_checkpoint(model, "best_acc.pt", val_loader)
evaluate_checkpoint(model, "best_loss.pt", val_loader)
evaluate_checkpoint(model, "last.pt", val_loader)

"""## Cell 12 - Generating predictions for each Checkpoint"""

def predict_with_model(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    preds = []

    for img_name in test_imgs:
        path = os.path.join(test_dir, img_name)
        img = Image.open(path).convert("RGB")

        img = val_tfms(img).unsqueeze(0).to(CFG.device)

        with torch.no_grad():
            out = torch.sigmoid(model(img)).item()

        preds.append(out)

    return np.array(preds)
p1 = predict_with_model(model, "best_acc.pt")
p2 = predict_with_model(model, "best_loss.pt")
p3 = predict_with_model(model, "last.pt")

"""## Cell -- 13 Trying individually"""

def save_preds(probs, name):
    preds = (probs > 0.5).astype(int)
    print(name, "Class1 %:", preds.mean())

    df = pd.DataFrame({
        "ID": test_imgs,
        "Label": preds
    })
    df.to_csv(name, index=False)

save_preds(p1, "sub_acc.csv")
save_preds(p2, "sub_loss.csv")
save_preds(p3, "sub_last.csv")

"""## Cell 14 -- Ensembling"""

ensemble = (p1 + p2 + p3) / 3
save_preds(ensemble, "sub_ensemble.csv")

import pandas as pd

df = pd.read_csv("sub_loss.csv")

print(df["Label"].value_counts(normalize=True))
print(df["Label"].value_counts())