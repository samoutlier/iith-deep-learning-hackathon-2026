""" 
## Cell 1 — Imports
"""

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR   # NEW: cosine LR decay

from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split

print("Imports done.")
print("PyTorch version:", torch.__version__)

"""## Cell 2 — Config"""

class CFG:
    data_dir   = "/kaggle/input/competitions/iith-deep-learning-2026-hackathon"
    img_size   = 96
    batch_size = 64
    epochs     = 50                        # slightly more to let cosine LR finish cleanly
    lr         = 3e-4
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    seed       = 2025

    milestone_epochs = [30, 40, 50]

print("Device:", CFG.device)
print("Epochs:", CFG.epochs)
print("Milestone saves at epochs:", CFG.milestone_epochs)

"""## Cell 3 — Seed"""

def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(CFG.seed)
print("Seed set to", CFG.seed)

"""## Cell 4 — Dataset"""

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

print("ImageDataset class ready.")

"""## Cell 5 — Transforms"""

# CHANGE 1: Added Normalize to both train and val transforms. 
# CHANGE 2: RandomGrayscale p=0.5 → p=0.15 

MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(CFG.img_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.15),          # was 0.5 
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),    # NEW
])

val_tfms = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),    # NEW  
])

print("Train transform: RandomResizedCrop + flips + ColorJitter + Blur + Normalize")
print("Val transform  : Resize + Normalize")

"""## Cell 6 — Load and Split"""

train_dir = os.path.join(CFG.data_dir, "train", "train")

paths, labels = [], []

for label in ["0", "1"]:
    folder = os.path.join(train_dir, label)
    for img in os.listdir(folder):
        paths.append(os.path.join(folder, img))
        labels.append(int(label))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

print(f"Train samples : {len(train_paths)}")
print(f"Val   samples : {len(val_paths)}")
print(f"Class 0 train : {train_labels.count(0)}   Class 1 train : {train_labels.count(1)}")

"""## Cell 7 — Loaders"""

train_ds = ImageDataset(train_paths, train_labels, train_tfms)
val_ds   = ImageDataset(val_paths,   val_labels,   val_tfms)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")

"""## Cell 8 — Model (ResNet + SE attention)"""

# CHANGE: Added SEBlock inside each residual block.

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise attention with ~0 extra params cost."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # squeeze: (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),                   # → (B, C)
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()                    # outputs per-channel weights in [0, 1]
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class Block(nn.Module):
    """Residual block with SE attention — same structure as your original Block."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.se   = SEBlock(out_c)          # NEW: channel attention after conv
        self.skip = (nn.Conv2d(in_c, out_c, 1, stride, bias=False)
                     if in_c != out_c or stride != 1 else nn.Identity())

    def forward(self, x):
        return torch.relu(self.se(self.conv(x)) + self.skip(x))


class Net(nn.Module):
    """Same topology as your Net — stem → 3 blocks → pool → fc(1)."""
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = Block(32,  64,  stride=2)
        self.layer2 = Block(64,  128, stride=2)
        self.layer3 = Block(128, 192, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(
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

_m = Net()
_p = sum(p.numel() for p in _m.parameters())
_y = _m(torch.zeros(2, 3, 96, 96))
print(f"Parameters : {_p:,}")
print(f"Output     : {_y.shape}  (expected [2, 1])")
del _m, _y

"""## Cell 9 — Train Setup"""

class SmoothBCE(nn.Module):
    """Binary cross-entropy with label smoothing — same as your original."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), targets)


model     = Net().to(CFG.device)
criterion = SmoothBCE(smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

# CHANGE: CosineAnnealingLR — LR starts at 3e-4 and smoothly decays to 0
scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)

print("Model      :", model.__class__.__name__)
print("Optimizer  : Adam  lr=", CFG.lr)
print("Scheduler  : CosineAnnealingLR  T_max=", CFG.epochs, " eta_min=1e-6")
print("Loss       : SmoothBCE(smoothing=0.05)")

"""## Cell 10 — Training Loop"""

# Checkpoint saves:
#   best_acc.pt        — best validation accuracy seen at any epoch
#   best_loss.pt       — best validation loss seen at any epoch
#   ckpt_ep30.pt       — snapshot at epoch 30
#   ckpt_ep40.pt       — snapshot at epoch 40
#   ckpt_ep50.pt       — snapshot at epoch 50  (= last)
#
# These 5 checkpoints are all genuinely different snapshots spread across
# the training curve, so ensembling them gives diverse, uncorrelated votes.

best_acc_val  = 0.0
best_loss_val = float("inf")

for epoch in range(1, CFG.epochs + 1):

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ── Validate ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss  = 0.0
    correct   = 0
    total     = 0
    class1_n  = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y  = x.to(CFG.device), y.to(CFG.device)
            out   = model(x)
            loss  = criterion(out, y)
            val_loss += loss.item()
            prob  = torch.sigmoid(out).squeeze(1)
            pred  = (prob > 0.5).long()
            correct  += (pred == y).sum().item()
            total    += y.size(0)
            class1_n += pred.sum().item()

    acc        = correct / total
    class1_r   = class1_n / total
    avg_vloss  = val_loss / len(val_loader)
    cur_lr     = optimizer.param_groups[0]["lr"]

    # ── Scheduler step ───────────────────────────────────────────────────────
    scheduler.step()   # CHANGE: step cosine LR every epoch

    # ── Save best_acc ────────────────────────────────────────────────────────
    if acc > best_acc_val:
        best_acc_val = acc
        torch.save(model.state_dict(), "best_acc.pt")

    # ── Save best_loss ───────────────────────────────────────────────────────
    if avg_vloss < best_loss_val:
        best_loss_val = avg_vloss
        torch.save(model.state_dict(), "best_loss.pt")

    # ── Save milestone checkpoints ───────────────────────────────────────────
    # CHANGE: save at epochs 30, 40, 50 so we have genuinely diverse snapshots
    if epoch in CFG.milestone_epochs:
        ckpt_name = f"ckpt_ep{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"  => Milestone checkpoint saved: {ckpt_name}")

    print(f"Epoch {epoch:2d}/{CFG.epochs} | "
          f"Acc {acc:.4f} | "
          f"ValLoss {avg_vloss:.4f} | "
          f"Class1 {class1_r:.2f} | "
          f"LR {cur_lr:.2e}")

print(f"\nTraining complete.")
print(f"Best val acc  : {best_acc_val:.4f}  → best_acc.pt")
print(f"Best val loss : {best_loss_val:.4f}  → best_loss.pt")

"""## Cell 11 — Evaluate Each Checkpoint on Validation"""

def evaluate_checkpoint(model, path, val_loader):
    """Load checkpoint and report val accuracy + class-1 ratio."""
    if not os.path.exists(path):
        print(f"{path} → NOT FOUND (skipping)")
        return
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model.eval()

    preds, gt = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x   = x.to(CFG.device)
            out = torch.sigmoid(model(x)).cpu().numpy().flatten()
            preds.extend(out)
            gt.extend(y.numpy())

    preds_bin   = [1 if p > 0.5 else 0 for p in preds]
    acc         = np.mean(np.array(preds_bin) == np.array(gt))
    class1_ratio = np.mean(preds_bin)
    print(f"{path:18s}  →  Acc: {acc:.4f}  |  Class1: {class1_ratio:.2f}")


# Evaluate all 5 checkpoints
for ckpt in ["best_acc.pt", "best_loss.pt",
             "ckpt_ep30.pt", "ckpt_ep40.pt", "ckpt_ep50.pt"]:
    evaluate_checkpoint(model, ckpt, val_loader)

"""## Cell 12 — TTA Inference (8 variants per checkpoint)"""

# CHANGE: TTA expanded from 2 (original + hflip) to 8 deterministic variants.

test_dir  = os.path.join(CFG.data_dir, "test", "test")
test_imgs = sorted(os.listdir(test_dir))

s  = CFG.img_size
sp = int(s * 1.12)   # slightly larger for center-crop TTA

norm = transforms.Normalize(mean=MEAN, std=STD)

TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((s, s)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((s, s)), transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((s, s)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((sp, sp)), transforms.CenterCrop(s),
                        transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((sp, sp)), transforms.CenterCrop(s),
                        transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((s, s)), transforms.RandomRotation((90, 90)),
                        transforms.ToTensor(), norm]),
    transforms.Compose([transforms.Resize((s, s)), transforms.RandomRotation((270, 270)),
                        transforms.ToTensor(), norm]),
]

print(f"TTA variants : {len(TTA_TRANSFORMS)}")
print(f"Test images  : {len(test_imgs)}")


class TestDataset(Dataset):
    """Simple dataset for test images (no labels)."""
    def __init__(self, img_dir, filenames, transform):
        self.img_dir   = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img  = Image.open(path).convert("RGB")
        return self.transform(img)


def predict_with_tta(model, ckpt_path):
    """
    Load checkpoint, run all 8 TTA passes, return averaged class-1 probability
    array of shape (N_test,).
    """
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} not found")
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
    model.eval()

    sum_probs = np.zeros(len(test_imgs), dtype=np.float64)

    for i, tfm in enumerate(TTA_TRANSFORMS):
        ds  = TestDataset(test_dir, test_imgs, tfm)
        ldr = DataLoader(ds, batch_size=CFG.batch_size * 2, shuffle=False,
                         num_workers=4, pin_memory=True)

        batch_probs = []
        with torch.no_grad():
            for x in ldr:
                x = x.to(CFG.device)
                prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
                batch_probs.append(prob)

        sum_probs += np.concatenate(batch_probs)

    avg_probs = sum_probs / len(TTA_TRANSFORMS)
    print(f"  {ckpt_path:18s}  →  Class1 ratio: {(avg_probs > 0.5).mean():.2f}")
    return avg_probs


print("\nRunning TTA inference on all 5 checkpoints...")

"""## Cell 13 — Collect Predictions per Checkpoint"""

p_best_acc  = predict_with_tta(model, "best_acc.pt")
p_best_loss = predict_with_tta(model, "best_loss.pt")
p_ep30      = predict_with_tta(model, "ckpt_ep30.pt")
p_ep40      = predict_with_tta(model, "ckpt_ep40.pt")
p_ep50      = predict_with_tta(model, "ckpt_ep50.pt")

"""## Cell 14 — Save Individual & Ensemble Submissions"""

def save_preds(probs, name):
    """Threshold at 0.5, save CSV, print class distribution."""
    if probs is None:
        print(f"{name}: skipped (checkpoint not found)")
        return
    preds = (probs > 0.5).astype(int)
    ratio = preds.mean()
    print(f"{name:30s}  Class1%: {ratio:.2f}  "
          f"(0:{(preds==0).sum()}  1:{(preds==1).sum()})")
    df = pd.DataFrame({"ID": test_imgs, "Label": preds})
    df.to_csv(name, index=False)


# Individual checkpoint submissions
save_preds(p_best_acc,  "sub_best_acc.csv")
save_preds(p_best_loss, "sub_best_loss.csv")
save_preds(p_ep30,      "sub_ep30.csv")
save_preds(p_ep40,      "sub_ep40.csv")
save_preds(p_ep50,      "sub_ep50.csv")


available = [p for p in [p_best_acc, p_best_loss, p_ep30, p_ep40, p_ep50]
             if p is not None]
ensemble  = np.mean(available, axis=0)
save_preds(ensemble, "sub_ensemble.csv")

print(f"\nEnsemble used {len(available)} checkpoints × 8 TTA = {len(available)*8} total votes per image")

"""## Cell 15 — Preview Final Ensemble Submission"""

df = pd.read_csv("sub_ensemble.csv")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["Label"].value_counts(normalize=True).round(3).to_string())
print()
print(df["Label"].value_counts().to_string())
print()
df.head(10)

"""## Cell 16 — `generate_predictions(data_dir)` for .py Submission"""

# This function is fully self-contained.
# Save it as solution.py and submit to MS Teams.
# Run: python solution.py  then type the data_dir path when prompted.

def generate_predictions(data_dir):
    import os, random
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    #  Config 
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE   = 96
    BATCH_SIZE = 128
    CKPT_DIR   = "/kaggle/working"
    OUTPUT_CSV = "/kaggle/working/predictions-ver18-bestacc.csv"
    MEAN       = [0.5, 0.5, 0.5]
    STD        = [0.5, 0.5, 0.5]
    EXTS       = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


    test_dir = os.path.join(data_dir )
    test_imgs = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(EXTS)])


    print(f"Using test_dir: {test_dir}")
    print(f"Exists: {os.path.isdir(test_dir)}")

    class _SE(nn.Module):
        def __init__(self, ch):
            super().__init__()
            mid = max(ch // 8, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(ch, mid, bias=False), nn.ReLU(),
                nn.Linear(mid, ch, bias=False), nn.Sigmoid())
        def forward(self, x):
            return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

    class _Block(nn.Module):
        def __init__(self, ic, oc, stride=1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(),
                nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc))
            self.se   = _SE(oc)
            self.skip = (nn.Conv2d(ic, oc, 1, stride, bias=False)
                         if ic != oc or stride != 1 else nn.Identity())
        def forward(self, x):
            return torch.relu(self.se(self.conv(x)) + self.skip(x))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem   = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(32), nn.ReLU())
            self.layer1 = _Block(32,  64,  2)
            self.layer2 = _Block(64,  128, 2)
            self.layer3 = _Block(128, 192, 2)
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.fc     = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, 1))
        def forward(self, x):
            x = self.stem(x)
            return self.fc(self.pool(self.layer3(self.layer2(self.layer1(x)))).flatten(1))

    #  TTA transforms 
    s, sp = IMG_SIZE, int(IMG_SIZE * 1.12)
    norm  = transforms.Normalize(MEAN, STD)
    tta_tfms = [
        transforms.Compose([transforms.Resize((s,s)), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((s,s)), transforms.RandomHorizontalFlip(p=0.79), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((s,s)), transforms.RandomVerticalFlip(p=0.85), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((sp,sp)), transforms.CenterCrop(s), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((s,s)), transforms.RandomRotation((90,90)), transforms.ToTensor(), norm]),
    ]

    class _DS(Dataset):
        def __init__(self, d, fs, t): self.d, self.fs, self.t = d, fs, t
        def __len__(self): return len(self.fs)
        def __getitem__(self, i):
            return self.t(Image.open(os.path.join(self.d, self.fs[i])).convert("RGB"))

    # ── Ensemble TTA inference ────────────────────────────────────────────────
    ckpt_names = ["best_acc.pt", "best_loss.pt",
                  "ckpt_ep30.pt", "ckpt_ep40.pt", "ckpt_ep50.pt"]
    # ckpt_names = [  "best_acc.pt"]

    sum_probs = np.zeros(len(test_imgs), dtype=np.float64)
    n_votes   = 0

    for ck in ckpt_names:
        ckpt_path = os.path.join(CKPT_DIR, ck)
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path}")
            continue

        model = _Net().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        print(f"  Loaded {ck}")

        for ti, tfm in enumerate(tta_tfms):
            ds  = _DS(test_dir, test_imgs, tfm)
            ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
            bp  = []
            with torch.no_grad():
                for x in ldr:
                    bp.append(torch.sigmoid(model(x.to(DEVICE))).squeeze(1).cpu().numpy())
            sum_probs += np.concatenate(bp)
            n_votes   += 1
            print(f"    TTA {ti+1}/{len(tta_tfms)}")

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    assert n_votes > 0, "No checkpoints found!"
    preds = (sum_probs / n_votes > 0.5).astype(int)
    print(f"Votes: {n_votes}  |  Class0: {(preds==0).sum()}  Class1: {(preds==1).sum()}")

    pd.DataFrame({"ID": test_imgs, "Label": preds}).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")
    return OUTPUT_CSV


if __name__ == "__main__":
    test_dir = os.path.join(data_dir)
    generate_predictions(data_dir)