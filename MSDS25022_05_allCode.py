"""
MSDS25022_05_allCode.py
========================
All tasks in one self-contained file.
Run individual tasks via the RUN_TASK flag at the bottom.

Assignment : DL Spring 2026 — Assignment 5 (SimCLR)
Student    : Muhammad Abdul Aleem
Roll No    : MSDS25022
"""

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════
import os, sys, json, csv, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.manifold  import TSNE
from sklearn.metrics   import confusion_matrix, ConfusionMatrixDisplay

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
SEED         = 2026
DATA_ROOT    = "./data"
SPLITS_DIR   = "./splits"
GRAPHS_DIR   = "./graphs"
RESULTS_DIR  = "./results"
MODELS_DIR   = "./models"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR_MEAN   = (0.4914, 0.4822, 0.4465)
CIFAR_STD    = (0.2470, 0.2435, 0.2616)
CLASS_NAMES  = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]
CLASS_COLORS = ["#E53935","#1E88E5","#43A047","#FB8C00","#8E24AA",
                "#00ACC1","#F4511E","#6D4C41","#546E7A","#FFB300"]

for d in [GRAPHS_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# UTILS — seed, splits, metrics
# ════════════════════════════════════════════════════════════════════════════
def lock_seeds(s=SEED):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _read_split(fname):
    with open(os.path.join(SPLITS_DIR, fname)) as fh:
        return [int(l.strip()) for l in fh if l.strip()]


def make_loader(split_file, is_train_set, transform, batch_size,
                shuffle=True, drop_last=False):
    ds   = torchvision.datasets.CIFAR10(DATA_ROOT, train=is_train_set,
                                         download=True, transform=transform)
    idx  = _read_split(split_file)
    sub  = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=drop_last)


plain_tfm = T.Compose([T.ToTensor(),
                        T.Normalize(CIFAR_MEAN, CIFAR_STD)])

sup_tfm   = T.Compose([T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(0.5),
                        T.ToTensor(),
                        T.Normalize(CIFAR_MEAN, CIFAR_STD)])

simclr_tfm = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class TwoViewTransform:
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x):   return self.tfm(x), self.tfm(x)


def accuracy(model, loader):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for x, y in loader:
            ok  += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
            tot += y.size(0)
    return ok / tot


def save_json(d, path):
    with open(path, "w") as fh: json.dump(d, fh, indent=2)
    print(f"saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ════════════════════════════════════════════════════════════════════════════
class CIFAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bb          = torchvision.models.resnet18(weights=None)
        bb.conv1    = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        bb.maxpool  = nn.Identity()
        self.feat_dim = bb.fc.in_features
        bb.fc       = nn.Identity()
        self.bb     = bb
    def forward(self, x): return self.bb(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_d=512, h=256, out_d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), nn.ReLU(True),
                                  nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = CIFAREncoder()
        self.proj_head = ProjectionHead(self.encoder.feat_dim)
    def forward(self, x):
        h = self.encoder(x); return h, self.proj_head(h)


class LinearHead(nn.Module):
    def __init__(self, in_d=512, nc=10):
        super().__init__()
        self.fc = nn.Linear(in_d, nc)
    def forward(self, x): return self.fc(x)


class FullNet(nn.Module):
    """Encoder + linear head, both trainable."""
    def __init__(self, enc): super().__init__(); self.enc = enc; self.head = LinearHead(enc.feat_dim)
    def forward(self, x): return self.head(self.enc(x))


class FrozenNet(nn.Module):
    """Frozen encoder + trainable linear head."""
    def __init__(self, enc):
        super().__init__()
        self.enc  = enc
        self.head = LinearHead(enc.feat_dim)
        for p in self.enc.parameters(): p.requires_grad_(False)
    def forward(self, x):
        with torch.no_grad(): h = self.enc(x)
        return self.head(h)


# ════════════════════════════════════════════════════════════════════════════
# NT-XENT LOSS
# ════════════════════════════════════════════════════════════════════════════
class NTXentLoss(nn.Module):
    def __init__(self, tau=0.5): super().__init__(); self.tau = tau
    def forward(self, z):
        two_n  = z.size(0); n = two_n // 2
        z_norm = F.normalize(z, dim=1)
        logits = (z_norm @ z_norm.T) / self.tau
        logits.masked_fill_(torch.eye(two_n, dtype=torch.bool, device=z.device), -1e9)
        targets = torch.cat([torch.arange(n, two_n, device=z.device),
                              torch.arange(0, n,     device=z.device)])
        return F.cross_entropy(logits, targets)


# ════════════════════════════════════════════════════════════════════════════
# SIMILARITY HELPERS
# ════════════════════════════════════════════════════════════════════════════
def cosine_sim_matrix(z):
    zn = F.normalize(z, dim=1)
    return zn @ zn.T


def avg_pos_neg(sim, n):
    pos = [(sim[i, i+n].item() + sim[i+n, i].item()) / 2 for i in range(n)]
    mask = torch.ones(2*n, 2*n, dtype=torch.bool)
    mask.fill_diagonal_(False)
    for i in range(n): mask[i, i+n] = mask[i+n, i] = False
    neg = sim[mask].tolist()
    return float(np.mean(pos)), float(np.mean(neg))


# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — SUPERVISED BASELINE
# ════════════════════════════════════════════════════════════════════════════
def task1_supervised():
    lock_seeds()
    print("\n── Task 1: Supervised Baseline ──")

    tr_loader  = make_loader("train_labeled_10percent.txt", True,  sup_tfm,  64)
    val_loader = make_loader("val.txt",                     True,  plain_tfm, 128, False)
    te_loader  = make_loader("test.txt",                    False, plain_tfm, 128, False)

    enc   = CIFAREncoder().to(DEVICE)
    model = FullNet(enc).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    tr_losses, val_losses = [], []
    best_val, best_sd = 1e9, None

    for ep in range(1, 31):
        model.train(); tl = 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y); opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * x.size(0)
        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                vl += crit(model(x.to(DEVICE)), y.to(DEVICE)).item() * x.size(0)
        tl /= len(tr_loader.dataset); vl /= len(val_loader.dataset)
        tr_losses.append(tl); val_losses.append(vl)
        sch.step()
        if vl < best_val: best_val = vl; best_sd = {k:v.clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0: print(f"  ep {ep:2d}  tr={tl:.4f}  val={vl:.4f}")

    model.load_state_dict(best_sd)
    test_acc = accuracy(model, te_loader)
    print(f"  Supervised test acc (10% labels): {test_acc*100:.2f}%")

    # Loss plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(tr_losses, label="Train", color="#2196F3", lw=2)
    ax.plot(val_losses, label="Val",  color="#FF5722", lw=2, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Supervised Baseline Loss")
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/supervised_loss.png", dpi=150); plt.close()

    # Confusion matrix
    model.eval(); preds, truths = [], []
    with torch.no_grad():
        for x, y in te_loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())
            truths.extend(y.tolist())
    cm = confusion_matrix(truths, preds)
    fig, ax = plt.subplots(figsize=(10,8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Supervised Baseline — Confusion Matrix"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/supervised_confusion_matrix.png", dpi=150); plt.close()

    torch.save(model.state_dict(), f"{MODELS_DIR}/supervised_best.pt")
    return test_acc


# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — AUGMENTATION VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
def task2_augmentations():
    lock_seeds()
    print("\n── Task 2: Augmentation Visualisation ──")
    raw_ds  = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=None)
    two_v   = TwoViewTransform(simclr_tfm)
    to_tens = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    inv_m   = torch.tensor(CIFAR_MEAN).view(3,1,1)
    inv_s   = torch.tensor(CIFAR_STD ).view(3,1,1)
    def show(t): return (t.cpu()*inv_s + inv_m).clamp(0,1).permute(1,2,0).numpy()

    rng  = np.random.default_rng(SEED)
    idxs = rng.choice(len(raw_ds), 10, replace=False)
    fig, axes = plt.subplots(10, 3, figsize=(7, 22))
    for col, ttl in enumerate(["Original","View A","View B"]):
        axes[0,col].set_title(ttl, fontsize=11, fontweight="bold")
    for row, idx in enumerate(idxs):
        pil, lbl = raw_ds[int(idx)]
        va, vb   = two_v(pil)
        for col, t in enumerate([to_tens(pil), va, vb]):
            axes[row,col].imshow(show(t)); axes[row,col].axis("off")
        axes[row,0].set_ylabel(CLASS_NAMES[lbl], fontsize=9, rotation=0, labelpad=42)
    plt.suptitle("SimCLR Augmentation Examples", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {RESULTS_DIR}/augmentation_examples.png")


# ════════════════════════════════════════════════════════════════════════════
# TASK 3 — SIMILARITY BEFORE TRAINING
# ════════════════════════════════════════════════════════════════════════════
def task3_similarity_before():
    lock_seeds()
    print("\n── Task 3: Similarity Before Training ──")
    N = 16
    raw_ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True,
                                           transform=TwoViewTransform(simclr_tfm))
    loader = DataLoader(raw_ds, batch_size=N, shuffle=True)
    (v1, v2), _ = next(iter(loader))

    enc = CIFAREncoder().to(DEVICE); enc.eval()
    with torch.no_grad():
        f1 = enc(v1.to(DEVICE)); f2 = enc(v2.to(DEVICE))
    z   = torch.cat([f1, f2])
    sim = cosine_sim_matrix(z)
    pos, neg = avg_pos_neg(sim, N)
    print(f"  Before training — same-image sim: {pos:.4f}  diff-image sim: {neg:.4f}")

    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(sim.cpu().numpy(), ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.axhline(N, color="white", lw=1.5, ls="--"); ax.axvline(N, color="white", lw=1.5, ls="--")
    ax.set_title("Similarity Matrix — Before Training"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/similarity_matrix_before_training.png", dpi=150); plt.close()
    return pos, neg


# ════════════════════════════════════════════════════════════════════════════
# TASK 4 — NT-XENT SANITY CHECK (model defined above)
# ════════════════════════════════════════════════════════════════════════════
def task4_sanity():
    lock_seeds()
    print("\n── Task 4: NT-Xent Sanity Check ──")
    z    = torch.randn(16, 128).to(DEVICE)
    loss = NTXentLoss()(z)
    print(f"  NT-Xent random input: {loss.item():.4f}  expected ≈ {math.log(15):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# TASK 5 — SIMCLR PRETRAINING
# ════════════════════════════════════════════════════════════════════════════
def task5_pretrain():
    lock_seeds()
    print("\n── Task 5: SimCLR Pretraining ──")

    ssl_loader = make_loader("train_ssl_unlabeled.txt", True,
                              TwoViewTransform(simclr_tfm), 64, drop_last=True)
    model  = SimCLRModel().to(DEVICE)
    crit   = NTXentLoss(tau=0.5)
    opt    = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []

    for ep in range(1, 51):
        model.train(); ep_loss = 0; nb = 0
        for (va, vb), _ in ssl_loader:
            imgs = torch.cat([va, vb]).to(DEVICE)
            _, z = model(imgs); loss = crit(z)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        avg = ep_loss / nb; losses.append(avg)
        if ep % 10 == 0 or ep == 1: print(f"  ep {ep:3d}/50  loss={avg:.4f}")

    torch.save(model.encoder.state_dict(), f"{MODELS_DIR}/simclr_encoder.pt")
    torch.save(model.state_dict(),         f"{MODELS_DIR}/simclr_full_model.pt")

    # Loss plot
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(range(1,51), losses, color="#9C27B0", lw=2, marker="o", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pretraining Loss"); ax.grid(alpha=.35); plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/simclr_pretraining_loss.png", dpi=150); plt.close()

    # Similarity after training
    N = 16
    raw_ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True,
                                           transform=TwoViewTransform(simclr_tfm))
    loader = DataLoader(raw_ds, batch_size=N, shuffle=False)
    (v1, v2), _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        _, z1 = model(v1.to(DEVICE)); _, z2 = model(v2.to(DEVICE))
    z_all = torch.cat([z1, z2])
    sim   = cosine_sim_matrix(z_all)
    pos, neg = avg_pos_neg(sim, N)
    print(f"  After training  — same-image sim: {pos:.4f}  diff-image sim: {neg:.4f}")

    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(sim.cpu().numpy(), ax=ax, cmap="RdYlBu_r", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.axhline(N, color="black", lw=1.2, ls="--", alpha=.6)
    ax.axvline(N, color="black", lw=1.2, ls="--", alpha=.6)
    ax.set_title("Similarity Matrix — After SimCLR Training"); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/similarity_matrix_after_training.png", dpi=150); plt.close()
    return pos, neg


# ════════════════════════════════════════════════════════════════════════════
# TASK 6 — LINEAR PROBE
# ════════════════════════════════════════════════════════════════════════════
def _run_probe(enc, label):
    lock_seeds()
    model = FrozenNet(enc).to(DEVICE)
    tr_l  = make_loader("train_labeled_10percent.txt", True,  sup_tfm,  64)
    val_l = make_loader("val.txt",                     True,  plain_tfm, 128, False)
    te_l  = make_loader("test.txt",                    False, plain_tfm, 128, False)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.head.parameters(), lr=3e-4)
    best_val, best_sd, val_hist = 0, None, []
    for ep in range(1, 21):
        model.train()
        for x, y in tr_l:
            x,y = x.to(DEVICE), y.to(DEVICE)
            l = crit(model(x), y); opt.zero_grad(); l.backward(); opt.step()
        va = accuracy(model, val_l); val_hist.append(va)
        if va > best_val: best_val = va; best_sd = {k:v.clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0: print(f"    [{label}] ep {ep:2d}  val={va*100:.2f}%")
    model.load_state_dict(best_sd)
    te_acc = accuracy(model, te_l)
    print(f"  [{label}] test acc: {te_acc*100:.2f}%")
    return te_acc, val_hist, model


def task6_linear_probe():
    print("\n── Task 6: Linear Probe ──")
    rand_enc   = CIFAREncoder().to(DEVICE)
    acc_rand, hist_rand, _ = _run_probe(rand_enc, "Random Enc")

    simclr_enc = CIFAREncoder().to(DEVICE)
    simclr_enc.load_state_dict(torch.load(f"{MODELS_DIR}/simclr_encoder.pt", map_location=DEVICE))
    acc_ssl, hist_ssl, probe_model = _run_probe(simclr_enc, "SimCLR Enc")
    torch.save(probe_model.state_dict(), f"{MODELS_DIR}/linear_probe.pt")

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot([v*100 for v in hist_rand], label="Random Encoder",  color="#607D8B", lw=2, ls="--")
    ax.plot([v*100 for v in hist_ssl],  label="SimCLR Encoder",  color="#4CAF50", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Linear Probe: Random vs SimCLR Encoder"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{GRAPHS_DIR}/linear_probe_accuracy.png", dpi=150); plt.close()
    return acc_rand, acc_ssl


# ════════════════════════════════════════════════════════════════════════════
# TASK 7 — FINE-TUNING
# ════════════════════════════════════════════════════════════════════════════
def task7_finetune():
    print("\n── Task 7: Fine-tuning ──")
    lock_seeds()
    enc   = CIFAREncoder().to(DEVICE)
    enc.load_state_dict(torch.load(f"{MODELS_DIR}/simclr_encoder.pt", map_location=DEVICE))
    model = FullNet(enc).to(DEVICE)
    tr_l  = make_loader("train_labeled_10percent.txt", True,  sup_tfm,  64)
    val_l = make_loader("val.txt",                     True,  plain_tfm, 128, False)
    te_l  = make_loader("test.txt",                    False, plain_tfm, 128, False)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val, best_sd, val_hist = 0, None, []
    for ep in range(1, 21):
        model.train()
        for x, y in tr_l:
            x,y = x.to(DEVICE), y.to(DEVICE)
            l = crit(model(x), y); opt.zero_grad(); l.backward(); opt.step()
        sch.step()
        va = accuracy(model, val_l); val_hist.append(va)
        if va > best_val: best_val = va; best_sd = {k:v.clone() for k,v in model.state_dict().items()}
        if ep % 5 == 0: print(f"  ep {ep:2d}  val={va*100:.2f}%")
    model.load_state_dict(best_sd)
    te_acc = accuracy(model, te_l)
    print(f"  Fine-tune test acc: {te_acc*100:.2f}%")
    torch.save(model.state_dict(), f"{MODELS_DIR}/finetuned_model.pt")

    # save test_predictions.csv
    model.eval(); rows = []
    with torch.no_grad():
        for i, (x, y) in enumerate(te_l):
            probs = F.softmax(model(x.to(DEVICE)), 1).cpu().numpy()
            preds = probs.argmax(1)
            for k in range(len(y)):
                rows.append([i*te_l.batch_size+k, y[k].item(), preds[k]] + list(probs[k]))
    with open(f"{RESULTS_DIR}/test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_index","true_label","predicted_label"]+[f"prob_class_{c}" for c in range(10)])
        w.writerows(rows)
    return te_acc, val_hist


# ════════════════════════════════════════════════════════════════════════════
# TASK 8 — t-SNE VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
def task8_tsne():
    print("\n── Task 8: t-SNE Visualisation ──")
    lock_seeds()

    base_ds  = torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
                                             download=True, transform=plain_tfm)
    val_idx  = _read_split("val.txt")
    rng      = np.random.default_rng(SEED)
    chosen   = rng.choice(val_idx, 1000, replace=False).tolist()
    sub      = Subset(base_ds, chosen)
    loader   = DataLoader(sub, batch_size=128, shuffle=False, num_workers=2)

    @torch.no_grad()
    def get_feats(enc):
        enc.eval(); fs, ls = [], []
        for x, y in loader:
            fs.append(enc(x.to(DEVICE)).cpu().numpy()); ls.append(y.numpy())
        return np.concatenate(fs), np.concatenate(ls)

    def plot_tsne(feats, labels, title, path):
        coords = TSNE(2, perplexity=40, n_iter=1000, random_state=SEED,
                      learning_rate="auto", init="pca").fit_transform(feats)
        fig, ax = plt.subplots(figsize=(9,7))
        for c in range(10):
            m = labels == c
            ax.scatter(coords[m,0], coords[m,1], c=CLASS_COLORS[c], s=8, alpha=.7, linewidths=0)
        ax.legend(handles=[mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(10)],
                  loc="upper right", fontsize=8)
        ax.set_title(title, fontsize=12); ax.axis("off"); plt.tight_layout()
        plt.savefig(path, dpi=150); plt.close(); print(f"  saved → {path}")

    # Random encoder
    enc_r = CIFAREncoder().to(DEVICE)
    f, l  = get_feats(enc_r)
    plot_tsne(f, l, "t-SNE — Random Encoder", f"{RESULTS_DIR}/random_encoder_pca_or_tsne.png")

    # SimCLR encoder
    enc_s = CIFAREncoder().to(DEVICE)
    enc_s.load_state_dict(torch.load(f"{MODELS_DIR}/simclr_encoder.pt", map_location=DEVICE))
    f, l  = get_feats(enc_s)
    plot_tsne(f, l, "t-SNE — SimCLR Pretrained Encoder", f"{RESULTS_DIR}/simclr_encoder_pca_or_tsne.png")

    # Fine-tuned encoder
    enc_f  = CIFAREncoder().to(DEVICE)
    ft_net = FullNet(enc_f).to(DEVICE)
    ft_net.load_state_dict(torch.load(f"{MODELS_DIR}/finetuned_model.pt", map_location=DEVICE))
    f, l   = get_feats(ft_net.enc)
    plot_tsne(f, l, "t-SNE — Fine-Tuned Encoder", f"{RESULTS_DIR}/finetuned_encoder_pca_or_tsne.png")


# ════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all",
                        help="Which task to run: 1,2,3,4,5,6,7,8 or all")
    args = parser.parse_args()

    task_map = {
        "1": task1_supervised,
        "2": task2_augmentations,
        "3": task3_similarity_before,
        "4": task4_sanity,
        "5": task5_pretrain,
        "6": task6_linear_probe,
        "7": task7_finetune,
        "8": task8_tsne,
    }

    if args.task == "all":
        sup_acc            = task1_supervised()
        task2_augmentations()
        pos_b, neg_b       = task3_similarity_before()
        task4_sanity()
        pos_a, neg_a       = task5_pretrain()
        acc_rand, acc_ssl  = task6_linear_probe()
        acc_ft, _          = task7_finetune()
        task8_tsne()

        metrics = {
            "student_name": "Muhammad Abdul Aleem", "roll_number": "MSDS25022",
            "seed": SEED, "batch_size": 64, "simclr_epochs": 50,
            "linear_probe_epochs": 20, "finetuning_epochs": 20,
            "learning_rate": 3e-4, "temperature": 0.5, "optimizer": "Adam",
            "supervised_10percent_test_acc":  round(sup_acc,  4),
            "random_linear_probe_test_acc":   round(acc_rand, 4),
            "simclr_linear_probe_test_acc":   round(acc_ssl,  4),
            "simclr_finetune_test_acc":        round(acc_ft,   4),
            "same_view_similarity_before":     round(pos_b,   4),
            "different_image_similarity_before": round(neg_b, 4),
            "same_view_similarity_after":      round(pos_a,   4),
            "different_image_similarity_after":  round(neg_a, 4),
        }
        save_json(metrics, f"{RESULTS_DIR}/metrics.json")
        print("\nAll tasks done.")
    elif args.task in task_map:
        task_map[args.task]()
    else:
        print(f"Unknown task: {args.task}")
