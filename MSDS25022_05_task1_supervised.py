"""
Task 1 — Supervised Baseline with Limited Labels
=================================================
Trains a ResNet-18 from scratch on the fixed 10 % labeled split.
This gives us the floor to beat with self-supervised learning.

Expected outputs:
  graphs/supervised_loss.png
  results/supervised_confusion_matrix.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simclr_utils import (
    lock_seeds, SEED,
    load_labeled_train, load_val, load_test,
    compute_accuracy, plot_confusion_matrix,
    save_metrics, CIFAR10_CLASSES,
)

# ── config ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = os.path.join(_HERE, "data")
GRAPHS_DIR     = os.path.join(_HERE, "graphs")
RESULTS_DIR    = os.path.join(_HERE, "results")
MODELS_DIR     = os.path.join(_HERE, "models")

NUM_EPOCHS     = 30
BATCH_SIZE     = 64
LEARNING_RATE  = 3e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ── model ──────────────────────────────────────────────────────────────────────
def build_supervised_net(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 modified for 32x32 CIFAR images:
      - conv1 changed to 3x3, stride 1, padding 1
      - maxpool replaced with Identity
      - final fc → Linear(512, num_classes)
    """
    net = torchvision.models.resnet18(weights=None)
    # CIFAR images are small; the original 7x7 conv + maxpool destroys too much detail
    net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc      = nn.Linear(net.fc.in_features, num_classes)
    return net


# ── training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimiser, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def plot_loss_curves(train_losses, val_losses, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs  = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss",      color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses,   label="Validation Loss", color="#FF5722",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Supervised Baseline — Training and Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved loss curves → {save_path}")


def main() -> None:
    lock_seeds(SEED)
    print(f"Device: {DEVICE}")

    # ── data loaders ──────────────────────────────────────────────────────────
    # Task 2 augmentation transform for training; plain for val/test
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std =(0.2470, 0.2435, 0.2616)),
    ])
    train_loader = load_labeled_train(DATA_ROOT, transform=train_transform,
                                      batch_size=BATCH_SIZE)
    val_loader   = load_val(DATA_ROOT,  batch_size=128)
    test_loader  = load_test(DATA_ROOT, batch_size=128)

    print(f"Train batches : {len(train_loader)}  "
          f"(~{len(train_loader.dataset)} images)")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # ── model + training setup ────────────────────────────────────────────────
    model     = build_supervised_net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_val_epoch = 0

    print("\nStarting supervised training…")
    for ep in range(1, NUM_EPOCHS + 1):
        tr_loss  = run_epoch(model, train_loader, criterion, optimiser, DEVICE, True)
        val_loss = run_epoch(model, val_loader,   criterion, None,      DEVICE, False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        val_acc = compute_accuracy(model, val_loader, DEVICE)
        print(f"  Epoch {ep:3d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}%")

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_epoch = ep
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, "supervised_best.pt"))

    print(f"\nBest checkpoint at epoch {best_val_epoch} "
          f"(val_loss={best_val_loss:.4f})")

    # ── evaluate on test set ──────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "supervised_best.pt"),
                   map_location=DEVICE)
    )
    test_acc = compute_accuracy(model, test_loader, DEVICE)
    print(f"\nTest accuracy (supervised, 10% labels): {test_acc*100:.2f}%")

    # ── save outputs ──────────────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(GRAPHS_DIR, "supervised_loss.png"))
    plot_confusion_matrix(model, test_loader, DEVICE,
                          os.path.join(RESULTS_DIR, "supervised_confusion_matrix.png"))

    # partial metrics save so we can fill in the rest later
    partial = {
        "student_name"                : "Muhammad Abdul Aleem",
        "roll_number"                 : "MSDS25022",
        "seed"                        : SEED,
        "batch_size"                  : BATCH_SIZE,
        "supervised_epochs"           : NUM_EPOCHS,
        "learning_rate"               : LEARNING_RATE,
        "optimizer"                   : "Adam",
        "supervised_10percent_test_acc": round(test_acc, 4),
    }
    save_metrics(partial, os.path.join(RESULTS_DIR, "metrics_partial.json"))

    print("\nTask 1 complete.")


if __name__ == "__main__":
    main()
