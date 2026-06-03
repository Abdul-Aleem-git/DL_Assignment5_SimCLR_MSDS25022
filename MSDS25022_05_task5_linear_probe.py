"""
Task 6 — Linear Probe Evaluation
===================================
Freezes the encoder and trains only a linear head.

Experiment A: Random (untrained) encoder — frozen
Experiment B: SimCLR pretrained encoder  — frozen

Both use the 10 % labeled split for training
and the test split for final accuracy.

Expected outputs:
  graphs/linear_probe_accuracy.png
  models/linear_probe.pt   (SimCLR probe — best val checkpoint)
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simclr_utils import (
    lock_seeds, SEED,
    load_labeled_train, load_val, load_test,
    compute_accuracy,
)
from MSDS25022_05_task4_simclr import CIFAREncoder, ClassificationHead

DATA_ROOT   = "./data"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"

NUM_EPOCHS  = 20   # same as linear probe for fair comparison   # 20 epochs is enough, val acc plateaus around epoch 15
BATCH_SIZE  = 64
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(GRAPHS_DIR, exist_ok=True)


# ── full model: frozen encoder + trainable linear head ────────────────────────
class FrozenEncoderClassifier(nn.Module):
    """
    Wraps a frozen encoder with a trainable linear classification head.
    Only the head parameters are updated during training.
    """
    def __init__(self, encoder: CIFAREncoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head    = ClassificationHead(encoder.feature_dim, num_classes)
        # Freeze encoder
        # freeze all encoder weights — only the linear head trains
    for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
        return self.head(h)


def run_linear_probe(encoder: CIFAREncoder,
                     label: str) -> tuple[float, list, list]:
    """
    Trains a linear probe on top of a frozen encoder for NUM_EPOCHS.
    Returns (test_accuracy, train_acc_history, val_acc_history).
    """
    lock_seeds(SEED)

    train_loader = load_labeled_train(DATA_ROOT, batch_size=BATCH_SIZE)
    val_loader   = load_val(DATA_ROOT,  batch_size=128)
    test_loader  = load_test(DATA_ROOT, batch_size=128)

    model     = FrozenEncoderClassifier(encoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Only optimise the head
    optimiser = torch.optim.Adam(model.head.parameters(), lr=LR)

    best_val_acc   = 0.0   # will update whenever val improves   # track best to restore later
    best_state     = None
    train_acc_hist = []
    val_acc_hist   = []

    print(f"\n  [{label}] Linear probe training ({NUM_EPOCHS} epochs)…")
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = criterion(model(imgs), lbls)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        tr_acc  = compute_accuracy(model, train_loader, DEVICE)
        val_acc = compute_accuracy(model, val_loader,   DEVICE)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            print(f"    Epoch {ep:3d}/{NUM_EPOCHS}  "
                  f"train_acc={tr_acc*100:.2f}%  val_acc={val_acc*100:.2f}%")

    # Restore best checkpoint
    model.load_state_dict(best_state)
    test_acc = compute_accuracy(model, test_loader, DEVICE)
    print(f"  [{label}] Test accuracy: {test_acc*100:.2f}%")
    return test_acc, train_acc_hist, val_acc_hist, model


def plot_probe_comparison(hist_a: list, hist_b: list, save_path: str) -> None:
    epochs = range(1, len(hist_a) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [v * 100 for v in hist_a], label="Random Encoder (val)",
            color="#607D8B", linewidth=2, linestyle="--")
    ax.plot(epochs, [v * 100 for v in hist_b], label="SimCLR Encoder (val)",
            color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Linear Probe Validation Accuracy: Random vs SimCLR Encoder")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved linear probe accuracy plot → {save_path}")


def main() -> None:
    lock_seeds(SEED)

    # ── Experiment A: random encoder ──────────────────────────────────────
    rand_encoder = CIFAREncoder().to(DEVICE)   # random weights
    acc_rand, tr_hist_a, val_hist_a, _ = run_linear_probe(
        rand_encoder, label="Random Encoder"
    )

    # ── Experiment B: SimCLR encoder ──────────────────────────────────────
    simclr_encoder = CIFAREncoder().to(DEVICE)
    enc_path       = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(
            f"SimCLR encoder not found at {enc_path}. "
            "Run task5_pretraining.py first."
        )
    simclr_encoder.load_state_dict(
        torch.load(enc_path, map_location=DEVICE)
    )
    acc_simclr, tr_hist_b, val_hist_b, probe_model = run_linear_probe(
        simclr_encoder, label="SimCLR Encoder"
    )

    # ── save probe model ──────────────────────────────────────────────────
    torch.save(probe_model.state_dict(),
               os.path.join(MODELS_DIR, "linear_probe.pt"))

    # ── plot ──────────────────────────────────────────────────────────────
    plot_probe_comparison(
        val_hist_a, val_hist_b,
        os.path.join(GRAPHS_DIR, "linear_probe_accuracy.png")
    )

    print("\nLinear Probe Summary:")
    print(f"  Random encoder + linear head  →  test acc: {acc_rand*100:.2f}%")
    print(f"  SimCLR encoder + linear head  →  test acc: {acc_simclr*100:.2f}%")
    print("\nTask 6 complete.")

    return acc_rand, acc_simclr


if __name__ == "__main__":
    main()
