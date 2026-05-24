"""
Task 5 — SimCLR Pretraining
=============================
Trains the SimCLR model on the full unlabeled split.
Labels are never seen during this phase.

Fixed settings (from Section 6 of the assignment):
  epochs          : 50
  batch size      : 64
  learning rate   : 3e-4
  temperature tau : 0.5
  optimizer       : Adam
  seed            : 2026

Expected outputs:
  graphs/simclr_pretraining_loss.png
  results/similarity_matrix_after_training.png
  models/simclr_encoder.pt
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simclr_utils import lock_seeds, SEED, load_unlabeled_train
from MSDS25022_05_task2_augmentations import TwoViewTransform, simclr_augment
from MSDS25022_05_task4_simclr import (
    SimCLRModel, NTXentLoss,
    build_cosine_sim_matrix, visualise_sim_matrix,
)
from MSDS25022_05_task3_similarity import compute_avg_similarities

# ── config ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "./data"
GRAPHS_DIR  = "./graphs"
RESULTS_DIR = "./results"
MODELS_DIR  = "./models"

NUM_EPOCHS  = 50
BATCH_SIZE  = 64
LR          = 3e-4
TAU         = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ── similarity measurement after training ─────────────────────────────────────
def measure_trained_similarity(model: SimCLRModel,
                                device: torch.device,
                                n_samples: int = 16) -> tuple[float, float]:
    """
    Loads a tiny batch and computes avg positive / negative cosine similarity
    using the PROJECTION HEAD outputs (same space as training).
    """
    raw_ds   = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True,
        transform=TwoViewTransform(simclr_augment)
    )
    loader   = torch.utils.data.DataLoader(raw_ds, batch_size=n_samples,
                                            shuffle=True)
    (v1, v2), _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        _, z1 = model(v1.to(device))
        _, z2 = model(v2.to(device))
    z_all = torch.cat([z1, z2], dim=0)
    sim   = build_cosine_sim_matrix(z_all)
    return compute_avg_similarities(sim, n=n_samples)


# ── loss curve plot ────────────────────────────────────────────────────────────
def plot_pretraining_loss(losses: list[float], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(losses) + 1), losses,
            color="#9C27B0", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pretraining Loss Curve")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved pretraining loss curve → {save_path}")


# ── main pretraining loop ──────────────────────────────────────────────────────
def pretrain_simclr() -> SimCLRModel:
    lock_seeds(SEED)
    print(f"Device: {DEVICE}")

    # Data: unlabeled split with TwoViewTransform
    ssl_loader = load_unlabeled_train(
        DATA_ROOT,
        transform=TwoViewTransform(simclr_augment),
        batch_size=BATCH_SIZE
    )
    print(f"Unlabeled batches per epoch: {len(ssl_loader)}"
          f"  (~{len(ssl_loader.dataset)} images)")

    # Model + loss + optimiser
    model     = SimCLRModel().to(DEVICE)
    loss_fn   = NTXentLoss(temperature=TAU)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []

    print(f"\nPretraining SimCLR for {NUM_EPOCHS} epochs…")
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        num_batches  = 0

        for (view_a, view_b), _ in ssl_loader:
            # Concatenate views: first N rows = view-A, next N = view-B
            imgs = torch.cat([view_a, view_b], dim=0).to(DEVICE)

            _, z      = model(imgs)          # [2N, 128]
            loss      = loss_fn(z)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            num_batches  += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)

        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}")

    # ── save encoder weights ───────────────────────────────────────────────
    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"\nSaved SimCLR encoder → {enc_path}")

    # ── save full model for downstream tasks ──────────────────────────────
    torch.save(model.state_dict(),
               os.path.join(MODELS_DIR, "simclr_full_model.pt"))

    # ── plots ──────────────────────────────────────────────────────────────
    plot_pretraining_loss(
        epoch_losses,
        os.path.join(GRAPHS_DIR, "simclr_pretraining_loss.png")
    )

    # Similarity matrix AFTER training
    pos_after, neg_after = measure_trained_similarity(model, DEVICE)
    print(f"\n[after_training] Avg similarity — positive pairs : {pos_after:.4f}")
    print(f"[after_training] Avg similarity — negative pairs : {neg_after:.4f}")

    # Visualise the 2N×2N matrix after training
    raw_ds  = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True,
        transform=TwoViewTransform(simclr_augment)
    )
    loader  = torch.utils.data.DataLoader(raw_ds, batch_size=16, shuffle=False)
    (v1, v2), _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        _, z1 = model(v1.to(DEVICE))
        _, z2 = model(v2.to(DEVICE))
    z_cat   = torch.cat([z1, z2], dim=0)
    sim_np  = build_cosine_sim_matrix(z_cat).cpu().numpy()
    visualise_sim_matrix(
        sim_np, n=16,
        title="Similarity Matrix — After SimCLR Training",
        save_path=os.path.join(RESULTS_DIR, "similarity_matrix_after_training.png")
    )

    return model


def main() -> None:
    model = pretrain_simclr()
    print("\nTask 5 complete.")


if __name__ == "__main__":
    main()
