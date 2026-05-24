"""
Task 4 — SimCLR Implementation
================================
Implements:
  4.1  Encoder (ResNet-18 for CIFAR) + Projection Head
  4.2  Positive / negative pair construction (explained in report table)
  4.3  Cosine similarity matrix + visualisation
  4.4  NT-Xent loss (hand-rolled — no library)

This file is importable by task5 (pretraining) without running main().
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from simclr_utils import lock_seeds, SEED

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.1  Encoder and Projection Head
# ══════════════════════════════════════════════════════════════════════════════

class CIFAREncoder(nn.Module):
    """
    ResNet-18 backbone adapted for 32×32 CIFAR images.
    Output: 512-dimensional feature vector per image.
    The classification head is NOT part of this module.
    """
    def __init__(self):
        super().__init__()
        backbone    = torchvision.models.resnet18(weights=None)
        # CIFAR fix: replace 7×7 conv with 3×3, remove maxpool
        backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        # Drop the original FC so forward returns 512-d avg-pool output
        self.feature_dim = backbone.fc.in_features   # 512
        backbone.fc      = nn.Identity()
        self.backbone    = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # [B, 512]


class ProjectionHead(nn.Module):
    """
    Small MLP used ONLY during SimCLR pretraining.
    512 → 256 → ReLU → 128
    Following the paper, contrastive loss is applied on the projected space.
    """
    def __init__(self, in_dim: int = 512,
                 hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)   # [B, 128]


class SimCLRModel(nn.Module):
    """
    Full SimCLR model = Encoder + Projection Head.
    Only used during pretraining. For downstream tasks we use
    encoder.backbone directly.
    """
    def __init__(self):
        super().__init__()
        self.encoder    = CIFAREncoder()
        self.proj_head  = ProjectionHead(in_dim=self.encoder.feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h, z): backbone features and projected features."""
        h = self.encoder(x)
        z = self.proj_head(h)
        return h, z


class ClassificationHead(nn.Module):
    """
    Simple linear layer on top of the 512-d encoder.
    Used for supervised baseline, linear probing, and fine-tuning.
    """
    def __init__(self, in_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


# ══════════════════════════════════════════════════════════════════════════════
# 4.3  Cosine Similarity Matrix
# ══════════════════════════════════════════════════════════════════════════════

def build_cosine_sim_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Computes the full [2N, 2N] cosine similarity matrix for a batch of
    2N projected feature vectors z.
    """
    z_hat = F.normalize(z, dim=1)   # L2-normalise each row
    return z_hat @ z_hat.T          # [2N, 2N]


def visualise_sim_matrix(sim_mat: np.ndarray, n: int,
                          title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim_mat, ax=ax, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0,
                xticklabels=False, yticklabels=False, square=True)
    # Dashed lines separate the view-1 block from the view-2 block
    ax.axhline(n, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axvline(n, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("2N augmented views  (first N = view-A,  last N = view-B)")
    ax.set_ylabel("same ordering")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved similarity matrix → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.4  NT-Xent Loss  (hand-rolled — no library)
# ══════════════════════════════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy loss (NT-Xent).

    For a batch of N images generating 2N views [z_0, ..., z_{N-1},
    z_N, ..., z_{2N-1}] where z_i and z_{i+N} are a positive pair:

      loss(i, j) = -log [
          exp(sim(z_i, z_j) / tau)
          / sum_{k != i} exp(sim(z_i, z_k) / tau)
      ]

    The total loss averages over all 2N anchor views.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args
            z : [2N, proj_dim] — projected, NOT yet normalised

        Returns
            scalar loss
        """
        two_n = z.size(0)
        n     = two_n // 2
        device = z.device

        # Step 1: L2 normalise
        z_norm = F.normalize(z, dim=1)

        # Step 2: full similarity matrix scaled by temperature
        # shape [2N, 2N]
        logits = (z_norm @ z_norm.T) / self.tau

        # Step 3: mask out self-similarities on the diagonal
        # (we must not count sim(z_i, z_i) in the denominator)
        diag_mask = torch.eye(two_n, dtype=torch.bool, device=device)
        logits = logits.masked_fill(diag_mask, -1e9)

        # Step 4: build positive-pair targets
        # For anchor i in [0, N): its positive is at index i+N
        # For anchor j in [N, 2N): its positive is at index j-N
        targets = torch.cat([
            torch.arange(n, two_n, device=device),   # view-A → view-B partner
            torch.arange(0, n,     device=device),   # view-B → view-A partner
        ])  # shape [2N]

        # Step 5: standard cross-entropy over the row-wise softmax
        loss = F.cross_entropy(logits, targets)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# Standalone demo for Task 4
# ══════════════════════════════════════════════════════════════════════════════

def demo_pair_table(n: int = 4) -> None:
    print(f"\nPositive-pair index table for N={n} images (batch size = 2N={2*n}):")
    print(f"{'Original Image':<18} {'View-A Index':<14} {'View-B Index':<14} {'Positive Pair'}")
    print("-" * 62)
    for i in range(n):
        print(f"  image {i:<12}  {i:<14}  {i+n:<14}  yes")


def main() -> None:
    lock_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── show pair construction table ──────────────────────────────────────
    demo_pair_table(n=4)

    # ── build model and run one forward pass ──────────────────────────────
    model = SimCLRModel().to(device)
    model.eval()

    dummy_batch = torch.randn(8, 3, 32, 32).to(device)  # 4 images × 2 views
    with torch.no_grad():
        _, z = model(dummy_batch)

    print(f"\nEncoder output dim : {model.encoder.feature_dim}")
    print(f"Projection output  : {z.shape[1]}")

    # ── similarity matrix before training ────────────────────────────────
    z_all    = z   # reuse the 8-sample batch as a tiny demo
    sim_np   = build_cosine_sim_matrix(z_all).cpu().numpy()
    save_p   = os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png")
    visualise_sim_matrix(sim_np, n=dummy_batch.size(0) // 2,
                          title="Similarity Matrix — Before SimCLR Training",
                          save_path=save_p)

    # ── NT-Xent sanity check ──────────────────────────────────────────────
    loss_fn  = NTXentLoss(temperature=0.5)
    dummy_z  = torch.randn(16, 128).to(device)   # 8 images, 2 views each
    loss_val = loss_fn(dummy_z)
    print(f"\nNT-Xent sanity check  (random input, expect ~log(2N-1)): "
          f"{loss_val.item():.4f}  vs expected ≈ {math.log(15):.4f}")

    print("\nTask 4 complete.")
    print("\nAnswers to Q4.3:")
    print("  1. Diagonal is ignored — sim(z_i, z_i)=1 always; including it")
    print("     would make the loss trivially easy and distort gradients.")
    print("  2. Positive pairs are at off-diagonal positions [i, i+N] and [i+N, i].")
    print("  3. All other pairs are negatives by assumption — we treat any two")
    print("     images from different originals as semantically dissimilar.")


if __name__ == "__main__":
    main()
