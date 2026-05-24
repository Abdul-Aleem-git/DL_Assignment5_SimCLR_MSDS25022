"""
Task 3 — Feature Similarity Before Training
============================================
Passes a batch of images through a randomly initialised (untrained)
ResNet-18 encoder and measures cosine similarity between:
  - two augmented views of the same image (positive pairs)
  - views from different images (negative pairs)

Expected outputs:
  results/similarity_matrix_before_training.png   (also used by Task 4)
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from simclr_utils import lock_seeds, SEED, CIFAR10_MEAN, CIFAR10_STD
from MSDS25022_05_task2_augmentations import TwoViewTransform, simclr_augment

DATA_ROOT   = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_N  = 16   # N images → 2N views in the similarity matrix
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── encoder builder (shared with later tasks) ──────────────────────────────────
def build_encoder(pretrained: bool = False) -> torch.nn.Module:
    """
    ResNet-18 adapted for 32×32 CIFAR images.
    Returns 512-dim features from the average-pool layer.
    The fc layer is stripped so the output is pure feature vector.
    """
    net = torchvision.models.resnet18(weights=None)
    # Adjust for CIFAR's small spatial size
    net.conv1   = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
    net.maxpool = torch.nn.Identity()
    # Remove classification head — we just want features
    net.fc      = torch.nn.Identity()
    return net


# ── cosine similarity matrix ───────────────────────────────────────────────────
def cosine_sim_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Given feat of shape [2N, D], returns a [2N, 2N] cosine similarity matrix.
    feat is L2-normalised first so sim = feat @ feat^T.
    """
    feat_norm = F.normalize(feat, dim=1)
    return feat_norm @ feat_norm.T


def plot_similarity_heatmap(sim_mat: np.ndarray, title: str,
                             save_path: str, n: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(sim_mat, ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    # Draw a dividing line at N to separate view1 block from view2 block
    ax.axhline(n, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(n, color="white", linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("2N augmented views")
    ax.set_ylabel("2N augmented views")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved similarity heatmap → {save_path}")


def compute_avg_similarities(sim_mat: torch.Tensor,
                              n: int) -> tuple[float, float]:
    """
    From the 2N×2N similarity matrix, extract:
      - positive pair similarity  : sim[i, i+N]  for i in 0..N-1
      - negative pair similarity  : all other off-diagonal entries
    """
    pos_sims = []
    neg_sims = []
    size = 2 * n
    for i in range(n):
        j = i + n          # view2 of the same image lives at index i+N
        pos_sims.append(sim_mat[i, j].item())
        pos_sims.append(sim_mat[j, i].item())

    mask = torch.ones(size, size, dtype=torch.bool)
    mask.fill_diagonal_(False)       # ignore self-similarity
    for i in range(n):
        j = i + n
        mask[i, j] = False           # ignore positive pairs
        mask[j, i] = False

    neg_sims = sim_mat[mask].tolist()
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


def measure_random_encoder_similarity(save_path: str,
                                       tag: str = "before_training"
                                       ) -> tuple[float, float]:
    lock_seeds(SEED)

    # ── prepare a small batch with TwoViewTransform ────────────────────────
    two_view_ds = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True,
        transform=TwoViewTransform(simclr_augment)
    )
    loader = torch.utils.data.DataLoader(
        two_view_ds, batch_size=BATCH_N, shuffle=True
    )
    (view1_batch, view2_batch), _ = next(iter(loader))

    # ── random encoder ─────────────────────────────────────────────────────
    encoder = build_encoder().to(DEVICE)
    encoder.eval()
    with torch.no_grad():
        f1 = encoder(view1_batch.to(DEVICE))   # [N, 512]
        f2 = encoder(view2_batch.to(DEVICE))   # [N, 512]

    # Stack as [2N, 512]: first N rows = view1, next N rows = view2
    feats  = torch.cat([f1, f2], dim=0)
    sim_mat = cosine_sim_matrix(feats)

    avg_pos, avg_neg = compute_avg_similarities(sim_mat, n=BATCH_N)
    print(f"\n[{tag}] Average cosine similarity — positive pairs : {avg_pos:.4f}")
    print(f"[{tag}] Average cosine similarity — negative pairs : {avg_neg:.4f}")

    plot_similarity_heatmap(
        sim_mat.cpu().numpy(),
        title=f"Cosine Similarity Matrix ({tag.replace('_',' ').title()})",
        save_path=save_path,
        n=BATCH_N,
    )
    return avg_pos, avg_neg


def main() -> None:
    out = os.path.join(RESULTS_DIR, "similarity_matrix_before_training.png")
    pos_sim, neg_sim = measure_random_encoder_similarity(save_path=out)

    print("\nInterpretation:")
    print("  Diagonal        → self-similarity = 1.0, ignored in loss.")
    print("  Off-diag blocks → positive pairs sit at [i, i+N] / [i+N, i].")
    print("  All other pairs → treated as negatives during contrastive learning.")
    print("\n  With a random encoder, positive pairs are NOT noticeably more")
    print("  similar than negatives — the model hasn't learned anything yet.")
    print("\nTask 3 complete.")


if __name__ == "__main__":
    main()
