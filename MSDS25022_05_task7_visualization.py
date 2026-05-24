"""
Task 8 — PCA / t-SNE Feature Visualisation
=============================================
Extracts 512-d features from 1000 fixed validation images and reduces
them to 2-D using t-SNE (or PCA) for three encoder states:
  1. Random (untrained) encoder
  2. SimCLR pretrained encoder
  3. Fine-tuned encoder

Labels are used ONLY for colouring the scatter plot — not during training.

Expected outputs:
  results/random_encoder_pca_or_tsne.png
  results/simclr_encoder_pca_or_tsne.png
  results/finetuned_encoder_pca_or_tsne.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from simclr_utils import lock_seeds, SEED, get_val_subset_for_viz, CIFAR10_CLASSES
from MSDS25022_05_task4_simclr import CIFAREncoder, ClassificationHead
from MSDS25022_05_task6_finetune import FullClassifier

DATA_ROOT   = "./data"
RESULTS_DIR = "./results"
MODELS_DIR  = "./models"

N_VIZ   = 1000     # fixed val subset size
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Ten visually distinct colours — one per CIFAR-10 class
CLASS_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
    "#00ACC1", "#F4511E", "#6D4C41", "#546E7A", "#FFB300",
]


# ── feature extraction ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder: torch.nn.Module,
                     loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N, 512], labels [N])."""
    encoder.eval()
    all_feats  = []
    all_labels = []
    for imgs, lbls in loader:
        feats = encoder(imgs.to(DEVICE)).cpu().numpy()
        all_feats.append(feats)
        all_labels.append(lbls.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ── dimensionality reduction ───────────────────────────────────────────────────
def reduce_to_2d(feats: np.ndarray, method: str = "tsne",
                 seed: int = SEED) -> np.ndarray:
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=40, n_iter=1000,
                       random_state=seed, learning_rate="auto",
                       init="pca")
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(feats)


# ── scatter plot ───────────────────────────────────────────────────────────────
def plot_embedding(coords: np.ndarray, labels: np.ndarray,
                   title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in range(10):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CLASS_COLORS[cls], s=8, alpha=0.7, linewidths=0)

    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CIFAR10_CLASSES[c])
        for c in range(10)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, markerscale=1.5, framealpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved embedding plot → {save_path}")


def main() -> None:
    lock_seeds(SEED)

    # ── fixed 1000-image val subset ───────────────────────────────────────
    val_subset  = get_val_subset_for_viz(DATA_ROOT, n=N_VIZ, seed=SEED)
    val_loader  = DataLoader(val_subset, batch_size=128, shuffle=False,
                              num_workers=2)

    # ── 1. Random encoder ─────────────────────────────────────────────────
    print("Extracting features: random encoder…")
    rand_enc = CIFAREncoder().to(DEVICE)
    feats_r, labels = extract_features(rand_enc, val_loader)
    coords_r = reduce_to_2d(feats_r)
    plot_embedding(coords_r, labels,
                   title="t-SNE — Random (Untrained) Encoder",
                   save_path=os.path.join(RESULTS_DIR,
                                          "random_encoder_pca_or_tsne.png"))

    # ── 2. SimCLR encoder ─────────────────────────────────────────────────
    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError("Run task5_pretraining.py first.")
    print("Extracting features: SimCLR encoder…")
    simclr_enc = CIFAREncoder().to(DEVICE)
    simclr_enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    feats_s, _ = extract_features(simclr_enc, val_loader)
    coords_s   = reduce_to_2d(feats_s)
    plot_embedding(coords_s, labels,
                   title="t-SNE — SimCLR Pretrained Encoder",
                   save_path=os.path.join(RESULTS_DIR,
                                          "simclr_encoder_pca_or_tsne.png"))

    # ── 3. Fine-tuned encoder ─────────────────────────────────────────────
    ft_path = os.path.join(MODELS_DIR, "finetuned_model.pt")
    if not os.path.exists(ft_path):
        raise FileNotFoundError("Run task6_finetune.py first.")
    print("Extracting features: fine-tuned encoder…")
    ft_enc   = CIFAREncoder().to(DEVICE)
    ft_model = FullClassifier(ft_enc).to(DEVICE)
    ft_model.load_state_dict(torch.load(ft_path, map_location=DEVICE))
    feats_f, _ = extract_features(ft_model.encoder, val_loader)
    coords_f   = reduce_to_2d(feats_f)
    plot_embedding(coords_f, labels,
                   title="t-SNE — Fine-Tuned Encoder",
                   save_path=os.path.join(RESULTS_DIR,
                                          "finetuned_encoder_pca_or_tsne.png"))

    print("\nTask 8 complete.")
    print("\nExpected observations:")
    print("  Random encoder   → no class structure; points scattered uniformly.")
    print("  SimCLR encoder   → loose clusters start forming (semantics preserved).")
    print("  Fine-tuned enc.  → tightest class clusters; best separation.")
    print("  Likely confused pairs: cat↔dog, automobile↔truck, deer↔horse.")


if __name__ == "__main__":
    main()
