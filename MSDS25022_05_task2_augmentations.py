"""
Task 2 — Understanding Augmentations
=====================================
Implements the SimCLR augmentation pipeline and the TwoViewTransform
wrapper, then visualises 10 original+view1+view2 triplets side by side.

Expected outputs:
  results/augmentation_examples.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simclr_utils import lock_seeds, SEED, CIFAR10_MEAN, CIFAR10_STD, CIFAR10_CLASSES

DATA_ROOT   = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── SimCLR augmentation pipeline ──────────────────────────────────────────────
# Exactly as specified in the assignment sheet — no changes allowed.
simclr_augment = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ── TwoViewTransform ───────────────────────────────────────────────────────────
class TwoViewTransform:
    """
    Applies the same stochastic transform twice to produce two
    different augmented views of the same PIL image.
    The two views form the positive pair for SimCLR.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        view_a = self.transform(img)
        view_b = self.transform(img)
        return view_a, view_b


# ── helper: denormalise a tensor for display ──────────────────────────────────
_inv_mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
_inv_std  = torch.tensor(CIFAR10_STD ).view(3, 1, 1)


def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """
    Reverse the CIFAR normalisation and convert to HWC uint8 for plt.imshow.
    """
    img = t.cpu().clone() * _inv_std + _inv_mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return img


# ── visualisation ─────────────────────────────────────────────────────────────
def visualise_augmentations(num_examples: int = 10, save_path: str = "") -> None:
    """
    Displays num_examples rows of: Original | View A | View B.
    Original is shown with just ToTensor (no augmentation).
    """
    lock_seeds(SEED)

    # We need the raw PIL images → load with no transform first
    raw_ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                          download=True, transform=None)

    # Two-view transform
    two_view = TwoViewTransform(simclr_augment)
    plain_t  = T.Compose([T.ToTensor(),
                           T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])

    # Pick a fixed spread of indices (2 per class for 10 rows)
    rng      = np.random.default_rng(SEED)
    show_idx = rng.choice(len(raw_ds), size=num_examples, replace=False).tolist()

    fig, axes = plt.subplots(num_examples, 3,
                              figsize=(7, num_examples * 2.2))
    col_titles = ["Original", "Augmented View A", "Augmented View B"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(show_idx):
        pil_img, label = raw_ds[idx]

        orig_tensor     = plain_t(pil_img)
        view_a, view_b  = two_view(pil_img)

        axes[row, 0].imshow(tensor_to_display(orig_tensor))
        axes[row, 1].imshow(tensor_to_display(view_a))
        axes[row, 2].imshow(tensor_to_display(view_b))

        axes[row, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=9, rotation=0,
                                 labelpad=42)
        for col in range(3):
            axes[row, col].axis("off")

    plt.suptitle("SimCLR Augmentation Examples", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation examples → {save_path}")


def main() -> None:
    lock_seeds(SEED)
    out = os.path.join(RESULTS_DIR, "augmentation_examples.png")
    visualise_augmentations(num_examples=10, save_path=out)
    print("\nTask 2 complete.")
    print("\nObservations:")
    print("  1. The two augmented views are NOT identical — randomness ensures diversity.")
    print("  2. Both views clearly represent the same object despite colour/crop differences.")
    print("  3. SimCLR should treat them as a positive pair because they share the same")
    print("     semantic content; any differences are purely stylistic / viewpoint noise.")
    print("  4. Too-weak augmentations → trivial task; model learns nothing meaningful.")
    print("  5. Too-strong augmentations → views may lose shared semantics; loss diverges.")


if __name__ == "__main__":
    main()
