"""
Loads CIFAR-10 using the fixed index split files provided with the assignment.
We must NOT use random_split() or create our own splits — these txt files
are the ground truth for all experiments.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np


# ── paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def _load_index_file(filename: str) -> list[int]:
    """Read a txt file containing one integer index per line."""
    fpath = os.path.join(SPLITS_DIR, filename)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Split file not found: {fpath}\n"
            "Run generate_splits.py first to create the split files."
        )
    with open(fpath, "r") as fh:
        indices = [int(line.strip()) for line in fh if line.strip()]
    return indices


# ── base transforms ────────────────────────────────────────────────────────────
def get_plain_transform() -> T.Compose:
    """Simple transform — no augmentation, just normalise. Used for eval."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_supervised_train_transform() -> T.Compose:
    """Standard augmentation for supervised classification training."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ── dataset split loaders ──────────────────────────────────────────────────────
def load_labeled_train(data_root: str, transform=None, batch_size: int = 64,
                       shuffle: bool = True) -> DataLoader:
    """10 % labeled training split — used for supervised baseline and probing."""
    if transform is None:
        transform = get_supervised_train_transform()
    base_ds = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                           download=True, transform=transform)
    idx = _load_index_file("train_labeled_10percent.txt")
    ds  = Subset(base_ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def load_unlabeled_train(data_root: str, transform=None,
                         batch_size: int = 64) -> DataLoader:
    """Full unlabeled training split — used only for SimCLR pretraining."""
    if transform is None:
        transform = get_plain_transform()  # will be overridden with TwoViewTransform
    base_ds = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                           download=True, transform=transform)
    idx = _load_index_file("train_ssl_unlabeled.txt")
    ds  = Subset(base_ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)


def load_val(data_root: str, transform=None, batch_size: int = 128) -> DataLoader:
    """Validation split — used for model selection."""
    if transform is None:
        transform = get_plain_transform()
    base_ds = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                           download=True, transform=transform)
    idx = _load_index_file("val.txt")
    ds  = Subset(base_ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def load_test(data_root: str, transform=None, batch_size: int = 128) -> DataLoader:
    """Test split — only touched for final reporting."""
    if transform is None:
        transform = get_plain_transform()
    base_ds = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=transform)
    idx = _load_index_file("test.txt")
    ds  = Subset(base_ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def get_val_subset_for_viz(data_root: str, n: int = 1000,
                           seed: int = 2026) -> Subset:
    """
    Returns a fixed subset of n validation images for PCA / t-SNE plots.
    seed is fixed so every run picks the same images.
    """
    plain = get_plain_transform()
    base_ds = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                           download=True, transform=plain)
    val_idx = _load_index_file("val.txt")
    rng     = np.random.default_rng(seed)
    chosen  = rng.choice(val_idx, size=n, replace=False).tolist()
    return Subset(base_ds, chosen)
