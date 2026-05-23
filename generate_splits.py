"""
generate_splits.py
------------------
Creates the four fixed index-split files for the assignment.
Run this ONCE before any training. The splits are deterministic
because we use seed=2026 everywhere.

Strategy (balanced per class):
  - CIFAR-10 train: 50,000 images  (indices 0-49999)
  - CIFAR-10 test : 10,000 images  (indices 0-9999)

  val                  :  5,000 images  (500 per class, from train)
  train_labeled_10pct  :  5,000 images  (500 per class, from remaining train)
  train_ssl_unlabeled  : 40,000 images  (the rest of train — no labels used)
  test                 : 10,000 images  (entire test set)
"""

import os
import numpy as np
import torchvision

SEED       = 2026
_HERE      = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(_HERE, "splits")
DATA_ROOT  = os.path.join(_HERE, "data")

VAL_PER_CLASS      = 500   # 5 000 total
LABELED_PER_CLASS  = 500   # 5 000 total  (~10 % of 50 k)


def write_index_file(fname: str, indices) -> None:
    os.makedirs(SPLITS_DIR, exist_ok=True)
    fpath = os.path.join(SPLITS_DIR, fname)
    with open(fpath, "w") as fh:
        for idx in sorted(indices):
            fh.write(f"{idx}\n")
    print(f"  wrote {len(indices):>6d} indices → {fpath}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Downloading CIFAR-10 (if not already cached)…")
    train_ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                            download=True)
    test_ds  = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                            download=True)

    train_labels = np.array(train_ds.targets)
    num_classes  = 10

    val_idx     = []
    labeled_idx = []
    ssl_idx     = []

    for cls in range(num_classes):
        cls_positions = np.where(train_labels == cls)[0]
        # shuffle deterministically
        rng.shuffle(cls_positions)

        val_positions     = cls_positions[:VAL_PER_CLASS]
        labeled_positions = cls_positions[VAL_PER_CLASS: VAL_PER_CLASS + LABELED_PER_CLASS]
        ssl_positions     = cls_positions[VAL_PER_CLASS + LABELED_PER_CLASS:]

        val_idx.extend(val_positions.tolist())
        labeled_idx.extend(labeled_positions.tolist())
        ssl_idx.extend(ssl_positions.tolist())

    test_idx = list(range(len(test_ds)))

    print("\nWriting split files…")
    write_index_file("val.txt",                    val_idx)
    write_index_file("train_labeled_10percent.txt", labeled_idx)
    write_index_file("train_ssl_unlabeled.txt",     ssl_idx)
    write_index_file("test.txt",                    test_idx)

    print("\nSplit summary:")
    print(f"  val                  : {len(val_idx)}")
    print(f"  train_labeled_10pct  : {len(labeled_idx)}")
    print(f"  train_ssl_unlabeled  : {len(ssl_idx)}")
    print(f"  test                 : {len(test_idx)}")
    print("\nDone. Split files are in ./splits/")


if __name__ == "__main__":
    main()
