"""
Helper functions for evaluation: accuracy, confusion matrix,
metrics.json saving, and test_predictions.csv export.
"""

import json
import os
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ── accuracy ───────────────────────────────────────────────────────────────────
def compute_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    """
    Runs the model on every batch in loader and returns top-1 accuracy.
    Model must map inputs -> logits (shape [B, num_classes]).
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ── confusion matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(model: torch.nn.Module, loader,
                          device: torch.device, save_path: str) -> None:
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


# ── metrics.json ───────────────────────────────────────────────────────────────
def save_metrics(metrics_dict: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(metrics_dict, fh, indent=2)
    print(f"Saved metrics → {save_path}")


def load_metrics(save_path: str) -> dict:
    with open(save_path, "r") as fh:
        return json.load(fh)


# ── test_predictions.csv ───────────────────────────────────────────────────────
def save_test_predictions(model: torch.nn.Module, loader,
                          device: torch.device, save_path: str) -> None:
    """
    Saves image_index, true_label, predicted_label, and per-class
    softmax probabilities to a CSV file.
    """
    model.eval()
    rows = []
    img_idx = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            labs   = labels.numpy()
            for i in range(len(labs)):
                row = [img_idx, labs[i], preds[i]] + list(probs[i])
                rows.append(row)
                img_idx += 1

    header = (["image_index", "true_label", "predicted_label"] +
              [f"prob_class_{c}" for c in range(10)])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved test predictions → {save_path}")
