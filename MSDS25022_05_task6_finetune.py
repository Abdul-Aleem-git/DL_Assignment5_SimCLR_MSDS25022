"""
Task 7 — Fine-tuning the SimCLR Encoder
==========================================
Initialises the encoder from SimCLR pretrained weights and trains the
full model (encoder + classification head) end-to-end on the 10 % labeled split.

Expected outputs:
  graphs/finetuning_accuracy.png
  models/finetuned_model.pt
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
    compute_accuracy, plot_confusion_matrix,
    save_test_predictions, save_metrics,
    CIFAR10_CLASSES,
)
from MSDS25022_05_task4_simclr import CIFAREncoder, ClassificationHead

DATA_ROOT   = "./data"
GRAPHS_DIR  = "./graphs"
RESULTS_DIR = "./results"
MODELS_DIR  = "./models"

NUM_EPOCHS  = 20
BATCH_SIZE  = 64
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(GRAPHS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── full model (unfrozen) ──────────────────────────────────────────────────────
class FullClassifier(nn.Module):
    """Encoder + linear head — both sets of weights are trainable."""
    def __init__(self, encoder: CIFAREncoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head    = ClassificationHead(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def train_and_eval(model: nn.Module, tag: str) -> tuple[float, list, list]:
    """
    Trains model on labeled 10 % split, returns (test_acc, train_hist, val_hist).
    """
    lock_seeds(SEED)

    train_loader = load_labeled_train(DATA_ROOT, batch_size=BATCH_SIZE)
    val_loader   = load_val(DATA_ROOT,  batch_size=128)
    test_loader  = load_test(DATA_ROOT, batch_size=128)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=NUM_EPOCHS
    )

    best_val_acc   = 0.0
    best_state     = None
    train_acc_hist = []
    val_acc_hist   = []

    print(f"\n  [{tag}] Fine-tuning ({NUM_EPOCHS} epochs)…")
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = criterion(model(imgs), lbls)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()

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

    model.load_state_dict(best_state)
    test_acc = compute_accuracy(model, test_loader, DEVICE)
    print(f"  [{tag}] Test accuracy: {test_acc*100:.2f}%")
    return test_acc, train_acc_hist, val_acc_hist, model, test_loader


def plot_finetuning_comparison(histories: dict, save_path: str) -> None:
    """
    Plots val accuracy curves for all four experimental conditions.
    histories = {label: [val_acc_per_epoch, ...]}
    """
    palette = {
        "Supervised (scratch)"   : "#2196F3",
        "Random + Linear Probe"  : "#607D8B",
        "SimCLR + Linear Probe"  : "#4CAF50",
        "SimCLR + Fine-tune"     : "#FF5722",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, hist in histories.items():
        color = palette.get(label, "gray")
        ax.plot(range(1, len(hist) + 1), [v * 100 for v in hist],
                label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Comparison of All Training Strategies (Validation Accuracy)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved fine-tuning comparison plot → {save_path}")


def main() -> None:
    lock_seeds(SEED)

    enc_path = os.path.join(MODELS_DIR, "simclr_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(
            "SimCLR encoder checkpoint not found. Run task5_pretraining.py first."
        )

    # ── load pretrained encoder ───────────────────────────────────────────
    pretrained_enc = CIFAREncoder().to(DEVICE)
    pretrained_enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))

    # ── fine-tune full model ──────────────────────────────────────────────
    ft_model = FullClassifier(pretrained_enc).to(DEVICE)
    acc_ft, tr_hist, val_hist_ft, ft_model, test_loader = train_and_eval(
        ft_model, tag="SimCLR Fine-tune"
    )

    # Save fine-tuned model
    torch.save(ft_model.state_dict(),
               os.path.join(MODELS_DIR, "finetuned_model.pt"))

    # ── save test predictions (used in metrics.json) ──────────────────────
    save_test_predictions(
        ft_model, test_loader, DEVICE,
        os.path.join(RESULTS_DIR, "test_predictions.csv")
    )

    # ── aggregate all results into metrics.json ───────────────────────────
    # Load partial metrics from task 1 if available
    partial_path = os.path.join(RESULTS_DIR, "metrics_partial.json")
    if os.path.exists(partial_path):
        import json
        with open(partial_path) as fh:
            partial = json.load(fh)
    else:
        partial = {}

    # Load linear probe accuracies (written by task6 if run first)
    lp_path = os.path.join(RESULTS_DIR, "linear_probe_results.json")
    if os.path.exists(lp_path):
        import json
        with open(lp_path) as fh:
            lp_data = json.load(fh)
    else:
        lp_data = {}

    metrics = {
        "student_name"                  : "Muhammad Abdul Aleem",
        "roll_number"                   : "MSDS25022",
        "seed"                          : SEED,
        "batch_size"                    : BATCH_SIZE,
        "simclr_epochs"                 : 50,
        "linear_probe_epochs"           : 20,
        "finetuning_epochs"             : NUM_EPOCHS,
        "learning_rate"                 : LR,
        "temperature"                   : 0.5,
        "optimizer"                     : "Adam",
        "supervised_10percent_test_acc" : partial.get("supervised_10percent_test_acc", 0.0),
        "random_linear_probe_test_acc"  : lp_data.get("random_linear_probe_test_acc",  0.0),
        "simclr_linear_probe_test_acc"  : lp_data.get("simclr_linear_probe_test_acc",  0.0),
        "simclr_finetune_test_acc"      : round(acc_ft, 4),
        "same_view_similarity_before"   : lp_data.get("same_view_similarity_before",   0.0),
        "different_image_similarity_before": lp_data.get("different_image_similarity_before", 0.0),
        "same_view_similarity_after"    : lp_data.get("same_view_similarity_after",    0.0),
        "different_image_similarity_after" : lp_data.get("different_image_similarity_after",  0.0),
    }
    save_metrics(metrics, os.path.join(RESULTS_DIR, "metrics.json"))

    print(f"\nFine-tune test accuracy: {acc_ft*100:.2f}%")
    print("\nTask 7 complete.")

    return acc_ft, val_hist_ft


if __name__ == "__main__":
    main()
