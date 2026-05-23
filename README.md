# DL Assignment 5 — SimCLR
**Student:** Muhammad Abdul Aleem  
**Roll No:** MSDS25022  
**Course:** Deep Learning — Spring 2026  

---

## Overview
This assignment implements SimCLR (Self-Supervised Contrastive Learning) on CIFAR-10,
starting from a supervised baseline and progressively moving to self-supervised pretraining,
linear probing, and fine-tuning.

---

## Project Structure
```
MSDS25022_05/
├── splits/
│   ├── train_ssl_unlabeled.txt
│   ├── train_labeled_10percent.txt
│   ├── val.txt
│   └── test.txt
├── utils/
│   ├── seed.py
│   ├── dataset_splits.py
│   └── metrics.py
├── models/                          # saved checkpoints (not in git, submit via Drive)
├── results/                         # PNG outputs + metrics.json
├── graphs/                          # loss and accuracy curves
├── generate_splits.py
├── MSDS25022_05_task1_supervised.py
├── MSDS25022_05_task2_augmentations.py
├── MSDS25022_05_task3_similarity.py
├── MSDS25022_05_task4_simclr.py
├── MSDS25022_05_task5_pretraining.py
├── MSDS25022_05_task5_linear_probe.py
├── MSDS25022_05_task6_finetune.py
├── MSDS25022_05_task7_visualization.py
├── MSDS25022_05_allCode.py
├── requirements.txt
└── Report.pdf
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Step 0 — Generate split files (run once)
```bash
python generate_splits.py
```

### Run individual tasks
```bash
python MSDS25022_05_task1_supervised.py
python MSDS25022_05_task2_augmentations.py
python MSDS25022_05_task3_similarity.py
python MSDS25022_05_task4_simclr.py
python MSDS25022_05_task5_pretraining.py
python MSDS25022_05_task5_linear_probe.py   # requires task5 done first
python MSDS25022_05_task6_finetune.py       # requires task5 done first
python MSDS25022_05_task7_visualization.py  # requires task5+6 done first
```

### Or run everything at once
```bash
python MSDS25022_05_allCode.py --task all
```

---

## Fixed Training Settings

| Setting | Value |
|---|---|
| Seed | 2026 |
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| Temperature τ | 0.5 |
| Projection dim | 128 |

---

## Model Weights
Large `.pt` files are **not** included in this repository.  
They are uploaded to Google Drive: [link to be added before submission]

---

## Notes
- The dataset (CIFAR-10) is downloaded automatically by torchvision on first run.
- Do not commit the `data/` folder or `models/*.pt` files to git.
- All randomness is fixed via `seed=2026` for reproducibility.
