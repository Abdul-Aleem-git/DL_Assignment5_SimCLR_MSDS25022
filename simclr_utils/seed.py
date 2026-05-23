"""
Utility to fix all random seeds for reproducibility.
seed = 2026 as required by the assignment.
"""

import os
import random
import numpy as np
import torch


SEED = 2026


def lock_seeds(seed: int = SEED) -> None:
    """
    Set seeds for python random, numpy, and all torch backends.
    Also makes cudnn deterministic so results are reproducible
    even on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two lines slow things down a bit but guarantee reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
