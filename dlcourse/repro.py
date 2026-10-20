"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed=0, deterministic=False):
    """Seed Python, NumPy, and PyTorch (CPU and all GPUs).

    With ``deterministic=True``, also ask PyTorch to use deterministic
    algorithms and raise an error for operations that have none. This can
    be slower, and some GPU operations will refuse to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
