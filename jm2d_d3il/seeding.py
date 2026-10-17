"""Seeding and determinism helpers (same settings for the golden captures, the tests and all reported runs)."""
import os
import random

import numpy as np
import torch


def enable_determinism():
    """Deterministic cuBLAS / cuDNN. ``CUBLAS_WORKSPACE_CONFIG`` must be set before the CUDA context exists."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed):
    """Same as ``diffuser.utils.setup.set_seed`` (called by the original ``utils.Parser`` before loading)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
