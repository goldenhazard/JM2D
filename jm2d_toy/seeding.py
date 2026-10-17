"""Seeding and determinism helpers.

``CUBLAS_WORKSPACE_CONFIG`` must be set before the CUDA context is created; importing
``jm2d_toy`` sets a default, and ``enable_determinism`` sets the torch flags.
"""
import os
import random

import numpy as np
import torch


def enable_determinism():
    """Deterministic cuBLAS/cuDNN settings used for the golden outputs and all reported runs."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed):
    """Seed python, numpy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
