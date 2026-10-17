"""JM2D on D3IL-Avoiding (paper Table 2 / Table 3): planners, tasks, runner and result tables.

Importing the package sets ``CUBLAS_WORKSPACE_CONFIG`` (needed for deterministic cuBLAS, before CUDA starts) and puts
``third_party/`` (parent of the patched D3IL clone, imported as ``d3il``) and the repository root on ``sys.path``.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .paths import REPO_ROOT, THIRD_PARTY_DIR  # noqa: E402

for _p in (REPO_ROOT, THIRD_PARTY_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
