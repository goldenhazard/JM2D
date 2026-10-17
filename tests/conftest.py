"""Test environment checks.

Markers:
    gpu   needs a CUDA GPU and the checkpoints (golden equivalence cases, tiny sampling runs)
    ckpt  needs the checkpoint files only

Without the requirements these tests are skipped. With ``JM2D_REQUIRE_GOLDEN=1`` a missing GPU or checkpoint
is an error instead, so a CI or release check cannot pass silently without running the golden cases.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from jm2d_toy.config import load_config

REQUIRE_GOLDEN = os.environ.get("JM2D_REQUIRE_GOLDEN", "0").strip().lower() in ("1", "true", "yes")


def _skip_or_fail(reason):
    if REQUIRE_GOLDEN:
        pytest.fail(f"{reason} (JM2D_REQUIRE_GOLDEN=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


def _missing_checkpoints():
    return [load_config(task)["checkpoint"] for task in ("donut_joint", "donut_constrained")
            if not os.path.exists(load_config(task)["checkpoint"])]


@pytest.fixture(autouse=True)
def _check_requirements(request):
    needs_gpu = request.node.get_closest_marker("gpu") is not None
    needs_ckpt = needs_gpu or request.node.get_closest_marker("ckpt") is not None
    if needs_ckpt:
        missing = _missing_checkpoints()
        if missing:
            _skip_or_fail(f"checkpoint(s) missing: {', '.join(missing)}")
    if needs_gpu and not torch.cuda.is_available():
        _skip_or_fail("no CUDA GPU available")
