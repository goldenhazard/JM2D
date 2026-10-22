"""Refactored samplers == legacy samplers, bitwise, on every golden case (tests/golden/manifest.json).

The golden files were written by ``tests/golden/capture_legacy.py`` from SafeRobomimic@origin/toy_domain
(b649add0). Each case is re-run here through the public API with the same protocol:
build problem -> Sampler.sample(seed) [seed_everything -> build sampler -> reverse diffusion].
Two legacy captures in separate processes were bitwise identical, so the tolerance is exact equality.
Exception: the ``projection`` case runs scipy ``trust-constr`` on the CPU, whose last bits depend on the
numpy/scipy BLAS build (MKL in the capture env, OpenBLAS in the pip wheels of requirements.txt; max |diff|
~2e-4 after 100 projected steps). Its samples are compared with ``CPU_SOLVER_ATOL``; the counts stay exact.
Golden cases need a CUDA GPU (marker ``gpu``); set ``JM2D_REQUIRE_GOLDEN=1`` to fail instead of skip.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json

import numpy as np
import pytest

from jm2d_toy import samplers, tasks
from jm2d_toy.config import load_config
from jm2d_toy.results import md5
from jm2d_toy.seeding import enable_determinism

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
with open(os.path.join(GOLDEN_DIR, "manifest.json")) as _f:
    MANIFEST = json.load(_f)

# manifest task ids are the paper figure ids; tasks.canonical maps them to the task names
# legacy config section -> new sampler name
NEW_NAME = {"sequential": "sequential", "gibbs_exact": "gibbs", "yongxin": "jm2d", "projection": "projection"}
# legacy hyper-parameter -> new keyword
NEW_KEY = {"N_sample_action": "num_mc_samples", "N_sample_param": "num_param_samples", "temp_mppi": "temperature",
           "max_iter": "max_iter", "tol": "tol", "solver": "solver"}
# projection golden: CPU solver, BLAS-build dependent (see module docstring)
CPU_SOLVER_ATOL = 1e-3

@pytest.fixture(scope="module", autouse=True)
def _determinism():
    enable_determinism()


def assert_bitwise(new, old, what):
    new = np.asarray(new)
    assert new.shape == old.shape, f"{what}: shape {new.shape} != {old.shape}"
    assert new.dtype == old.dtype, f"{what}: dtype {new.dtype} != {old.dtype}"
    if new.tobytes() != old.tobytes():
        diff = np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))
        raise AssertionError(f"{what}: not bitwise equal (max |diff| = {diff:.3e})")


@pytest.mark.ckpt
def test_checkpoints_match_manifest():
    for task, ck in MANIFEST["checkpoints"].items():
        assert md5(load_config(tasks.canonical(task))["checkpoint"]) == ck["md5"], task


@pytest.mark.parametrize("case", sorted(MANIFEST["cases"]))
def test_config_maps_legacy_one_to_one(case):
    """Every legacy hyper-parameter that the sampler reads has the same value in configs/*.json."""
    info = MANIFEST["cases"][case]
    cfg = load_config(tasks.canonical(info["task"]))
    legacy = info["legacy_config"]
    section = "gibbs_exact" if info["legacy_section"] == "sequential" else info["legacy_section"]
    new = cfg["samplers"][NEW_NAME[info["legacy_section"]]]
    used = {"sequential": ["temp_mppi"]}.get(info["legacy_section"],
                                             [k for k in legacy[section] if k in NEW_KEY])
    for key in used:
        assert new[NEW_KEY[key]] == legacy[section][key], (case, key)
    for key, val in cfg["base"].items():
        assert legacy["base"][key] == val, (case, key)
    assert legacy["base"].get("use_ood_penalty", False) is False  # removed argument, unused by kept samplers
    if info["legacy_section"] == "projection":
        assert legacy["projection"]["method"] == "classic"


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(MANIFEST["cases"]))
def test_golden_equivalence(case):
    info = MANIFEST["cases"][case]
    golden = np.load(os.path.join(GOLDEN_DIR, case + ".npz"))

    task = tasks.canonical(info["task"])
    problem = tasks.make(task)
    sampler = samplers.from_config(task, NEW_NAME[info["legacy_section"]])
    res = sampler.sample(problem, n=info["n"], seed=info["seed"], return_history=info["return_histories"])

    if info["legacy_section"] == "projection":
        np.testing.assert_allclose(res.x, golden["x"], rtol=0, atol=CPU_SOLVER_ATOL, err_msg=f"{case}: x")
    else:
        assert_bitwise(res.x, golden["x"], f"{case}: x")
    if "k" in golden.files:
        assert_bitwise(res.k, golden["k"], f"{case}: k")
    for key in golden.files:
        if key.startswith("hist_guide_"):
            assert_bitwise(res.history["guide_log"][key[len("hist_guide_"):]], golden[key], f"{case}: {key}")
        elif key.startswith("hist_"):
            assert_bitwise(res.history[key[len("hist_"):]], golden[key], f"{case}: {key}")
    assert int(res.valid.sum()) == info["n_valid"]
    assert int(res.ood.sum()) == info["n_ood"]
