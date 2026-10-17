"""Refactored planners == legacy planners, bitwise, on every golden case (tests/golden/manifest.json).

Golden files: ``tests/golden/capture_legacy.py`` on SafeRobomimic@robomimic (8b2371d2; dp/rail/jm2d) and
SafeRobomimic@paper/gibbs (be46370b; gibbs): epoch-1900 DP checkpoint, env seeds 0 and 1, first 50 control steps.
Each case is re-run here in a fresh process (``tests/golden/run_refactored.py``: same protocol and recorder) and
every recorded array (state, executed action, intervention flag, backup parameter, nominal DP actions, nominal
plan, objective) and every per-episode statistic must be identical. Two legacy captures were bitwise identical.

Needs a CUDA GPU and the epoch-1900 checkpoint (``JM2D_MAZE_CKPT`` or ``checkpoints/maze2d_large_ep1900.pth``);
without them the golden tests are skipped, or fail with ``JM2D_REQUIRE_GOLDEN=1`` (use this for release checks).
``JM2D_TEST_JOBS`` (default 3) cases run concurrently; the full set takes about 3-4 min on an RTX 4090.

The goldens were captured with the default CPU thread count. RAIL rollouts change at round-off level with the
thread count (IPOPT, CPU zonotope algebra), so the golden subprocesses run without OMP_NUM_THREADS /
MKL_NUM_THREADS / OPENBLAS_NUM_THREADS even if they are set in the calling environment.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
GOLDEN_DIR = os.path.join(HERE, "golden")
sys.path.insert(0, REPO_ROOT)

with open(os.path.join(GOLDEN_DIR, "manifest.json")) as _f:
    MANIFEST = json.load(_f)
CASES = sorted(MANIFEST["cases"])
CKPT = os.environ.get("JM2D_MAZE_CKPT", os.path.join(REPO_ROOT, "checkpoints", "maze2d_large_ep1900.pth"))


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


REQUIRE_GOLDEN = os.environ.get("JM2D_REQUIRE_GOLDEN", "0") == "1"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def skip_or_fail(reason):
    """Skip, or fail when JM2D_REQUIRE_GOLDEN=1 (a release check must not silently skip the golden cases)."""
    if REQUIRE_GOLDEN:
        pytest.fail(f"{reason} (JM2D_REQUIRE_GOLDEN=1)")
    pytest.skip(reason)


def require_gpu_and_ckpt():
    if not _cuda_available():
        skip_or_fail("golden cases need a CUDA GPU")
    if not os.path.exists(CKPT):
        skip_or_fail(f"golden cases need the epoch-1900 checkpoint ({CKPT}; scripts/download_checkpoints.py)")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def test_checkpoint_md5():
    if not os.path.exists(CKPT):
        skip_or_fail(f"checkpoint not downloaded ({CKPT})")
    from jm2d_maze.results import md5
    assert md5(CKPT) == MANIFEST["checkpoint"]["md5"]


def test_config_matches_legacy():
    """configs/maze2d_large.json == the paper's config_ws.json (key for key) up to output_dir and the mode name."""
    from safediffusion.algo.jm2d_policy import GUIDANCE_MODES
    with open(os.path.join(REPO_ROOT, "configs", "maze2d_large.json")) as f:
        new = _flatten(json.load(f))
    for name, mode in [("config_ws.json", "jm2d"), ("config_ws_gibbs.json", "gibbs")]:
        with open(os.path.join(GOLDEN_DIR, "legacy_configs", name)) as f:
            old = _flatten(json.load(f))
        assert set(new) == set(old), name
        for key in old:
            if key == "safety.output_dir":
                continue
            if key == "safety.align.mode":
                assert GUIDANCE_MODES[old[key]] == mode
                continue
            assert new[key] == old[key], (name, key)


def test_load_config_sets_planner_mode():
    from jm2d_maze.config import load_config
    assert load_config(planner="gibbs").safety.align.mode == "gibbs"
    assert load_config(planner="jm2d").safety.align.mode == "jm2d"
    assert load_config(planner="dp").safety.align.N_sample_action == 16


@pytest.fixture(scope="module")
def refactored_outputs(request):
    """Run the selected golden cases (all, or those kept by ``-k``) through the refactored code (fresh process per
    case, default CPU threads)."""
    require_gpu_and_ckpt()
    selected = [item.callspec.params["case"] for item in request.session.items
                if getattr(item, "originalname", None) == "test_golden_equivalence" and hasattr(item, "callspec")]
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    out_dir = tempfile.mkdtemp(prefix="jm2d_maze_equiv_")
    jobs = int(os.environ.get("JM2D_TEST_JOBS", "3"))
    pending, running, logs = [c for c in CASES if c in selected], [], {}
    while pending or running:
        while pending and len(running) < jobs:
            case = pending.pop(0)
            log = open(os.path.join(out_dir, case + ".log"), "w")
            cmd = [sys.executable, os.path.join(GOLDEN_DIR, "run_refactored.py"), "--case", case, "--ckpt", CKPT,
                   "--out", out_dir]
            running.append((case, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env),
                            log))
            logs[case] = log.name
        for item in list(running):
            if item[1].poll() is not None:
                item[2].close()
                running.remove(item)
        time.sleep(0.5)
    return out_dir, logs


def assert_bitwise(new, old, what):
    assert new.shape == old.shape, f"{what}: shape {new.shape} != {old.shape}"
    assert new.dtype == old.dtype, f"{what}: dtype {new.dtype} != {old.dtype}"
    if new.tobytes() != old.tobytes():
        diff = np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))
        raise AssertionError(f"{what}: not bitwise equal (max |diff| = {diff:.3e})")


@pytest.mark.parametrize("case", CASES)
def test_golden_equivalence(case, refactored_outputs):
    out_dir, logs = refactored_outputs
    path = os.path.join(out_dir, case + ".npz")
    assert os.path.exists(path), f"refactored run failed, see {logs[case]}"
    new = np.load(path)
    old = np.load(os.path.join(GOLDEN_DIR, case + ".npz"))
    assert sorted(new.files) == sorted(old.files)
    for key in old.files:
        assert_bitwise(new[key], old[key], f"{case}: {key}")
    with open(os.path.join(out_dir, case + ".json")) as f:
        info = json.load(f)
    assert info["stats"] == MANIFEST["cases"][case]["stats"]
