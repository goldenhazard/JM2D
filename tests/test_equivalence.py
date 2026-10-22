"""Refactored planners == original code on every golden case (tests/golden/manifest_*.json).

Golden files were written by ``tests/golden/capture_legacy.py`` running the ORIGINAL evaluation scripts:
  mi       origin/mpd-improved (0d99e9f0)  safediffuser-res, safediffuser-tvs, mpd            4 scenarios
  um       origin/um_dpcc      (a71ff97a)  jm2d, u-steps full (K=20), 0, 1, 2, 5, 10          4 scenarios
  um_dev   origin/um_dpcc      (a71ff97a)  dpcc (scripts/eval_dev.py)                         4 scenarios
  um_apr3  5804c472                         jm2d, full denoising of the Apr-3 code             4 scenarios
Each "steps" case = checkpoint seed 0, trial 0, first 3 planning steps (input observation, executed action, the
full planned batch). Each "episodes" case = 2 full trials (every step + the per-trial metrics). Two captures in
separate processes were bitwise identical.

The cases are re-run through the public API (``jm2d_d3il.experiment.setup`` + ``runner.run_episodes``), in two tiers:

- First planning step (always run): the reset observation, the executed action and the executed plan (the batch
  member whose first action was executed; not for DPCC, see ``PLAN_NOT_COMPARED``) of the first planning step of
  every case, within ``FIRST_STEP_ATOL``; shapes, dtypes and the index of the executed plan in the batch are exact.
  Another CUDA / cuDNN / cuBLAS / OpenBLAS build changes the float32 sampling by ~1e-7..1e-5 per step, and the
  differences grow along an episode (up to 5e-2 in a clean install from the README).
- Bitwise (``JM2D_GOLDEN_BITWISE=1``): every recorded step and the per-trial metrics, bitwise. Needs the numeric
  stack of the captures: torch 2.2.2+cu121 with the cuDNN 8.9.7 / cuBLAS 11.11 of CUDA 11.8 on ``LD_LIBRARY_PATH``
  (``/usr/local/cuda-11.8/lib64``) and conda-forge libopenblas 0.3.29.

``test_resume_*`` re-run trial 1 of an episodes case in a fresh process state without trial 0, which is what a
resumed sweep job does.

Without a CUDA GPU, the checkpoints or the D3IL install the golden tests are skipped; set ``JM2D_REQUIRE_GOLDEN=1``
to make them fail instead (release checks).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import functools
import glob
import json

import numpy as np
import pytest
import torch

import jm2d_d3il  # noqa: F401  (sys.path for third_party/d3il)
from jm2d_d3il.experiment import make_planner, setup
from jm2d_d3il.paths import DEFAULT_CKPT_DIR, THIRD_PARTY_DIR
from jm2d_d3il.results import md5
from jm2d_d3il.runner import run_episodes

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
LEGACY_SCENARIO = {"top-left-hard": "top_left", "top-right-hard": "top_right", "both-hard": "both",
                   "cluttered": "cluttered"}
CASES = {}
MANIFESTS = {}
for _p in sorted(glob.glob(os.path.join(GOLDEN_DIR, "manifest_*.json"))):
    with open(_p) as _f:
        _m = json.load(_f)
    MANIFESTS[os.path.basename(_p)] = _m
    for _name, _info in _m["cases"].items():
        CASES[_name] = dict(_info, mode=_m["mode"], n=_m["n"], lineage=_m["lineage"])
STEP_CASES = sorted(k for k, v in CASES.items() if v["mode"] == "steps")
EPISODE_CASES = sorted(k for k, v in CASES.items() if v["mode"] == "episodes")

# JM2D_REQUIRE_GOLDEN=1: a missing GPU / checkpoint / D3IL install fails the golden tests instead of skipping them.
REQUIRE_GOLDEN = os.environ.get("JM2D_REQUIRE_GOLDEN", "") == "1"
# JM2D_GOLDEN_BITWISE=1: also run the bitwise tier (every step of every case; needs the numeric stack of the captures).
GOLDEN_BITWISE = os.environ.get("JM2D_GOLDEN_BITWISE", "") == "1"
# First-planning-step tier: max |new - golden| (rtol 0) per recorded quantity. Largest difference in a clean install
# from the README (torch's own cuDNN 8.9.2, OpenBLAS 0.3.34): reset observation 4e-16, executed action 1e-5,
# executed plan 2e-5 (actions) / 1.4e-4 (positions).
FIRST_STEP_ATOL = {"obs": 1e-4, "action": 1e-4, "plan_actions": 1e-4, "plan_observations": 1e-3}
# DPCC projects the batch with scipy SLSQP after every denoising step; a ~1e-7 change can move a projection to another
# active set, so in the clean install its first plans differ by up to 1e-3 (executed plan) and 1e-1 (the others)
# while its executed action stays within 3e-6. Its first step is checked through the executed action and the index of
# the executed plan; its plans only in the bitwise tier.
PLAN_NOT_COMPARED = ("dpcc",)


def golden_requirement(ok, reason):
    """Skip the test if ``ok`` is False, or fail it when ``JM2D_REQUIRE_GOLDEN=1``."""
    if not REQUIRE_GOLDEN:
        return pytest.mark.skipif(not ok, reason=reason)

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not ok:
                pytest.fail(f"JM2D_REQUIRE_GOLDEN=1 but {reason}")
            return fn(*args, **kwargs)
        return wrapper
    return decorate


requires_cuda = golden_requirement(torch.cuda.is_available(), "golden cases need a CUDA GPU")
requires_ckpt = golden_requirement(os.path.exists(os.path.join(DEFAULT_CKPT_DIR, "0", "state_best.pt")),
                                   "checkpoints not installed (scripts/download_checkpoints.py)")
_D3IL = os.path.join(THIRD_PARTY_DIR, "d3il")
requires_d3il = golden_requirement(
    os.path.isdir(os.path.join(_D3IL, "environments", "d3il")) and
    os.path.isdir(os.path.join(_D3IL, "environments", "dataset", "data", "avoiding", "data")),
    "the patched D3IL clone + Avoiding data are not installed in third_party/d3il (README, Installation)")
# opt-in, so not turned into a failure by JM2D_REQUIRE_GOLDEN=1
requires_bitwise = pytest.mark.skipif(
    not GOLDEN_BITWISE,
    reason="bitwise golden tier: set JM2D_GOLDEN_BITWISE=1; needs the numeric stack of the captures "
           "(README, Installation: cuDNN 8.9.7 of CUDA 11.8 on LD_LIBRARY_PATH, OpenBLAS 0.3.29)")


class StopAfter(Exception):
    pass


def planner_for(case):
    c = case["case"]
    name = c["planner"]
    if name.startswith("safediffuser-"):
        return make_planner("safediffuser", variant=name.split("-")[1])
    if name == "jm2d":
        return make_planner("jm2d", u_steps=c["u_steps"])
    return make_planner(name)


def run_case(case, trials, n_steps=None):
    c = case["case"]
    planner = planner_for(case)
    task, experiment, env = setup(planner, LEGACY_SCENARIO[c["scenario"]], c["seed"])
    records = {"obs": [], "action": [], "plan_actions": [], "plan_observations": []}

    def on_step(trial, step, obs, action, samples):
        records["obs"].append(np.array(obs, dtype=np.float64, copy=True))
        records["action"].append(np.array(action, copy=True))
        records["plan_actions"].append(np.array(samples.actions, copy=True))
        records["plan_observations"].append(np.array(samples.observations, copy=True))
        if n_steps is not None and len(records["action"]) >= n_steps:
            raise StopAfter()

    per_trial = None
    try:
        per_trial = run_episodes(planner, task, env, trials, on_step=on_step)
    except StopAfter:
        pass
    env.close()
    return {k: np.stack(v) for k, v in records.items()}, per_trial, planner


def assert_bitwise(new, old, what):
    new = np.asarray(new)
    assert new.shape == old.shape, f"{what}: shape {new.shape} != {old.shape}"
    assert new.dtype == old.dtype, f"{what}: dtype {new.dtype} != {old.dtype}"
    if new.tobytes() != old.tobytes():
        diff = np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))
        raise AssertionError(f"{what}: not bitwise equal (max |diff| = {diff:.3e})")


def executed_plans(action, plan_actions):
    """Indices of the batch members whose first planned action is (bitwise) the executed action."""
    return [k for k in range(len(plan_actions)) if plan_actions[k, 0].tobytes() == action.tobytes()]


def first_step_diffs(records, golden, i, compare_plan=True):
    """Planning step 0 of ``records`` vs step ``i`` of ``golden``: max |diff| per ``FIRST_STEP_ATOL`` key.

    Shapes, dtypes and the index of the executed plan in the batch are asserted exactly. The plan quantities are those
    of the executed plan (left out if not ``compare_plan``); the other plans are compared only in the bitwise tier.
    """
    for key in FIRST_STEP_ATOL:
        new, old = np.asarray(records[key][0]), golden[key][i]
        assert new.shape == old.shape, f"{key}: shape {new.shape} != {old.shape}"
        assert new.dtype == old.dtype, f"{key}: dtype {new.dtype} != {old.dtype}"
    executed_new = executed_plans(records["action"][0], records["plan_actions"][0])
    executed_old = executed_plans(golden["action"][i], golden["plan_actions"][i])
    assert len(executed_old) == 1, f"golden step {i}: executed plan not unique ({executed_old})"
    assert executed_new == executed_old, f"executed plan {executed_new} != {executed_old}"
    k = executed_old[0]
    pairs = {"obs": (records["obs"][0], golden["obs"][i]),
             "action": (records["action"][0], golden["action"][i])}
    if compare_plan:
        pairs["plan_actions"] = (records["plan_actions"][0][k], golden["plan_actions"][i][k])
        pairs["plan_observations"] = (records["plan_observations"][0][k], golden["plan_observations"][i][k])
    return {key: float(np.max(np.abs(np.asarray(new, np.float64) - np.asarray(old, np.float64))))
            for key, (new, old) in pairs.items()}


def assert_first_step_close(records, golden, i, what, compare_plan=True):
    try:
        diffs = first_step_diffs(records, golden, i, compare_plan)
    except AssertionError as e:
        raise AssertionError(f"{what}: {e}") from None
    over = {key: d for key, d in diffs.items() if not d <= FIRST_STEP_ATOL[key]}  # `not <=` also catches NaN
    assert not over, f"{what}: max |diff| above FIRST_STEP_ATOL: " + ", ".join(
        f"{key} {d:.3e} > {FIRST_STEP_ATOL[key]:g}" for key, d in over.items())


def test_golden_cases_present():
    """The golden files ship with the repository (44 three-step cases, 2 two-episode cases)."""
    assert len(STEP_CASES) == 44 and len(EPISODE_CASES) == 2, (len(STEP_CASES), len(EPISODE_CASES))
    for name in STEP_CASES + EPISODE_CASES:
        assert os.path.exists(os.path.join(GOLDEN_DIR, name + ".npz")), name


@pytest.fixture(scope="module", autouse=True)
def _determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@requires_ckpt
def test_checkpoints_match_manifest():
    for name, m in MANIFESTS.items():
        for seed, digest in m["checkpoints"].items():
            assert md5(os.path.join(DEFAULT_CKPT_DIR, seed, "state_best.pt")) == digest, (name, seed)


def test_configs_match_legacy():
    """Planner settings == the settings the original runners derived from the variant string / config."""
    for case in CASES.values():
        planner = planner_for(case)
        cfg = case["legacy_config"]
        if case["lineage"] == "mi":
            assert planner.batch_size == 2  # origin/mpd-improved config/avoiding-d3il.py plan.batch_size
            tightened = "tightened" in case["legacy_variant"]
            assert planner.constraint_padding == (cfg["enlarge_constraints"]["avoiding"] if tightened else 0.0)
            assert planner.dynamics_constraints is False  # eval_paper.py passes action_dim=0
        else:
            assert planner.batch_size == 4  # origin/um_dpcc config/avoiding-d3il.py plan.batch_size
            # um generator: enlarge_constraints 0.01 for model_based variants (jm2d), 0.025 otherwise (dpcc)
            expected = 0.01 if planner.name == "jm2d" else 0.025
            assert planner.constraint_padding == cfg["enlarge_constraints"]["avoiding"] == expected
            assert planner.dynamics_constraints is True  # the um_dpcc runner passes action_dim = 2
        assert planner.legacy_variant == case["legacy_variant"]
        assert planner.diffusion_timestep_threshold == cfg["diffusion_timestep_threshold"]


def test_scenarios_match_legacy():
    """configs/scenarios.yaml == the legacy index selection (get_poly_obs_constraints) of every captured config."""
    from jm2d_d3il.tasks import AvoidingTask
    picks = {"top-left-hard": ([0], [3]), "top-right-hard": ([1], [4]), "both-hard": ([2, 3], [5]),
             "cluttered": ([4, 5], [2, 3, 4, 6])}
    for case in CASES.values():
        cfg = case["legacy_config"]
        scen = case["case"]["scenario"]
        hs, ob = picks[scen]
        task = AvoidingTask(LEGACY_SCENARIO[scen])
        assert task.polytopic_constraints == [cfg["halfspace_constraints"]["avoiding-d3il"][i] for i in hs]
        assert task.obstacle_constraints == [cfg["obstacle_constraints"]["avoiding-d3il"][i] for i in ob]
        assert task.init_end_eff_pos == cfg["init_end_eff_pos"]
        assert task.constraint_types == cfg["constraint_types"]


# -- tier 1: the first planning step, within FIRST_STEP_ATOL -------------------------------------------- #
@requires_d3il
@requires_cuda
@requires_ckpt
@pytest.mark.parametrize("name", STEP_CASES + EPISODE_CASES)
def test_golden_first_step(name):
    """Trial 0's first planning step (reset observation, executed action, executed plan) == the original code."""
    case = CASES[name]
    golden = np.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    records, _, planner = run_case(case, trials=[0], n_steps=1)
    assert_first_step_close(records, golden, 0, name, compare_plan=planner.name not in PLAN_NOT_COMPARED)


@requires_d3il
@requires_cuda
@requires_ckpt
@pytest.mark.parametrize("name", EPISODE_CASES)
def test_resume_first_step(name):
    """Trial 1 run on its own: its first planning step == that of trial 1 of the uninterrupted original run."""
    case = CASES[name]
    golden = np.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    n0 = int(case["per_trial"]["n_steps"][0]) + 1  # policy calls of trial 0
    records, _, planner = run_case(case, trials=[1], n_steps=1)
    assert_first_step_close(records, golden, n0, f"{name} (trial 1 alone)",
                            compare_plan=planner.name not in PLAN_NOT_COMPARED)


# -- tier 2 (JM2D_GOLDEN_BITWISE=1): every recorded step, bitwise --------------------------------------- #
@requires_bitwise
@requires_d3il
@requires_cuda
@requires_ckpt
@pytest.mark.parametrize("name", STEP_CASES)
def test_golden_steps(name):
    case = CASES[name]
    golden = np.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    records, _, _ = run_case(case, trials=[0], n_steps=case["n"])
    for key in golden.files:
        assert_bitwise(records[key], golden[key], f"{name}: {key}")


@requires_bitwise
@requires_d3il
@requires_cuda
@requires_ckpt
@pytest.mark.parametrize("name", EPISODE_CASES)
def test_golden_episodes(name):
    case = CASES[name]
    golden = np.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    records, per_trial, _ = run_case(case, trials=list(range(case["n"])))
    for key in ["obs", "action", "plan_actions", "plan_observations"]:
        assert_bitwise(records[key], golden[key], f"{name}: {key}")
    for key, legacy in case["per_trial"].items():
        new = [per_trial[t][key] for t in range(case["n"])]
        assert new == legacy, (name, key, new, legacy)


@requires_bitwise
@requires_d3il
@requires_cuda
@requires_ckpt
@pytest.mark.parametrize("name", EPISODE_CASES)
def test_resume_trial_alone(name):
    """Trial 1 run on its own (as after a resume) == trial 1 of the uninterrupted original run."""
    case = CASES[name]
    golden = np.load(os.path.join(GOLDEN_DIR, name + ".npz"))
    n0 = int(case["per_trial"]["n_steps"][0]) + 1  # policy calls of trial 0
    records, per_trial, _ = run_case(case, trials=[1])
    for key in ["obs", "action", "plan_actions", "plan_observations"]:
        assert_bitwise(records[key], golden[key][n0:], f"{name} (trial 1 alone): {key}")
    for key, legacy in case["per_trial"].items():
        assert per_trial[1][key] == legacy[1], (name, key)
