"""Fast checks of the public API, problems, result I/O and the stored paper results (< 1 min)."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import subprocess
import sys

import numpy as np
import pytest
import torch

from jm2d_toy import samplers, tasks
from jm2d_toy.compare import aggregate, paper_check
from jm2d_toy.config import REPO_ROOT, load_config
from jm2d_toy.data import DonutJointDataset
from jm2d_toy.results import load_results, save_result

PAPER_RESULTS = os.path.join(REPO_ROOT, "results", "paper")


def test_registries():
    assert sorted(samplers.REGISTRY) == ["jm2d", "projection"]
    assert sorted(samplers.JOINT_REGISTRY) == ["gibbs", "jm2d", "sequential"]
    with pytest.raises(KeyError):
        samplers.make("dp", ckpt=load_config("donut_constrained")["checkpoint"])
    assert [tasks.canonical(f) for f in ("fig2", "fig9")] == list(tasks.TASKS)  # paper figure ids


def test_donut_dataset_pairing_matches_legacy():
    """setdiff1d replaces the legacy O(n^2) list comprehension; same indices, same RNG use."""
    rng = np.random.RandomState(3)
    half = rng.choice(200, 100, replace=False)
    legacy = np.array([i for i in range(200) if i not in half])
    assert np.array_equal(np.setdiff1d(np.arange(200), half), legacy)
    ds = DonutJointDataset(100, rng=np.random.RandomState(0))
    assert ds.all_samples.shape == (100, 4)
    r = np.linalg.norm(ds.all_samples.reshape(-1, 2), axis=-1)
    assert np.all((r >= 0.6) & (r <= 1.0))


@pytest.mark.gpu
def test_problem_functions():
    constrained = tasks.make("donut_constrained")
    x = torch.tensor([[-0.8, 0.0], [0.9, 0.0], [1.2, 0.0]], device="cuda")
    assert constrained.is_valid(x).tolist() == [False, True, True]   # wall covers x <= 0.8
    assert constrained.is_ood(x).tolist() == [False, False, True]    # donut 0.6 <= r <= 1
    joint = tasks.make("donut_joint")
    x = torch.tensor([[0.0, 0.9, 0.0, 0.95], [0.7, 0.7, -0.7, -0.7]], device="cuda")
    k = torch.tensor([[1.0, 0.25], [0.0, 0.0]], device="cuda")       # k -> (0, 0.6) and the origin
    # segments (0, 0.6) -> (0, 0.9 / 0.95) stay clear of the obstacles; origin -> (+-0.7, +-0.7) cross them
    assert joint.is_valid(x, k).tolist() == [True, False]
    assert joint.is_ood(x).tolist() == [False, False]
    assert joint.global_optimal_value() == joint.global_optimal_value()  # seeded


@pytest.mark.gpu
def test_tiny_runs_and_result_roundtrip(tmp_path):
    res2 = samplers.from_config("donut_joint", "jm2d", num_mc_samples=4, num_param_samples=4).sample(
        tasks.make("donut_joint"), n=3, seed=0)
    assert res2.x.shape == (3, 4) and res2.k.shape == (3, 2) and res2.valid.shape == (3,)
    again = samplers.from_config("donut_joint", "jm2d", num_mc_samples=4, num_param_samples=4).sample(
        tasks.make("donut_joint"), n=3, seed=0)
    assert np.array_equal(res2.x, again.x) and np.array_equal(res2.k, again.k)  # reproducible per seed
    res9 = samplers.from_config("donut_constrained", "jm2d", num_mc_samples=4).sample(
        tasks.make("donut_constrained"), n=3, seed=1)
    assert res9.x.shape == (3, 2) and res9.k is None
    for res in (res2, res9):
        save_result(res, str(tmp_path))
    rows = load_results(str(tmp_path))
    assert {(r["task"], r["sampler"], r["run_id"]) for r in rows} == {("donut_joint", "jm2d", "seed0"),
                                                                     ("donut_constrained", "jm2d", "seed1")}
    row2 = [r for r in rows if r["task"] == "donut_joint"][0]
    assert row2["n_compatible"] == int(res2.valid.sum()) and np.array_equal(row2["x"], res2.x)


@pytest.mark.gpu
def test_run_and_compare_scripts(tmp_path):
    out = str(tmp_path / "runs")
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "run.py"), "--task", "donut_joint",
                    "--planner", "sequential", "--n", "4", "--seeds", "0-1", "--out", out], check=True)
    assert sorted(os.listdir(os.path.join(out, "donut_joint", "sequential"))) == [
        "seed0.json", "seed0.npz", "seed1.json", "seed1.npz"]
    proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "compare.py"), "--results", out,
                           "--task", "donut_joint", "--out", str(tmp_path / "donut_joint.png")],
                          check=True, capture_output=True, text=True)
    assert "sequential" in proc.stdout and os.path.exists(tmp_path / "donut_joint.png")


def test_cli_flags_and_deprecated_aliases():
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    import compare as compare_cli
    import run as run_cli
    new = run_cli.build_parser().parse_args(["--task", "donut_joint", "--planner", "jm2d", "gibbs", "--seeds", "0-2",
                                             "5"])
    old = run_cli.build_parser().parse_args(["--task", "fig2", "--sampler", "jm2d", "gibbs", "--seed", "0-2,5"])
    assert new.task == old.task == "donut_joint" and new.planner == old.planner == ["jm2d", "gibbs"]
    assert run_cli.parse_seeds(",".join(new.seeds)) == run_cli.parse_seeds(",".join(old.seeds)) == [0, 1, 2, 5]
    new = compare_cli.build_parser().parse_args(["--paper", "--task", "donut_constrained", "--out", "x.png",
                                                 "--check-paper"])
    old = compare_cli.build_parser().parse_args(["--from-paper-results", "--figure", "fig9", "--check-paper"])
    assert (new.paper, new.task, new.check_paper) == (old.paper, old.task, old.check_paper) == \
        (True, "donut_constrained", True)
    with pytest.raises(SystemExit):  # --task is a task name, never a path
        compare_cli.main(["--task", "results/x.png"])


def test_compare_paper_results_on_cpu(tmp_path):
    """``compare.py --paper`` (table, paper check, figure) runs without a GPU."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for task in tasks.TASKS:
        png = tmp_path / f"{task}.png"
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "compare.py"), "--paper",
                               "--task", task, "--check-paper", "--out", str(png)],
                              check=True, capture_output=True, text=True, env=env)
        assert "FAIL" not in proc.stdout and png.exists()


def test_paper_results_match_targets():
    """results/paper (stored legacy runs) reproduce the paper counts exactly where the code is shipped."""
    agg2 = aggregate(load_results(PAPER_RESULTS, task="donut_joint"), "donut_joint")
    rows, checks = paper_check(agg2, "donut_joint")
    assert len(rows) == 3 and all(r[-1] for r in rows) and all(checks.values())
    agg9 = aggregate(load_results(PAPER_RESULTS, task="donut_constrained"), "donut_constrained")
    rows, _ = paper_check(agg9, "donut_constrained")
    assert {(r[0], r[1]) for r in rows} == {("projection", "n_valid"), ("projection", "n_ood")}
    assert all(r[-1] for r in rows)
