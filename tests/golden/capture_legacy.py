"""Record golden outputs from the LEGACY toy code.

Legacy source: SafeRobomimic@origin/toy_domain (b649add0), package ``dev``.
Export it once (never check it out in place) and point ``LEGACY_TOY_ROOT`` at the export:

    git -C <SafeRobomimic> archive origin/toy_domain | tar -x -C <export_dir>
    LEGACY_TOY_ROOT=<export_dir> python tests/golden/capture_legacy.py --out tests/golden

Protocol per case (the refactored ``Sampler.sample`` follows the same order):
    build problem -> seed_everything(seed) -> build sampler (loads ckpt) -> get_action(...) -> save x (and k)

The legacy constructor creates ``trained_models/dataset/{model,plots}`` in the current working
directory, so this script runs inside a throw-away temporary directory.

``--diff OTHER_DIR`` compares the freshly written cases against another capture (bitwise check).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import json
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

LEGACY_COMMIT = "b649add025d53fb41c53f552251d3a0f72e5c020"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Fig 2 obstacles, copied from legacy dev/paper_scripts/02_toy_joint_diffusion.py:207-208
FIG2_OBS_CENTERS = np.array([[-0.50, 0.50], [-0.50, -0.50], [0.50, -0.50], [0.50, 0.50]])
FIG2_OBS_RADIUS = np.array([0.30, 0.50, 0.50, 0.50])

# (case name, task, legacy config section, seed, N, return_histories)
CASES = [
    ("fig2_sequential_seed0", "fig2", "sequential", 0, 16, False),
    ("fig2_sequential_seed1", "fig2", "sequential", 1, 16, False),
    ("fig2_gibbs_seed0", "fig2", "gibbs_exact", 0, 16, False),
    ("fig2_gibbs_seed1", "fig2", "gibbs_exact", 1, 16, False),
    ("fig2_jm2d_seed0", "fig2", "yongxin", 0, 16, False),
    ("fig2_jm2d_seed1", "fig2", "yongxin", 1, 16, False),
    ("fig2_jm2d_hist_seed0", "fig2", "yongxin", 0, 16, True),
    ("fig9_jm2d_seed0", "fig9", "yongxin", 0, 64, False),
    ("fig9_jm2d_hist_seed0", "fig9", "yongxin", 0, 64, True),
    ("fig9_projection_seed0", "fig9", "projection", 0, 64, False),
]

# history keys stored in the golden files (small ones only); all stored as (B, T, ...)
HIST_KEYS_JOINT = ["noisy_samples", "clean_samples", "clean_params", "noisy_params", "scores",
                   "problem_objective", "ood_penalty"]
HIST_KEYS_2D = ["noisy_samples", "clean_samples", "clean_samples_lazy", "clean_samples_variance",
                "scores", "problem_objective", "ood_penalty"]
HIST_KEYS_2D_GUIDE = ["ess"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_config(legacy_root, task, section):
    """Paper-run legacy JSON config with exactly one section enabled."""
    if task == "fig2":
        path = os.path.join(legacy_root, "dev/paper_scripts/configs/02_joint_guide_config.json")
    else:
        path = os.path.join(legacy_root, "dev/exps/base_guide_config.json")
    with open(path) as f:
        cfg = json.load(f)
    for key, val in cfg.items():
        if key != "base":
            val["enabled"] = (key == section)
    return path, cfg


def run_case(legacy, legacy_root, ckpts, name, task, section, seed, n, return_histories):
    _, cfg_dict = legacy_config(legacy_root, task, section)
    if task == "fig2":
        guide_config = legacy["GuidedSamplingConfigJoint"](copy.deepcopy(cfg_dict))
        policy_cls = {"sequential": legacy["SequentialGDPExact"],
                      "gibbs_exact": legacy["GibbsAlternatingGDPExact"],
                      "yongxin": legacy["YongxinExactGDP"]}[section]
        # 1) build problem
        problem = legacy["DonutJointProblem"](obs_centers=FIG2_OBS_CENTERS, obs_radius=FIG2_OBS_RADIUS)
        ckpt = ckpts["fig2"]
    else:
        guide_config = legacy["GuidedSamplingConfig"](copy.deepcopy(cfg_dict))
        policy_cls = {"yongxin": legacy["YongxinMethod"], "projection": legacy["ProjectionDP"]}[section]
        problem = legacy["Donut2DProblem"](**legacy["SCENARIO_DICT"]["great_wall_rectangle"])
        ckpt = ckpts["fig9"]

    # 2) seed, 3) build sampler, 4) sample
    seed_everything(seed)
    policy = policy_cls(model_path=ckpt, problem=problem, **guide_config)
    t0 = time.time()
    out = policy.get_action(batch_size=n, use_constraints=True, return_histories=return_histories)
    elapsed = time.time() - t0

    arrays = {}
    if task == "fig2":
        if return_histories:
            x, k, hist = out
        else:
            x, k = out
        arrays["x"] = np.asarray(x)
        arrays["k"] = np.asarray(k)
        xt, kt = problem.to_tensor(arrays["x"]), problem.to_tensor(arrays["k"])
        n_valid = int(problem.is_valid(xt, kt).sum().item())
        n_ood = int(problem.is_ood(xt).sum().item())
        if return_histories:
            for key in HIST_KEYS_JOINT:
                arrays["hist_" + key] = np.swapaxes(np.array(hist[key]), 0, 1)  # (T,B,..) -> (B,T,..)
    else:
        if return_histories:
            x, hist = out
        else:
            x = out
        arrays["x"] = np.asarray(x)
        xt = problem.to_tensor(arrays["x"])
        n_valid = int(problem.is_valid(xt).sum().item())
        n_ood = int(problem.is_ood(xt).sum().item())
        if return_histories:
            for key in HIST_KEYS_2D:
                arrays["hist_" + key] = np.asarray(hist[key])
            for key in HIST_KEYS_2D_GUIDE:
                arrays["hist_guide_" + key] = np.asarray(hist["guide_log"][key])

    info = dict(task=task, legacy_class=policy_cls.__name__, legacy_section=section, seed=seed, n=n,
                return_histories=return_histories, use_constraints=True,
                n_valid=n_valid, n_ood=n_ood, sample_time_s=round(elapsed, 2),
                legacy_config={"base": cfg_dict["base"], section: cfg_dict[section],
                               **({"gibbs_exact": cfg_dict["gibbs_exact"]} if section == "sequential" else {})})
    return arrays, info


def diff_dirs(dir_a, dir_b, names):
    report = {}
    for name in names:
        a = np.load(os.path.join(dir_a, name + ".npz"))
        b = np.load(os.path.join(dir_b, name + ".npz"))
        entry = {}
        for key in a.files:
            va, vb = a[key], b[key]
            same = va.shape == vb.shape and va.dtype == vb.dtype and va.tobytes() == vb.tobytes()
            maxabs = float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))) if va.shape == vb.shape else float("nan")
            entry[key] = {"bitwise_equal": bool(same), "max_abs_diff": maxabs}
        report[name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "tests", "golden"))
    parser.add_argument("--ckpt-joint", default=os.path.join(REPO_ROOT, "checkpoints", "donut.pth"))
    parser.add_argument("--ckpt-2d", default=os.path.join(REPO_ROOT, "checkpoints", "donut2d.pth"))
    parser.add_argument("--cases", nargs="*", default=None, help="subset of case names (default: all)")
    parser.add_argument("--diff", default=None, help="compare the written cases against this capture dir")
    args = parser.parse_args()

    legacy_root = os.path.abspath(os.environ["LEGACY_TOY_ROOT"])
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpts = {"fig2": os.path.abspath(args.ckpt_joint), "fig9": os.path.abspath(args.ckpt_2d)}

    enable_determinism()
    sys.path.insert(0, legacy_root)
    import dev
    dev_file = os.path.realpath(dev.__file__)
    if os.path.commonpath([dev_file, os.path.realpath(legacy_root)]) != os.path.realpath(legacy_root):
        raise RuntimeError(f"imported legacy package `dev` from {dev_file}, not from LEGACY_TOY_ROOT={legacy_root} "
                           "(another `dev` package shadows it; uninstall it or fix PYTHONPATH)")
    from dev.config.base_config import GuidedSamplingConfig, GuidedSamplingConfigJoint
    from dev.policies.projection_guided_policy import ProjectionDP
    from dev.policies.scg_guided_policy import YongxinMethod
    from dev.policies.scg_guided_policy_joint import (GibbsAlternatingGDPExact, SequentialGDPExact,
                                                      YongxinExactGDP)
    from dev.problems.donut2D_problem import SCENARIO_DICT, Donut2DProblem
    from dev.problems.donut_problem import DonutJointProblem
    legacy = dict(GuidedSamplingConfig=GuidedSamplingConfig, GuidedSamplingConfigJoint=GuidedSamplingConfigJoint,
                  ProjectionDP=ProjectionDP, YongxinMethod=YongxinMethod, GibbsAlternatingGDPExact=GibbsAlternatingGDPExact,
                  SequentialGDPExact=SequentialGDPExact, YongxinExactGDP=YongxinExactGDP,
                  SCENARIO_DICT=SCENARIO_DICT, Donut2DProblem=Donut2DProblem, DonutJointProblem=DonutJointProblem)

    cases = [c for c in CASES if args.cases is None or c[0] in args.cases]
    workdir = tempfile.mkdtemp(prefix="jm2d_legacy_cwd_")
    cwd = os.getcwd()
    os.chdir(workdir)  # legacy constructor writes trained_models/dataset/... into cwd
    infos = {}
    try:
        for name, task, section, seed, n, hist in cases:
            print(f"[capture] {name}", flush=True)
            arrays, info = run_case(legacy, legacy_root, ckpts, name, task, section, seed, n, hist)
            np.savez(os.path.join(out_dir, name + ".npz"), **arrays)
            infos[name] = info
            print(f"[capture] {name}: n_valid={info['n_valid']}/{n} n_ood={info['n_ood']}/{n} "
                  f"({info['sample_time_s']} s)", flush=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    manifest = dict(
        legacy_repo="SafeRobomimic", legacy_ref="origin/toy_domain", legacy_commit=LEGACY_COMMIT,
        legacy_package="dev",
        checkpoints={"fig2": {"file": "checkpoints/donut.pth", "md5": md5(ckpts["fig2"])},
                     "fig9": {"file": "checkpoints/donut2d.pth", "md5": md5(ckpts["fig9"])}},
        protocol="build problem -> seed_everything(seed) -> build sampler (loads ckpt) -> "
                 "get_action(batch_size=N, use_constraints=True, return_histories=H) -> save x (and k)",
        seed_everything="random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all",
        determinism={"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                     "cudnn.deterministic": True, "cudnn.benchmark": False,
                     "use_deterministic_algorithms": "True (warn_only=True)"},
        fig2_problem={"class": "DonutJointProblem", "obs_centers": FIG2_OBS_CENTERS.tolist(),
                      "obs_radius": FIG2_OBS_RADIUS.tolist()},
        fig9_problem={"class": "Donut2DProblem", "scenario": "great_wall_rectangle"},
        history_layout="hist_* arrays are (B, T, ...); legacy joint histories (T, B, ...) were transposed",
        versions={"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
                  "scipy": __import__("scipy").__version__, "diffusers": __import__("diffusers").__version__,
                  "gpu": torch.cuda.get_device_name(0)},
        captured_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        cases=infos,
    )
    manifest_path = os.path.join(out_dir, "manifest.json")
    if args.cases is not None and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        old["cases"].update(infos)
        manifest["cases"] = old["cases"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[capture] wrote {len(infos)} cases + manifest.json to {out_dir}")

    if args.diff:
        report = diff_dirs(out_dir, os.path.abspath(args.diff), [c[0] for c in cases])
        all_equal = all(v["bitwise_equal"] for e in report.values() for v in e.values())
        for name, entry in report.items():
            worst = max(v["max_abs_diff"] for v in entry.values())
            eq = all(v["bitwise_equal"] for v in entry.values())
            print(f"[diff] {name}: bitwise_equal={eq} max_abs_diff={worst:.3e}")
        print(f"[diff] ALL BITWISE EQUAL: {all_equal}")


if __name__ == "__main__":
    main()
