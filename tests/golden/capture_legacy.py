"""Record golden outputs from the LEGACY D3IL code (paper Table 2 / Table 3 planners).

Legacy sources (exported with ``git archive``, never checked out in place):

    lineage  ref                    commit    used for
    mi       origin/mpd-improved    0d99e9f0  safediffuser (ReS, TVS), mpd           scripts/eval_paper.py
    um       origin/um_dpcc         a71ff97a  jm2d (u-steps = K_value)               scripts/eval_dev_Kfunc.py
    um_dev   origin/um_dpcc         a71ff97a  dpcc (the runner of the stored DPCC)     scripts/eval_dev.py
    um_apr3  5804c472               5804c472  jm2d, full denoising (cross-check)     scripts/eval_dev.py

    git -C <SafeRobomimic> archive origin/mpd-improved | tar -x -C <export_dir>
    LEGACY_ROOT=<export_dir> python tests/golden/capture_legacy.py --lineage mi --workdir <tmp> --out tests/golden

The legacy evaluation ``main()`` is run unchanged. ``Policy.__call__`` is wrapped to record, for every
planning step, the observation passed in, the executed action and the full batch of planned trajectories.
``--steps N`` stops after N planning steps of trial 0 (seed 0); ``--episodes N`` runs N full trials and also
keeps the per-trial arrays written by the legacy script (``n_success``, ``n_violations``, ...).

The working directory must contain ``logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion/<seed>/``
(the checkpoints; a directory of symlinks is enough). The legacy code writes its result files below
``logs/avoiding-d3il/plans*`` in that directory.

Scenario geometry: the Cluttered obstacle #6 is set to the canonical (0.5, 0.20), r = 0.02 for
every lineage; ``um`` and ``um_apr3`` already use it, the ``mi`` per-seed configs used (0.5, 0.26), r = 0.05.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
import types

import numpy as np
import yaml

LINEAGES = {
    "mi": dict(ref="origin/mpd-improved", commit="0d99e9f08968bdecb2207eb996272dd795918b19",
               script="scripts/eval_paper.py", base_config="config/SafeDiffuser_configs/avoiding_{scenario}__seed_{seed}.yaml"),
    "um": dict(ref="origin/um_dpcc", commit="a71ff97a298f8e386b0377b281f4570aa9000f26",
               script="scripts/eval_dev_Kfunc.py", base_config="config/projection_eval_dev.yaml"),
    "um_dev": dict(ref="origin/um_dpcc", commit="a71ff97a298f8e386b0377b281f4570aa9000f26",
                   script="scripts/eval_dev.py", base_config="config/projection_eval_dev.yaml"),
    "um_apr3": dict(ref="5804c472", commit="5804c47253911f91821ec83227948dac5eaa5e0c",
                    script="scripts/eval_dev.py", base_config="config/projection_eval_dev.yaml"),
}

# legacy variant string per public planner name
VARIANTS = {
    "dpcc": "dpcc-c-tightened",
    "safediffuser-res": "invariance-ReS-dpcc-c",
    "safediffuser-tvs": "invariance-TVS-dpcc-c",
    "mpd": "mpd-gradient",
    "jm2d": "model_based-dpcc-c-tightened",
}
SCENARIOS = ["top-left-hard", "top-right-hard", "both-hard", "cluttered"]
CLUTTERED_OBSTACLE_6 = {"type": "sphere_outside", "dimensions": ["x", "y"], "center": [0.5, 0.2], "radius": 0.02}
FULL_K = 20  # K_value >= n_diffusion_steps - 1 means full denoising in the um code (end_timestep = 0)


class StopCapture(Exception):
    pass


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cases(lineage, mode):
    cases = []
    if lineage == "mi":
        planners = ["safediffuser-res", "safediffuser-tvs", "mpd"]  # dpcc: lineage um_dev (stored DPCC runner)
        for p in planners:
            for s in SCENARIOS:
                cases.append(dict(planner=p, scenario=s, seed=0, u_steps=None))
    elif lineage == "um":
        u_list = ["full", 0, 1, 2, 5, 10] if mode == "steps" else ["full"]
        for u in u_list:
            for s in SCENARIOS:
                cases.append(dict(planner="jm2d", scenario=s, seed=0, u_steps=u))
    elif lineage == "um_dev":
        for s in SCENARIOS:
            cases.append(dict(planner="dpcc", scenario=s, seed=0, u_steps=None))
    elif lineage == "um_apr3":
        for s in SCENARIOS:
            cases.append(dict(planner="jm2d", scenario=s, seed=0, u_steps="full"))
    return cases


def case_name(c, mode):
    u = "" if c["u_steps"] is None else f"_u{c['u_steps']}"
    tag = "steps" if mode == "steps" else "episodes"
    return f"{c['lineage']}_{c['planner']}_{c['scenario']}{u}_seed{c['seed']}_{tag}"


def legacy_config(legacy_root, lineage, case, n_trials):
    info = LINEAGES[lineage]
    path = os.path.join(legacy_root, info["base_config"].format(scenario=case["scenario"], seed=case["seed"]))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["seeds"] = [case["seed"]]
    cfg["avoiding_halfspace_variants"] = [case["scenario"]]
    cfg["projection_variants"] = [VARIANTS[case["planner"]]]
    cfg["n_trials"] = n_trials
    cfg["write_to_file"] = True
    obstacles = cfg["obstacle_constraints"]["avoiding-d3il"]
    obstacles[6] = copy.deepcopy(CLUTTERED_OBSTACLE_6)
    if lineage == "um":
        cfg["K_values"] = [FULL_K if case["u_steps"] == "full" else int(case["u_steps"])]
        # um generator (scripts/eval_dev_Kfunc.py, commented block): 0.01 for model_based variants, else 0.025
        cfg["enlarge_constraints"]["avoiding"] = 0.01
    if lineage == "um_dev":
        # um generator: enlarge_constraints = 0.025 for variants without 'model_based'
        cfg["enlarge_constraints"]["avoiding"] = 0.025
    if lineage == "um_apr3":
        cfg["enlarge_constraints"]["avoiding"] = 0.01
    return path, cfg


def install_recorder(n_stop):
    from diffuser.sampling import policies
    records = []
    orig_call = policies.Policy.__call__

    def recording_call(self, conditions, *args, **kwargs):
        obs = np.array(conditions[0], dtype=np.float64, copy=True)
        out = orig_call(self, conditions, *args, **kwargs)
        action, samples = out[0], out[1]
        records.append(dict(obs=obs, action=np.array(action, copy=True),
                            plan_actions=np.array(samples.actions, copy=True),
                            plan_observations=np.array(samples.observations, copy=True)))
        if n_stop is not None and len(records) >= n_stop:
            raise StopCapture()
        return out

    policies.Policy.__call__ = recording_call
    return records, orig_call


def run_case(legacy_root, lineage, case, mode, n, workdir, legacy_mod):
    from diffuser.sampling import policies
    n_trials = 1 if mode == "steps" else n
    src_cfg, cfg = legacy_config(legacy_root, lineage, case, n_trials)
    cfg_path = os.path.join(workdir, "capture_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    records, orig_call = install_recorder(n if mode == "steps" else None)
    t0 = time.time()
    try:
        if lineage == "mi":
            legacy_mod.main(argparse.Namespace(config=cfg_path))
        elif lineage in ("um", "um_dev"):
            legacy_mod.main(cfg)
        else:  # um_apr3 reads config/projection_eval_dev.yaml relative to the working directory
            os.makedirs(os.path.join(workdir, "config"), exist_ok=True)
            with open(os.path.join(workdir, "config", "projection_eval_dev.yaml"), "w") as f:
                yaml.safe_dump(cfg, f)
            legacy_mod.main()
    except StopCapture:
        pass
    finally:
        policies.Policy.__call__ = orig_call
    elapsed = time.time() - t0

    arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
    per_trial = None
    if mode == "episodes":
        variant = VARIANTS[case["planner"]]
        sub = {"mi": "results_final", "um": f"{cfg.get('K_values', [''])[0]}/results_dev", "um_dev": "results_dev",
               "um_apr3": "results_dev"}[lineage]
        prefix = {"mi": "plans_guide_all_steps", "um": "plans_Kfunc", "um_dev": "plans_Kfunc", "um_apr3": "plans"}[lineage]
        res = os.path.join(workdir, "logs/avoiding-d3il", prefix, "H8_K20_Dmodels.GaussianDiffusion",
                           str(case["seed"]), sub, f"halfspace_{case['scenario']}", variant + ".json")
        with open(res) as f:
            legacy_json = json.load(f)
        per_trial = {k: legacy_json[k] for k in ["n_success", "n_success_and_constraints", "n_steps",
                                                    "n_violations", "total_violations", "collision_free_completed"]}
        for k, v in per_trial.items():
            arrays["trial_" + k] = np.asarray(v)
    return arrays, per_trial, cfg, src_cfg, elapsed


def versions():
    import scipy
    import torch
    import diffusers
    import mujoco
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "diffusers": diffusers.__version__, "mujoco": mujoco.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=sorted(LINEAGES), required=True)
    ap.add_argument("--workdir", required=True, help="directory holding logs/avoiding-d3il/diffusion/...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3, help="planning steps of trial 0 per case")
    ap.add_argument("--episodes", type=int, default=0, help="if > 0: run this many full trials instead")
    ap.add_argument("--cases", default="", help="comma list planner:scenario[:u] to restrict the default cases")
    args = ap.parse_args()

    legacy_root = os.path.abspath(os.environ["LEGACY_ROOT"])
    workdir = os.path.abspath(args.workdir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    mode = "episodes" if args.episodes > 0 else "steps"
    n = args.episodes if mode == "episodes" else args.steps

    # legacy imports: the snapshot must win over any other checkout on sys.path
    sys.path.insert(0, legacy_root)
    if args.lineage in ("um", "um_dev", "um_apr3"):
        # utils/submitit_utils.py needs `submitit` (Slurm only; never called on the evaluation path)
        stub = types.ModuleType("utils.submitit_utils")
        stub.create_executor = None
        sys.modules["utils.submitit_utils"] = stub
    sys.argv = [sys.argv[0]]  # the legacy utils.Parser parses sys.argv
    os.chdir(workdir)

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    script = os.path.join(legacy_root, LINEAGES[args.lineage]["script"])
    spec = importlib.util.spec_from_file_location("legacy_eval", script)
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)

    # The legacy per-trial plots index ``ax_all[iter, variant_idx]``, which assumes >= 2 variants per config.
    # Keep 2-D axes arrays for multi-panel figures (plotting only; no effect on planning or metrics).
    _orig_subplots = legacy_mod.plt.subplots

    def _subplots(nrows=1, ncols=1, **kw):
        kw.setdefault("squeeze", nrows * ncols == 1)
        return _orig_subplots(nrows, ncols, **kw)

    legacy_mod.plt.subplots = _subplots

    import diffuser
    import d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding as avoiding_mod
    module_paths = {"diffuser": diffuser.__file__, "avoiding_env": avoiding_mod.__file__, "legacy_script": script,
                    "config_module": importlib.import_module("config.avoiding-d3il").__file__}
    print("[capture] module paths:", json.dumps(module_paths, indent=1), flush=True)
    assert diffuser.__file__.startswith(legacy_root), diffuser.__file__
    if os.environ.get("D3IL_PARENT"):  # e.g. <repo>/third_party: the patched D3IL clone must be the one imported
        assert avoiding_mod.__file__.startswith(os.path.abspath(os.environ["D3IL_PARENT"])), avoiding_mod.__file__

    cases = default_cases(args.lineage, mode)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if f"{c['planner']}:{c['scenario']}" in keep
                 or f"{c['planner']}:{c['scenario']}:{c['u_steps']}" in keep]

    manifest_path = os.path.join(out_dir, f"manifest_{args.lineage}_{mode}.json")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def portable(path):
        for root, tag in ((legacy_root, "<LEGACY_ROOT>"), (repo_root, "<repo>")):
            if path.startswith(root):
                return tag + path[len(root):]
        return path

    manifest = {"lineage": args.lineage, **LINEAGES[args.lineage], "mode": mode, "n": n,
                "module_paths": {k: portable(v) for k, v in module_paths.items()}, "versions": versions(),
                "cases": {}}
    ckpt = os.path.join(workdir, "logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion")
    manifest["checkpoints"] = {s: md5(os.path.join(ckpt, s, "state_best.pt")) for s in sorted(os.listdir(ckpt))}
    for c in cases:
        c["lineage"] = args.lineage
        name = case_name(c, mode)
        print(f"[capture] {name}", flush=True)
        arrays, per_trial, cfg, src_cfg, elapsed = run_case(legacy_root, args.lineage, c, mode, n, workdir, legacy_mod)
        npz = os.path.join(out_dir, name + ".npz")
        np.savez(npz, **arrays)
        manifest["cases"][name] = dict(case=c, legacy_variant=VARIANTS[c["planner"]],
                                       legacy_base_config=os.path.relpath(src_cfg, legacy_root),
                                       n_policy_calls=int(arrays["action"].shape[0]),
                                       per_trial=per_trial, time_s=round(elapsed, 1), npz_md5=md5(npz),
                                       legacy_config=cfg)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print("[capture] done", manifest_path, flush=True)


if __name__ == "__main__":
    main()
