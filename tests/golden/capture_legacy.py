"""Record golden PointMaze rollouts from the LEGACY code.

Legacy sources (export them with ``git archive``; never check them out in place):

    git -C <SafeRobomimic> archive robomimic   | tar -x -C <rail_release_export>   # 8b2371d2 (= origin/rail_release)
    git -C <SafeRobomimic> archive paper/gibbs | tar -x -C <gibbs_export>          # be46370b (Gibbs baseline)

    LEGACY_MAZE_ROOT=<rail_release_export> LEGACY_GIBBS_ROOT=<gibbs_export> \
    LEGACY_CONFIG_DIR=<SafeRobomimic>/diffusion_policy_trained_models/maze2d_large \
        python tests/golden/capture_legacy.py --ckpt <model_epoch_1900.pth> --out tests/golden

``dp``/``rail``/``jm2d`` cases run on the rail_release export, ``gibbs`` cases on the paper/gibbs export
(the only ref with ``multistep_gibbs_sampling``). Every case runs in its own python process with the legacy
root first on ``sys.path``; the resolved ``robomimic``/``safediffusion``/``d4rl`` files are recorded.

Protocol per case (identical to the legacy ``run_rail_v2.py`` + ``rollout_random_seed``):

    set_random_seed(42) -> policy_and_env_from_checkpoint_and_config(...)
    for seed in seeds:  set_random_seed(seed, deterministic_torch=False) -> env.reset() -> rollout(horizon)

The rollout is the legacy ``rollout`` function itself; only the video writer and ``env.render`` are replaced
by no-ops (rendering reads the simulator and never changes it), and unseeded gym spaces are seeded with 0
(``--no-seed-gym-spaces`` to disable): gym's ``MujocoEnv.__init__`` steps every new env once with an action
drawn from OS entropy, which perturbs ``SafeMazeEnv.robotqpos_to_worldpos`` (hence the DP goal) by O(1e-3) and
made two otherwise identical legacy captures differ. A recorder wrapped around the policy's
``__call__`` stores, per control step, the state, the executed action, the intervention flag and the backup
parameter, and, per replanning step, the nominal (DP) actions, the nominal plan and the objective.

``--diff OTHER_DIR`` compares the freshly written cases with another capture (bitwise).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEGACY_REFS = {
    "maze": {"ref": "robomimic (= origin/rail_release)", "commit": "8b2371d25525aca3554c5fd7d3d2c6b563b2c317"},
    "gibbs": {"ref": "paper/gibbs", "commit": "be46370b7c9a546f02870765e199e7fbf47f96b3"},
}
# planner -> (legacy root, legacy policy_mode, legacy config file)
LEGACY_PLANNER = {
    "dp": ("maze", "diffusion", "config_ws.json"),
    "rail": ("maze", "rail", "config_ws.json"),
    "jm2d": ("maze", "rail_aligned", "config_ws.json"),
    "gibbs": ("gibbs", "rail_aligned", "config_ws_gibbs.json"),
}
LEGACY_BACKUP = {"hq": "simple2d", "xpyp": "simple2d_xpyp", "xmym": "simple2d_xmym"}
LEGACY_MAZE = {0.0: "large", 0.05: "large_pad005", 0.10: "large_pad010", 0.15: "large_pad015",
               0.20: "large_pad020", 0.25: "large_pad025"}

# (case name, planner, pad, backup, env seeds, horizon)
CASES = [(f"{p}_pad{int(round(w * 100)):03d}_{b}", p, w, b, [0, 1], 50)
         for p in ["dp", "rail", "jm2d"] for w in [0.0, 0.25] for b in ["hq"]]
CASES += [(f"{p}_pad020_xpyp", p, 0.20, "xpyp", [0, 1], 50) for p in ["rail", "jm2d"]]
CASES += [(f"gibbs_pad{int(round(w * 100)):03d}_hq", "gibbs", w, "hq", [0, 1], 50) for w in [0.0, 0.25]]
# added in review (2026-10-16): the x-y- backup of Table 1, and JM2D-HQ with interventions (env seed 2 of the
# pad-0.25 chain intervenes from step 32; seeds 0 and 1 run first so that seed 2 gets its stored-run start state)
CASES += [(f"{p}_pad020_xmym", p, 0.20, "xmym", [0, 1], 50) for p in ["rail", "jm2d"]]
CASES += [("jm2d_pad025_hq_intervene", "jm2d", 0.25, "hq", [0, 1, 2], 50)]
CASE_BY_NAME = {c[0]: c for c in CASES}

GYM_SPACE_SEED = 0
STEP_KEYS = ["state", "action", "intervened", "action_idx", "k_backup"]
REPLAN_KEYS = ["replan_step", "nominal_actions", "nominal_plan_x", "objective"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _NullWriter:
    def append_data(self, img):
        pass

    def close(self):
        pass


def _to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def install_recorder(policy, episodes):
    """Wrap ``type(policy).__call__``/``start_episode`` to record per-step outputs (read-only)."""
    cls = type(policy)
    orig_call, orig_start = cls.__call__, cls.start_episode

    def start_episode(self):
        episodes.append({"steps": [], "replans": []})
        return orig_start(self)

    def call(self, obs_dict, goal_dict=None, **kwargs):
        state = np.array(obs_dict["flat"][-1], dtype=np.float64, copy=True)
        t0 = time.time()
        ac, ctrl = orig_call(self, obs_dict, goal_dict, **kwargs)
        dt = time.time() - t0
        ep = episodes[-1]
        step = len(ep["steps"])
        k = None
        if self._backup_plan is not None:
            k = _to_np(self._backup_plan.get_trajectory_parameter()).astype(np.float64)
        ep["steps"].append(dict(state=state, action=None if ac is None else _to_np(ac).astype(np.float64),
                                intervened=bool(self.intervened), action_idx=int(self.action_idx), k_backup=k))
        if ac is not None and self.action_idx == 1:
            plan_x = None if self.nominal_plan is None else _to_np(self.nominal_plan.x_des).astype(np.float64)
            obj = self.opt_info["objective"] if self.opt_info is not None else np.nan
            ep["replans"].append(dict(replan_step=step, nominal_actions=_to_np(self.nominal_actions).astype(np.float64),
                                      nominal_plan_x=plan_x, objective=float(_to_np(obj)), time_s=dt))
        return ac, ctrl

    cls.__call__ = call
    cls.start_episode = start_episode


def episodes_to_arrays(episodes):
    """Stack recorder output to ``ep<i>_<key>`` arrays."""
    arrays = {}
    for i, ep in enumerate(episodes):
        steps = [s for s in ep["steps"] if s["action"] is not None]
        arrays[f"ep{i}_state"] = np.stack([s["state"] for s in steps])
        arrays[f"ep{i}_action"] = np.stack([s["action"] for s in steps])
        arrays[f"ep{i}_intervened"] = np.array([s["intervened"] for s in steps])
        arrays[f"ep{i}_action_idx"] = np.array([s["action_idx"] for s in steps])
        arrays[f"ep{i}_k_backup"] = np.stack([s["k_backup"] for s in steps])
        arrays[f"ep{i}_replan_step"] = np.array([r["replan_step"] for r in ep["replans"]])
        arrays[f"ep{i}_nominal_actions"] = np.stack([r["nominal_actions"] for r in ep["replans"]])
        arrays[f"ep{i}_nominal_plan_x"] = np.stack([r["nominal_plan_x"] for r in ep["replans"]])
        arrays[f"ep{i}_objective"] = np.array([r["objective"] for r in ep["replans"]])
    return arrays


def run_worker(args):
    name, planner, pad, backup, seeds, horizon = CASE_BY_NAME[args.worker]
    root_key, policy_mode, cfg_name = LEGACY_PLANNER[planner]
    legacy_root = os.path.abspath(os.environ["LEGACY_MAZE_ROOT" if root_key == "maze" else "LEGACY_GIBBS_ROOT"])
    config_path = os.path.join(os.environ["LEGACY_CONFIG_DIR"], cfg_name)
    sys.path.insert(0, legacy_root)

    import multiprocessing as mp
    mp.cpu_count = lambda: args.workers  # legacy pool size = min(cpu_count, N_sample_action)

    import torch
    import gym
    import d4rl
    import robomimic
    import safediffusion.algo.safe_diffusion_policy as SDP
    import safediffusion_d4rl.pointmaze.utils as MazeUtils
    from safediffusion.utils.rand_utils import set_random_seed

    modules = {"robomimic": robomimic.__file__, "safediffusion": SDP.__file__, "d4rl": d4rl.__file__,
               "maze_utils": MazeUtils.__file__}
    for key in ("robomimic", "safediffusion", "maze_utils"):
        assert modules[key].startswith(legacy_root), f"{key} resolved to {modules[key]}, not the legacy export"

    MazeUtils.imageio.get_writer = lambda *a, **k: _NullWriter()
    if args.seed_gym_spaces:
        # gym's MujocoEnv.__init__ steps the new env once with action_space.sample() from an UNSEEDED space
        # (OS entropy). That random step shifts SafeMazeEnv.robotqpos_to_worldpos by O(1e-3), hence the goal
        # fed to the DP, so two legacy runs with the same seeds differ. Seed unseeded spaces with 0 instead.
        def _np_random(self):
            if self._np_random is None:
                self.seed(GYM_SPACE_SEED)
            return self._np_random
        gym.spaces.Space.np_random = property(_np_random)
    if args.deterministic:
        # Same RNG seeding as the legacy set_random_seed; only the GPU determinism flags differ (the legacy
        # per-episode call set_random_seed(seed, deterministic_torch=False) allows non-deterministic cuDNN).
        def set_random_seed_deterministic(seed, deterministic_torch=True):
            set_random_seed(seed, deterministic_torch=deterministic_torch)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        MazeUtils.set_random_seed = set_random_seed_deterministic

    MazeUtils.set_random_seed(42)
    maze_spec = MazeUtils.get_maze_spec_from_maze_name(LEGACY_MAZE[pad])
    build_kwargs = dict(env_kwargs={"maze_spec": maze_spec})
    if root_key == "maze":
        build_kwargs["backup_policy_type"] = LEGACY_BACKUP[backup]
    else:
        assert backup == "hq", "paper/gibbs only supports the HQ backup planner"
    policy, env, eval_env, config = MazeUtils.policy_and_env_from_checkpoint_and_config(
        args.ckpt, config_path, policy_mode, **build_kwargs)
    env.render = lambda *a, **k: np.zeros((8, 8, 3), dtype=np.uint8)

    episodes = []
    install_recorder(policy, episodes)
    with tempfile.TemporaryDirectory() as save_dir:
        t0 = time.time()
        stats = MazeUtils.rollout_random_seed(policy=policy, env=env, eval_env=eval_env, horizon=horizon,
                                              render_mode="rgb_array", seeds=np.array(seeds), save_dir=save_dir)
        elapsed = time.time() - t0
    if hasattr(policy, "pool"):
        policy.pool.close()
        policy.pool.join()

    arrays = episodes_to_arrays(episodes)
    np.savez(os.path.join(args.out, name + ".npz"), **arrays)
    per_seed = {str(s): {k: v for k, v in stats[str(s)].items() if k != "Time"} for s in seeds}
    replan_times = [r["time_s"] for ep in episodes for r in ep["replans"]]
    info = dict(planner=planner, pad=pad, backup=backup, seeds=seeds, horizon=horizon,
                legacy_root=root_key, legacy_policy_mode=policy_mode,
                legacy_backup_policy_type=LEGACY_BACKUP[backup], legacy_maze_name=LEGACY_MAZE[pad],
                legacy_config=cfg_name, legacy_config_md5=md5(config_path), stats=per_seed,
                n_steps=[int(len(arrays[f"ep{i}_action"])) for i in range(len(seeds))],
                n_replans=[int(len(arrays[f"ep{i}_replan_step"])) for i in range(len(seeds))],
                mean_replan_time_s=float(np.mean(replan_times)), wall_time_s=round(elapsed, 1),
                pool_workers=getattr(policy, "num_processes", None), modules=modules)
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(info, f, indent=2)


def diff_dirs(dir_a, dir_b, names):
    report = {}
    for name in names:
        a = np.load(os.path.join(dir_a, name + ".npz"))
        b = np.load(os.path.join(dir_b, name + ".npz"))
        entry = {}
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                entry[key] = {"bitwise_equal": False, "note": "missing"}
                continue
            va, vb = a[key], b[key]
            same = va.shape == vb.shape and va.dtype == vb.dtype and va.tobytes() == vb.tobytes()
            maxabs = float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))) \
                if (va.shape == vb.shape and va.size) else float("nan")
            entry[key] = {"bitwise_equal": bool(same), "max_abs_diff": maxabs}
        report[name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "tests", "golden"))
    parser.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "checkpoints", "maze2d_large_ep1900.pth"))
    parser.add_argument("--cases", nargs="*", default=None, help="subset of case names (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="legacy process-pool size (legacy default 16)")
    parser.add_argument("--jobs", type=int, default=2, help="cases run concurrently")
    parser.add_argument("--log-dir", default=os.path.join(tempfile.gettempdir(), "jm2d_maze_golden_logs"))
    parser.add_argument("--diff", default=None, help="compare the written cases against this capture dir")
    parser.add_argument("--no-seed-gym-spaces", dest="seed_gym_spaces", action="store_false",
                        help="keep gym's unseeded construction-time action sample (legacy, non-reproducible)")
    parser.add_argument("--deterministic", action="store_true",
                        help="force deterministic torch/cuDNN after every legacy set_random_seed (not needed: with "
                             "seeded gym spaces the legacy flags already give bitwise-identical captures)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    args.ckpt = os.path.abspath(args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    if args.worker:
        run_worker(args)
        return

    names = args.cases or [c[0] for c in CASES]
    os.makedirs(args.log_dir, exist_ok=True)
    procs, t0 = [], time.time()
    pending = list(names)
    while pending or procs:
        while pending and len(procs) < args.jobs:
            name = pending.pop(0)
            log = open(os.path.join(args.log_dir, name + ".log"), "w")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", name, "--out", args.out,
                   "--ckpt", args.ckpt, "--workers", str(args.workers)]
            if args.deterministic:
                cmd.append("--deterministic")
            if not args.seed_gym_spaces:
                cmd.append("--no-seed-gym-spaces")
            procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        for item in list(procs):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                procs.remove(item)
                print(f"{name}: exit {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)
                if proc.returncode != 0:
                    raise SystemExit(f"case {name} failed, see {log.name}")
        time.sleep(1)

    import torch
    infos = {}
    for name in names:
        with open(os.path.join(args.out, name + ".json")) as f:
            infos[name] = json.load(f)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"cases": {}}
    manifest["cases"].update(infos)
    for name in names:
        os.remove(os.path.join(args.out, name + ".json"))
    manifest.update(
        legacy_repo="SafeRobomimic (goldenhazard/SafeMimic)", legacy_refs=LEGACY_REFS,
        checkpoint={"file": "checkpoints/maze2d_large_ep1900.pth", "md5": md5(args.ckpt),
                    "source": "SafeRobomimic/diffusion_policy_trained_models/maze2d_large/model_epoch_1900.pth"},
        protocol=("set_random_seed(42) -> policy_and_env_from_checkpoint_and_config -> for seed in seeds: "
                  "set_random_seed(seed, deterministic_torch=False) -> env.reset -> legacy rollout(horizon); "
                  "video writer and env.render replaced by no-ops"),
        seed_gym_spaces=(f"unseeded gym spaces seeded with {GYM_SPACE_SEED} (MujocoEnv.__init__ random step)"
                         if args.seed_gym_spaces else "legacy (OS entropy)"),
        deterministic_gpu=(("after every legacy set_random_seed: torch.use_deterministic_algorithms(True, "
                            "warn_only=True), cudnn.deterministic=True, cudnn.benchmark=False")
                           if args.deterministic else "legacy flags (non-deterministic cuDNN)"),
        recorded=dict(per_step=STEP_KEYS, per_replan=REPLAN_KEYS,
                      note="ep<i>_* arrays, i = index into seeds; actions/states/k stored as float64 (lossless)"),
        environment=dict(CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                         pool_workers=args.workers, python=platform.python_version(), torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                         conda_env=os.environ.get("CONDA_DEFAULT_ENV", os.path.dirname(os.path.dirname(sys.executable)))),
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(names)} cases to {args.out} ({time.time() - t0:.0f} s)")

    if args.diff:
        report = diff_dirs(args.out, os.path.abspath(args.diff), names)
        n_bad = sum(1 for e in report.values() for v in e.values() if not v["bitwise_equal"])
        print(json.dumps(report, indent=1))
        print("DIFF:", "all bitwise equal" if n_bad == 0 else f"{n_bad} arrays differ")


if __name__ == "__main__":
    main()
