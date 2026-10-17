"""Result files shared by ``scripts/run.py``, ``scripts/summarize.py`` and the sweep.

One run = one (planner label, scenario, checkpoint seed) with ``n_trials`` trials:

    <out>/<label>/<scenario>/seed<k>.trials.jsonl   one line per finished trial (written as the run goes; resumable)
    <out>/<label>/<scenario>/seed<k>.json           summary, written when all trials are done
    <out>/<label>/<scenario>/seed<k>.traj.npz       end-effector paths, only with ``scripts/run.py --save-traj/--plot``

Summary fields: schema, label, planner, scenario, legacy_scenario, seed, n_trials, trials (per-trial arrays under the
keys of the original result JSON, ordered by trial index), metrics (means over trials: ssucc = success with no
violation, n_vio = violating time steps, success, collision_free, n_steps, time_per_step_s), settings (planner),
ckpt, ckpt_md5, versions, module_paths, wall_time_s.

Trajectory file (``write_traj``): ``trial`` (trial indices), per-trial ``success``, ``ssucc``, ``violated``
(n_violations > 0), ``n_violations``, ``n_steps``, one array ``path_<trial>`` (n_steps + 2, 2) of end-effector (x, y)
per trial, and the scenario drawn with them: ``label``, ``scenario``, ``seed``, ``halfspaces`` (rows cx, cy, d;
violated iff cx x + cy y >= d), ``discs`` (rows cx, cy, r; violated iff the distance to (cx, cy) is < r), ``xlim``,
``ylim`` (workspace).
"""
import glob
import hashlib
import json
import os
import platform

import numpy as np

from .runner import TRIAL_KEYS

SCHEMA = "jm2d_d3il/run_result/v1"


def run_paths(out_dir, label, scenario, seed):
    base = os.path.join(out_dir, label, scenario, f"seed{seed}")
    return base + ".json", base + ".trials.jsonl"


def traj_path(out_dir, label, scenario, seed):
    return os.path.join(out_dir, label, scenario, f"seed{seed}.traj.npz")


TRAJ_FLAGS = ("success", "ssucc", "violated", "n_violations", "n_steps")


def traj_row(path, result):
    """Trajectory entry of one trial from its path and its result (``runner.run_episodes``)."""
    return {"path": np.asarray(path, dtype=np.float64), "success": result["n_success"] > 0,
            "ssucc": result["n_success_and_constraints"] > 0, "violated": result["n_violations"] > 0,
            "n_violations": int(result["n_violations"]), "n_steps": int(result["n_steps"])}


def read_traj(path):
    """(meta, {trial: row}) of a trajectory file (rows as ``traj_row``); ({}, {}) if the file does not exist."""
    if not os.path.exists(path):
        return {}, {}
    with np.load(path) as z:
        rows = {int(t): {"path": z[f"path_{int(t)}"], **{k: z[k][i].item() for k in TRAJ_FLAGS}}
                for i, t in enumerate(z["trial"])}
        meta = {k: z[k] for k in z.files if k != "trial" and k not in TRAJ_FLAGS and not k.startswith("path_")}
    return meta, rows


def write_traj(path, meta, rows):
    """Write the trajectory file atomically (``meta``: scenario arrays, see the module docstring)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx = sorted(rows)
    arrays = {"trial": np.array(idx, dtype=np.int64), **{k: np.array([rows[t][k] for t in idx]) for k in TRAJ_FLAGS},
              **{f"path_{t}": rows[t]["path"] for t in idx}, **meta}
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import scipy
    import torch
    import diffusers
    try:
        import mujoco
        mj = mujoco.__version__
    except ImportError:  # pragma: no cover
        mj = None
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "diffusers": diffusers.__version__, "mujoco": mj,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def read_trials(jsonl_path):
    """{trial: result} from a (possibly partial) trials file; a torn last line is ignored."""
    done = {}
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(row["trial"])] = {k: row[k] for k in TRIAL_KEYS}
    return done


def append_trial(jsonl_path, trial, result, extra=None):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    row = {"trial": int(trial), **result, **(extra or {})}
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def summarize_trials(trials):
    """Per-trial arrays (sorted by trial) and the trial means."""
    idx = sorted(trials)
    arrays = {k: [trials[i][k] for i in idx] for k in TRIAL_KEYS}
    success = np.asarray(arrays["n_success"])
    metrics = {
        "ssucc": float(np.mean(arrays["n_success_and_constraints"])),
        "n_vio": float(np.mean(arrays["n_violations"])),
        "success": float(np.mean(success)),
        "collision_free": float(np.mean(arrays["collision_free_completed"])),
        "n_steps_success": float(np.mean(np.asarray(arrays["n_steps"])[success > 0])) if success.sum() > 0 else 0.0,
        "total_violations": float(np.mean(arrays["total_violations"])),
        "time_per_step_s": float(np.mean(arrays["avg_time"])),
    }
    return idx, arrays, metrics


def write_summary(json_path, **fields):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"schema": SCHEMA, **fields}, f, indent=1)
    os.replace(tmp, json_path)


def load_runs(results_dir):
    """All run summaries below ``results_dir``."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "seed*.json"), recursive=True)):
        with open(path) as f:
            row = json.load(f)
        if row.get("schema") != SCHEMA:
            continue
        row["path"] = path
        rows.append(row)
    return rows
