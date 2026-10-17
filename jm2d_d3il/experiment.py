"""One evaluation job = (planner, scenario, checkpoint seed, trials). Used by ``scripts/run.py``, the sweep and tests.

Order of operations (same as the original runners):
    set_seed(seed) -> load checkpoint <ckpt_dir>/<seed> -> create env -> build projector + policy
    -> for each trial: torch.manual_seed(trial), env.reset(), episode
Trials are independent (every trial re-seeds torch and resets the env), so a job can be resumed from the trials
already written to ``seed<k>.trials.jsonl``; ``tests/test_equivalence.py`` checks this against the original code.
"""
import json
import os
import sys
import time

import numpy as np

from . import planners as planners_mod
from . import tasks
from .paths import DEFAULT_CKPT_DIR, REPO_ROOT
from .results import (append_trial, md5, read_traj, read_trials, run_paths, summarize_trials, traj_path, traj_row,
                      versions, write_summary, write_traj)
from .runner import load_experiment, run_episodes
from .seeding import enable_determinism, set_seed


def portable(path):
    """Path relative to the repository root ('<repo>/...') when inside it (results carry no absolute paths)."""
    path = os.path.abspath(path)
    return "<repo>" + path[len(REPO_ROOT):] if path.startswith(REPO_ROOT) else path


def module_paths():
    import diffuser
    import d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding as avoiding_mod
    return {"diffuser": portable(os.path.dirname(diffuser.__file__)), "avoiding_env": portable(avoiding_mod.__file__)}


def setup(planner, scenario, seed, ckpt_dir=DEFAULT_CKPT_DIR, device="cuda"):
    """Seed, load the checkpoint, create the env and build the planner; returns (task, experiment, env)."""
    enable_determinism()
    task = tasks.make(scenario)
    set_seed(seed)
    experiment = load_experiment(ckpt_dir, seed, diffusion_class=planner.diffusion_class, device=device)
    env = task.make_env()
    planner.build(experiment, task)
    return task, experiment, env


def run_job(planner, scenario, seed, n_trials=100, out_dir="results/runs", ckpt_dir=DEFAULT_CKPT_DIR, resume=True,
            log=None, save_traj=False):
    """Run trials 0..n_trials-1 (skipping those already in the trials file); write the summary JSON when complete.

    ``save_traj``: also write the end-effector path of every trial run to ``seed<k>.traj.npz`` (``results.write_traj``,
    updated after every trial); the trials and their results are the same with or without it.
    """
    log = log or (lambda msg: print(msg, flush=True))
    json_path, jsonl_path = run_paths(out_dir, planner.label, scenario, seed)
    traj_file = traj_path(out_dir, planner.label, scenario, seed)
    if resume and os.path.exists(json_path):
        with open(json_path) as f:
            if json.load(f).get("n_trials") == n_trials:
                log(f"[run] {json_path} exists, skipping")
                return json_path
    done = read_trials(jsonl_path) if resume else {}
    if not resume:
        for path in (jsonl_path, traj_file):
            if os.path.exists(path):
                os.remove(path)
    todo = [t for t in range(n_trials) if t not in done]
    t0 = time.time()
    paths = module_paths()
    if todo:
        task, experiment, env = setup(planner, scenario, seed, ckpt_dir=ckpt_dir)
        log(f"[run] {planner.label} {scenario} seed{seed}: {len(done)} trials done, running {len(todo)} "
            f"| diffuser={paths['diffuser']} | env={paths['avoiding_env']}")

        ee_paths = {} if save_traj else None
        if save_traj:
            traj_rows = read_traj(traj_file)[1]
            halfspaces, discs = task.xy_constraints()
            traj_meta = dict(label=np.array(planner.label), scenario=np.array(scenario), seed=np.array(seed),
                             halfspaces=halfspaces, discs=discs, xlim=np.array(task.ax_limits[0], dtype=np.float64),
                             ylim=np.array(task.ax_limits[1], dtype=np.float64))

        def on_trial(trial, result):
            append_trial(jsonl_path, trial, result)
            if save_traj:
                traj_rows[trial] = traj_row(ee_paths.pop(trial), result)
                write_traj(traj_file, traj_meta, traj_rows)
            log(f"[run] {planner.label} {scenario} seed{seed} trial {trial}: ssucc={result['n_success_and_constraints']:.0f} "
                f"vio={result['n_violations']:.0f} steps={result['n_steps']:.0f} t/step={result['avg_time']:.2f}s")

        run_episodes(planner, task, env, todo, on_trial=on_trial, paths=ee_paths)
        env.close()
        legacy_scenario = task.legacy_name
    else:
        legacy_scenario = tasks.make(scenario).legacy_name

    trials = read_trials(jsonl_path)
    missing = [t for t in range(n_trials) if t not in trials]
    if missing:
        raise RuntimeError(f"{jsonl_path}: trials {missing[:5]}... missing")
    trials = {t: trials[t] for t in range(n_trials)}
    idx, arrays, metrics = summarize_trials(trials)
    ckpt = os.path.join(ckpt_dir, str(seed), "state_best.pt")
    write_summary(json_path, label=planner.label, planner=planner.name, scenario=scenario,
                  legacy_scenario=legacy_scenario, seed=seed, n_trials=n_trials, trials=arrays, metrics=metrics,
                  settings=planner.settings(), ckpt=os.path.relpath(ckpt, ckpt_dir), ckpt_md5=md5(ckpt),
                  versions=versions(), module_paths=paths,
                  argv=[portable(a) if os.path.isabs(a) else a for a in sys.argv],
                  wall_time_s=round(time.time() - t0, 1))
    log(f"[run] wrote {json_path}: ssucc={metrics['ssucc']:.3f} n_vio={metrics['n_vio']:.2f}")
    return json_path


def make_planner(planner, variant=None, u_steps=None, config=None, **overrides):
    """Planner from its name (+ safediffuser variant / jm2d u-steps)."""
    extra = dict(overrides)
    if planner == "safediffuser":
        extra["variant"] = variant
    elif variant is not None:
        raise ValueError("--variant only applies to safediffuser")
    if planner == "jm2d":
        extra["u_steps"] = u_steps
    elif u_steps is not None:
        raise ValueError("--u-steps only applies to jm2d")
    return planners_mod.from_config(planner, config_path=config, **extra)
