#!/usr/bin/env python
"""Run planners on the padded PointMaze and write one JSON line per episode.

    python scripts/run.py --planner jm2d --pad 0.15 --backup hq \
        --ckpt checkpoints/maze2d_large_ep1900.pth --seeds 0-99 --out results/runs
    python scripts/run.py --planner dp rail gibbs jm2d --pad 0 0.25 --seeds 0-9 --jobs 2   # one process per combo

Writes ``<out>/<task>/<planner>/<ckpt tag>.jsonl`` (schema: ``jm2d_maze/results.py``), ``<tag>.meta.json`` (config,
command, versions, checkpoint md5) and ``<tag>.log`` (the planners' console output). ``task`` = ``maze_pad015_hq``.

Common flags (same in every JM2D branch): ``--planner NAME [NAME ...]``, ``--seeds SPEC [SPEC ...]`` (``3``, ``0-99``,
``0,2,7``; several specs are concatenated), ``--out DIR``. Deprecated aliases: ``--planners``, ``--seed``,
``--num-seeds N`` (= ``--seeds 0-(N-1)``, the research scripts' ``--num_seeds``).
``scripts/sweep_fig4.py`` does not call this script (its ``job`` subcommand builds the same planner directly).

Protocol (paper runs, legacy ``run_rail_v2.py``): ``set_random_seed(42)`` -> build planner and environments ->
for each env seed in order: ``set_random_seed(seed)`` -> reset -> closed-loop rollout of at most ``--horizon``
steps. Env seeds must start at 0 to reproduce the stored runs (the start/goal of seed s depend on seeds < s).
"""
import argparse
import contextlib
import itertools
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import jm2d_maze  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)

DEFAULT_CKPT = os.path.join(REPO_ROOT, "checkpoints", "maze2d_large_ep1900.pth")
PLANNERS = ["dp", "rail", "gibbs", "jm2d"]
BACKUPS = ["hq", "xpyp", "xmym"]


def parse_seeds(specs):
    """'3' -> [3]; '0-4' -> [0, 1, 2, 3, 4]; '0,2,7' -> [0, 2, 7]; a list of specs is concatenated."""
    if isinstance(specs, (str, int)):
        specs = [specs]
    seeds = []
    for spec in specs:
        for part in str(spec).split(","):
            if "-" in part:
                lo, hi = part.split("-")
                seeds.extend(range(int(lo), int(hi) + 1))
            elif part:
                seeds.append(int(part))
    return seeds


def ckpt_tag(path):
    """checkpoints/maze2d_large_ep1900.pth -> ep1900; .../model_epoch_2000.pth -> ep2000."""
    m = re.search(r"(\d+)\.pth$", os.path.basename(path))
    return f"ep{m.group(1)}" if m else os.path.splitext(os.path.basename(path))[0]


def rel(path):
    """Path for the metadata without user directories: repo-relative, else '<package dir>/<file>'."""
    path = os.path.abspath(path)
    if path.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(path, REPO_ROOT)
    top = os.path.dirname(os.path.dirname(path))
    return os.path.join(os.path.basename(top), os.path.relpath(path, top))


def run_one(args):
    """One planner x pad x backup x checkpoint in this process."""
    import multiprocessing as mp
    import d4rl
    import robomimic
    import safediffusion.algo.jm2d_policy as JM2DPolicyModule

    from jm2d_maze.config import load_config
    from jm2d_maze.planners import build_planner
    from jm2d_maze.results import EpisodeResult, JSONLWriter, md5, run_path, versions
    from jm2d_maze.rollout import rollout_seeds
    from jm2d_maze.seeding import set_random_seed
    from jm2d_maze.tasks import read_maze_layout, task_name

    planner_name, pad, backup = args.planner[0], float(args.pad[0]), args.backup[0]
    task = task_name(pad, backup)
    layout = read_maze_layout(args.maze) if args.maze else None
    if args.maze:   # custom layout: task <file stem>_pad<w>_<backup>
        task = task.replace("maze", os.path.splitext(os.path.basename(args.maze))[0], 1)
    tag = args.tag or ckpt_tag(args.ckpt)
    out_path = run_path(args.out, task, planner_name, tag)
    writer = JSONLWriter(out_path)
    base = out_path[:-len(".jsonl")]
    seeds = parse_seeds(args.seeds)
    ckpt_md5 = md5(args.ckpt) if not args.no_md5 else ""

    config = load_config(args.config, planner=planner_name, num_workers=args.workers)

    modules = {"jm2d_maze": rel(jm2d_maze.__file__), "safediffusion": rel(JM2DPolicyModule.__file__),
               "robomimic": rel(robomimic.__file__), "d4rl": rel(d4rl.__file__)}
    meta = dict(task=task, planner=planner_name, pad=pad, backup=backup, ckpt=tag, ckpt_path=rel(args.ckpt),
                ckpt_md5=ckpt_md5, seeds=seeds, horizon=args.horizon, seed_gym_spaces=not args.no_seed_gym_spaces,
                maze=rel(args.maze) if args.maze else None, maze_layout=layout,
                command=" ".join(rel(a) if os.path.isabs(a) else a for a in sys.argv), versions=versions(),
                modules=modules, config=json.loads(config.dump()))
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    def on_episode(seed, stat, traj):
        rec = EpisodeResult.from_legacy_stat(stat, task=task, planner=planner_name, pad=pad, backup=backup,
                                             ckpt=tag, env_seed=int(seed), ckpt_md5=ckpt_md5)
        writer.write(rec)
        d = rec.to_dict()
        print(f"{task} {planner_name} {tag} seed{seed}: success={d['success']} unsafe={d['n_unsafe']} "
              f"int={d['n_intervention']}/{d['n_plan']} horizon={d['horizon_steps']} "
              f"({d['time_per_replan_s'] or 0:.2f} s/replan)", file=sys.__stdout__, flush=True)

    log = open(base + ".log", "w") if args.quiet else None
    with contextlib.redirect_stdout(log) if log else contextlib.nullcontext():
        t0 = time.time()
        set_random_seed(42)
        planner, env, eval_env = build_planner(planner_name, args.ckpt, pad=pad, backup=backup, config=config,
                                               seed_gym_spaces=not args.no_seed_gym_spaces, maze_layout=layout)
        try:
            rollout_seeds(planner, env, eval_env, horizon=args.horizon, seeds=seeds,
                          render_mode="rgb_array" if args.video else None,
                          video_dir=base + "_videos", on_episode=on_episode)
        finally:
            if hasattr(planner, "pool"):
                planner.pool.close()
                planner.pool.join()
            env.close() if hasattr(env, "close") else None
    if log:
        log.close()
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)", file=sys.__stdout__, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", "--planners", dest="planner", nargs="+", required=True, choices=PLANNERS,
                        help="dp rail gibbs jm2d (--planners: deprecated alias)")
    parser.add_argument("--pad", nargs="+", type=float, default=[0.0],
                        help="wall padding w >= 0 (paper: 0 .05 .10 .15 .20 .25)")
    parser.add_argument("--maze", default=None, metavar="FILE",
                        help="maze layout instead of maze2d-large: 9 lines of 12 tiles (# wall, O free)")
    parser.add_argument("--backup", nargs="+", default=["hq"], choices=BACKUPS)
    parser.add_argument("--ckpt", nargs="+", default=[DEFAULT_CKPT], help="DP checkpoint(s)")
    parser.add_argument("--seeds", "--seed", dest="seeds", nargs="+", default=["0-99"], metavar="SPEC",
                        help="env seeds: 0-99, 0,2,7 or several specs (start at 0, see docstring; --seed: deprecated)")
    parser.add_argument("--num-seeds", type=int, default=None, help=argparse.SUPPRESS)  # deprecated: --seeds 0-(N-1)
    parser.add_argument("--horizon", type=int, default=2000, help="max control steps per episode (20 s)")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "runs"))
    parser.add_argument("--config", default=None, help="experiment config (default configs/maze2d_large.json)")
    parser.add_argument("--workers", type=int, default=8,
                        help="process pool for JM2D/Gibbs candidate rollouts (research code: 16); no effect on results")
    parser.add_argument("--tag", default=None, help="run name (default: ep<epoch> from the checkpoint file name)")
    parser.add_argument("--jobs", type=int, default=1, help="combinations run concurrently (one process each)")
    parser.add_argument("--video", action="store_true", help="also write rgb_array videos (needs offscreen GL)")
    parser.add_argument("--quiet", action="store_true", help="planner console output to <tag>.log")
    parser.add_argument("--no-seed-gym-spaces", action="store_true",
                        help="keep the research code's unseeded construction-time action (non-reproducible)")
    parser.add_argument("--no-md5", action="store_true", help="skip the checkpoint md5 (faster)")
    args = parser.parse_args()
    for alias, new_flag in (("--planners", "--planner"), ("--seed", "--seeds")):
        if alias in sys.argv:
            print(f"note: {alias} is deprecated, use {new_flag}", file=sys.stderr)
    if args.num_seeds is not None:
        print("note: --num-seeds is deprecated, use --seeds 0-<N-1>", file=sys.stderr)
        args.seeds = [f"0-{args.num_seeds - 1}"]
    args.out = os.path.abspath(args.out)
    args.ckpt = [os.path.abspath(c) for c in args.ckpt]
    if args.maze:
        args.maze = os.path.abspath(args.maze)
        if not os.path.exists(args.maze):
            parser.error(f"maze layout {args.maze} not found")
    for c in args.ckpt:
        if not os.path.exists(c):
            parser.error(f"checkpoint {c} not found (see scripts/download_checkpoints.py)")

    combos = list(itertools.product(args.planner, args.pad, args.backup, args.ckpt))
    if len(combos) == 1:
        args.planner, args.pad, args.backup = [combos[0][0]], [combos[0][1]], [combos[0][2]]
        args.ckpt = combos[0][3]
        run_one(args)
        return

    # one fresh process per combination (as the research scripts did)
    passthrough = ["--seeds", *args.seeds, "--horizon", str(args.horizon), "--out", args.out,
                   "--workers", str(args.workers)]
    for flag in ("video", "quiet", "no_seed_gym_spaces", "no_md5"):
        if getattr(args, flag):
            passthrough.append("--" + flag.replace("_", "-"))
    if args.config:
        passthrough += ["--config", args.config]
    if args.maze:
        passthrough += ["--maze", args.maze]
    pending, running, failed = list(combos), [], []
    while pending or running:
        while pending and len(running) < args.jobs:
            planner, pad, backup, ckpt = pending.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--planner", planner, "--pad", str(pad),
                   "--backup", backup, "--ckpt", ckpt] + passthrough
            running.append((cmd, subprocess.Popen(cmd)))
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode != 0:
                    failed.append(" ".join(item[0]))
        time.sleep(1)
    if failed:
        raise SystemExit("failed:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
