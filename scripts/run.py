#!/usr/bin/env python
"""Run planners (samplers) on a task and write one SampleResult per seed.

    python scripts/run.py --task donut_constrained --planner jm2d --seeds 0 --out results/runs
    python scripts/run.py --task donut_joint --planner sequential gibbs jm2d --seeds 0-4 --out results/runs

Writes ``<out>/<task>/<planner>/seed<k>.json`` (summary, see jm2d_toy/results.py) and ``seed<k>.npz``.
Hyper-parameters come from ``configs/donut_joint.json`` / ``configs/donut_constrained.json`` (``--config``
to use another file).

Deprecated aliases (kept for compatibility): ``--sampler`` = ``--planner``, ``--seed`` = ``--seeds``.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import jm2d_toy  # noqa: E402,F401  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)
from jm2d_toy import samplers, tasks  # noqa: E402
from jm2d_toy.config import load_config  # noqa: E402
from jm2d_toy.results import save_result  # noqa: E402
from jm2d_toy.seeding import enable_determinism  # noqa: E402

DEPRECATED = {"--sampler": "--planner", "--seed": "--seeds"}


def parse_seeds(spec):
    """'3' -> [3]; '0-4' -> [0, 1, 2, 3, 4]; '0,2,7' or '0 2 7' -> [0, 2, 7]."""
    seeds = []
    for part in spec.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tasks.TASKS, type=tasks.canonical)
    parser.add_argument("--planner", "--sampler", dest="planner", required=True, nargs="+",
                        help="donut_joint: sequential gibbs jm2d; donut_constrained: projection jm2d "
                             "(--sampler: deprecated alias)")
    parser.add_argument("--seeds", "--seed", dest="seeds", nargs="+", default=["0"],
                        help="seeds: '0-4', '0,2,7' or '0 2 7' (--seed: deprecated alias)")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "runs"), help="output directory")
    parser.add_argument("--n", type=int, default=None, help="number of samples (default: config n)")
    parser.add_argument("--config", default=None, help="config JSON (default: configs/<task default>.json)")
    parser.add_argument("--ckpt", default=None, help="override the config checkpoint")
    parser.add_argument("--history", action="store_true",
                        help="also record per-step histories (changes the RNG stream; see README)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    for old, new in DEPRECATED.items():
        if any(a == old or a.startswith(old + "=") for a in argv):
            print(f"note: {old} is deprecated, use {new}", file=sys.stderr)

    enable_determinism()
    cfg = load_config(args.config or args.task)
    if tasks.canonical(cfg["task"]) != args.task:
        parser.error(f"config is for task {cfg['task']!r}, not {args.task!r}")
    if args.ckpt:
        cfg["checkpoint"] = os.path.abspath(args.ckpt)
    n = args.n or cfg["n"]
    seeds = parse_seeds(",".join(args.seeds))

    problem = tasks.make(args.task)
    for name in args.planner:
        sampler = samplers.from_config(cfg, name, verbose=args.verbose)
        for seed in seeds:
            res = sampler.sample(problem, n=n, seed=seed, return_history=args.history)
            path = save_result(res, args.out, save_history=args.history)
            s = res.summary()
            extra = f"compatible {s['n_compatible']}/{n}" if res.joint else f"valid {s['n_valid']}/{n}"
            print(f"{args.task} {name} seed{seed}: {extra}, ood {s['n_ood']}/{n} "
                  f"({s['time_s']:.1f} s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
