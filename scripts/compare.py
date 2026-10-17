#!/usr/bin/env python
"""Aggregate results over seeds, print the table, optionally check against paper/targets.json, draw the figure.

    python scripts/compare.py --results results/runs --task donut_joint --check-paper
    python scripts/compare.py --paper --task donut_constrained --check-paper   # shipped stored paper runs, CPU only

``--task`` names the task (donut_joint | donut_constrained), never a path. The plot goes to ``--out`` (default
``results/runs/figures/<task>_<results dir name>.png``) and shows one run per planner (``--run-id``,
default the first one found, e.g. ``seed0`` or ``paper``). Drawing and the paper check run on the CPU.

Deprecated aliases (kept for compatibility): ``--from-paper-results`` = ``--paper``.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from jm2d_toy.compare import TASKS, compare  # noqa: E402
from jm2d_toy.tasks import canonical  # noqa: E402

DEPRECATED = {"--from-paper-results": "--paper", "--no-figure": "--no-plot"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "runs"), help="results directory")
    parser.add_argument("--paper", "--from-paper-results", dest="paper", action="store_true",
                        help="use the shipped stored paper runs in results/paper (--from-paper-results: deprecated)")
    parser.add_argument("--task", "--figure", dest="task", required=True, metavar="NAME", type=canonical,
                        help=f"task: {' | '.join(TASKS)}")
    parser.add_argument("--out", default=None, metavar="PATH", help="output figure path (PNG)")
    parser.add_argument("--check-paper", action="store_true", help="compare with paper/targets.json")
    parser.add_argument("--no-plot", "--no-figure", dest="no_plot", action="store_true",
                        help="print the table only (--no-figure: deprecated alias)")
    parser.add_argument("--run-id", default=None, help="run drawn in the figure, e.g. seed0 or paper")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.task not in TASKS:
        hint = " (--task is a task name; give the output path with --out)" \
            if ("/" in args.task or args.task.endswith((".png", ".pdf"))) else ""
        parser.error(f"unknown task {args.task!r}; choose from {list(TASKS)}{hint}")
    for old, new in DEPRECATED.items():
        if any(a == old or a.startswith(old + "=") for a in argv):
            print(f"note: {old} is deprecated, use {new}", file=sys.stderr)

    results = os.path.join(REPO_ROOT, "results", "paper") if args.paper else args.results
    out = None
    if not args.no_plot:
        out = args.out or os.path.join(REPO_ROOT, "results", "runs", "figures",
                                       f"{args.task}_{os.path.basename(os.path.normpath(results))}.png")
    _, text = compare(results, args.task, check_paper=args.check_paper, out=out, run_id=args.run_id)
    print(text)


if __name__ == "__main__":
    main()
