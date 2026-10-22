#!/usr/bin/env python
"""Aggregate PointMaze results; reproduce Fig. 4 / Table 1; diff against paper/targets.json.

    python scripts/compare.py --paper --figure fig4 --check-paper    # shipped runs: DP/RAIL re-run, JM2D/Gibbs stored
    python scripts/compare.py --paper --figure fig4 --dp-rail-source stored   # all four curves from the stored runs
    python scripts/compare.py --paper --figure table1 --check-paper  # Table 1 from the stored runs
    python scripts/compare.py --results results/runs --figure fig4 --out fig4.png   # your own scripts/run.py output

Common flags (same in every JM2D branch): ``--results DIR`` (own runs) or ``--paper`` (the shipped paper runs; the
default when ``--results`` is not given), ``--figure NAME [NAME ...]`` (``fig4``, ``table1``; default both),
``--out PATH`` (figure file), ``--check-paper`` (diff against paper/targets.json). Deprecated aliases: ``--table1``
(= ``--figure table1``), ``--from-paper-results`` (= ``--paper``).

Fig. 4 uses the paper's pads {0, .05, .15, .20, .25} (``--all-pads`` adds w = 0.10, which the stored runs have).
With ``--paper``, the DP and RAIL curves come from the released-code re-run (``results/fig4_rerun``,
``scripts/sweep_fig4.py``) for every (planner, w) cell whose five checkpoint runs are complete, and from
the stored runs otherwise; JM2D and Gibbs always come from the stored runs. The source of every cell is printed and
shown in the legend. ``--allow-partial`` also uses incomplete re-runs (finished episodes of running jobs).
The figure goes to ``--out`` (default ``results/figures/fig4_<source>.png``).
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from jm2d_maze.compare import (PAPER_RESULTS, RERUN_RESULTS, fig4_summary, fig4_text, load_paper_runs,  # noqa: E402
                               load_rerun_runs, load_runs, load_targets, match_stored, match_stored_text, plot_fig4,
                               select_fig4_runs, sources_text, table1_summary, table1_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--results", default=None, metavar="DIR", help="results dir of scripts/run.py")
    src.add_argument("--paper", action="store_true",
                     help="the shipped paper runs (results/paper, results/fig4_rerun); default without --results")
    src.add_argument("--from-paper-results", dest="paper", action="store_true", help=argparse.SUPPRESS)  # deprecated
    parser.add_argument("--figure", nargs="+", choices=["fig4", "table1"], default=None, metavar="NAME",
                        help="fig4 and/or table1 (default: both)")
    parser.add_argument("--table1", action="store_true", help=argparse.SUPPRESS)  # deprecated: --figure table1
    parser.add_argument("--check-paper", action="store_true", help="diff against paper/targets.json")
    parser.add_argument("--all-pads", action="store_true", help="Fig. 4: include w = 0.10 (not in the paper)")
    parser.add_argument("--planners", nargs="+", default=None, help="restrict to these planners")
    parser.add_argument("--ckpts", nargs="+", default=None, help="restrict to these checkpoints, e.g. ep1900")
    parser.add_argument("--dp-rail-source", choices=["rerun", "stored"], default="rerun",
                        help="Fig. 4 DP/RAIL curves: released-code re-run (default; stored where not complete) "
                             "or the stored research-code runs")
    parser.add_argument("--rerun-dir", default=RERUN_RESULTS, help="output of scripts/sweep_fig4.py")
    parser.add_argument("--allow-partial", action="store_true",
                        help="use incomplete DP/RAIL re-runs (finished episodes) instead of falling back to stored")
    parser.add_argument("--match-stored", action="store_true",
                        help="per-episode comparison of --results with the stored run of the same planner/pad/ckpt")
    parser.add_argument("--out", default=None, metavar="PATH", help="figure path (PNG; Fig. 4)")
    parser.add_argument("--json", default=None, help="also write the summaries to this JSON file")
    args = parser.parse_args()
    figures = set(args.figure or [])
    if args.table1:
        print("note: --table1 is deprecated, use --figure table1", file=sys.stderr)
        figures.add("table1")
    if "--from-paper-results" in sys.argv:
        print("note: --from-paper-results is deprecated, use --paper", file=sys.stderr)
    if args.match_stored:
        if args.results is None:
            parser.error("--match-stored needs --results")
        rows = match_stored(load_runs(args.results), load_paper_runs(PAPER_RESULTS))
        print(match_stored_text(rows))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return
    if not figures:
        figures = {"fig4", "table1"}

    targets = load_targets() if args.check_paper else None
    source = "paper" if args.results is None else os.path.basename(os.path.normpath(args.results))
    dump = {}

    def select(runs):
        if args.planners:
            runs = [r for r in runs if r["planner"] in args.planners]
        if args.ckpts:
            runs = [r for r in runs if r["ckpt"] in args.ckpts]
        return runs

    if "fig4" in figures:
        pads = None if args.all_pads else [0.0, 0.05, 0.15, 0.20, 0.25]
        sources = None
        if args.results is None:
            stored = load_paper_runs(PAPER_RESULTS, figure="fig4")
            reruns = load_rerun_runs(args.rerun_dir, allow_partial=args.allow_partial) if os.path.isdir(args.rerun_dir) else []
            runs, sources = select_fig4_runs(stored, reruns, dp_rail_source=args.dp_rail_source,
                                             allow_partial=args.allow_partial)
            source = "paper_stored" if args.dp_rail_source == "stored" else "paper"
            title = ("Fig. 4: DP/RAIL " + ("re-run (released code)" if args.dp_rail_source == "rerun" else "stored")
                     + ", JM2D/Gibbs stored; crosses = paper")
        else:
            runs = load_runs(args.results)
            title = f"Fig. 4 ({source} runs; crosses = paper)"
        runs = select(runs)
        summary = fig4_summary(runs, pads=pads)
        if sources is not None:
            print(sources_text(sources, pads, args.dp_rail_source))
        print(fig4_text(summary, targets, sources=sources))
        out = args.out or os.path.join(REPO_ROOT, "results", "figures", f"fig4_{source}.png")
        plot_fig4(summary, out, targets=targets, title=title, sources=sources)
        print(f"\nfigure: {out}")
        dump["fig4"] = {m: {p: {str(k): v for k, v in c.items()} for p, c in d.items()} for m, d in summary.items()}
        if sources is not None:
            dump["fig4_sources"] = {f"{p}@{pad:.2f}": v for (p, pad), v in sources.items()}
    if "table1" in figures:
        runs = select(load_paper_runs(PAPER_RESULTS, figure="table1") if args.results is None else load_runs(args.results))
        summary = table1_summary(runs)
        print(table1_text(summary, targets))
        dump["table1"] = summary
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dump, f, indent=1)


if __name__ == "__main__":
    main()
