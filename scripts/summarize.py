"""Table 2 / Table 3 (mean +- std over checkpoint seeds) from run results, and the comparison with the paper.

    # stored paper runs only (DPCC and Ours rows of Table 2; no GPU needed)
    python scripts/summarize.py --paper --figure table2 --check-paper

    # re-runs (+ stored DPCC / Ours), written to a markdown file as well
    python scripts/summarize.py --results results/rerun --figure table2 table3 --check-paper --out results/rerun_tables.md

    # use re-run DPCC / Ours instead of the stored runs in Table 2
    python scripts/summarize.py --results results/runs --no-stored --figure table2

    # include seeds whose run is unfinished (trials.jsonl only), from >= 10 trials each
    python scripts/summarize.py --results results/rerun --include-partial --min-partial-trials 10

Flags shared with the other release branches: ``--results DIR`` (repeatable / several dirs), ``--paper`` (stored paper
results only, ``--results`` ignored), ``--figure {table2,table3}``, ``--out PATH`` (markdown report; ``.json`` writes
the tables as JSON), ``--check-paper`` (diff against paper/targets.json). Older aliases still work: ``--table 2 3``,
``--json PATH``.

``--include-partial`` (default off; off is byte-identical to not passing it at all) also aggregates seeds whose
run is unfinished: a ``seed<k>.trials.jsonl`` with no finished ``seed<k>.json`` and at least
``--min-partial-trials`` trials (default 1) is included, using its trials-finished-so-far metrics. Such cells are
marked ``[Ns, Mt, partial]`` in the markdown (N seeds, M total trials) and ``partial=True`` in the JSON output.

Table 2 output also lists DPCC and Ours side by side as re-run (labels ``dpcc``, ``jm2d``) / stored / paper.
Table 2 SafeDiffuser = the better of ReS / TVS per scenario (higher SSucc, then fewer violations); the cell names the
variant. "[3s]" marks cells aggregated over fewer than 5 seeds (stored Ours-Cluttered: 3 seeds).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jm2d_d3il import summarize as S  # noqa: E402

ROW_NAMES_T2 = {"dpcc": "DPCC", "safediffuser": "SafeDiffuser", "mpd": "MPD", "jm2d": "Ours (JM2D)"}
FIGURES = {"table2": "2", "table3": "3", "2": "2", "3": "3"}
PARTIAL_FOOTNOTE = ("_partial cells (`[Ns, Mt, partial]`) include seeds whose 100-trial run was stopped early, "
                    "aggregated over the trials finished so far (seed<k>.trials.jsonl)._")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", action="extend", default=None,
                    help="result directories (results/<dir>/<label>/<scenario>/seed<k>.json); may be repeated")
    ap.add_argument("--paper", action="store_true",
                    help="stored paper results only (results/paper); --results is ignored")
    ap.add_argument("--stored", default=S.DEFAULT_STORED, help="stored paper aggregate (results/paper/cummulative_data.json)")
    ap.add_argument("--no-stored", action="store_true", help="do not use the stored paper runs for DPCC / Ours")
    ap.add_argument("--figure", "--table", dest="figure", nargs="+", default=["table2", "table3"],
                    choices=sorted(FIGURES), help="table2 and/or table3 (alias: --table 2 3)")
    ap.add_argument("--check-paper", action="store_true", help="diff against paper/targets.json")
    ap.add_argument("--include-partial", action="store_true",
                    help="also aggregate seeds with only a seed<k>.trials.jsonl (no finished seed<k>.json); "
                         "default off, and off is identical to omitting the flag")
    ap.add_argument("--min-partial-trials", type=int, default=1,
                    help="minimum trials in a seed<k>.trials.jsonl to include it with --include-partial (default 1)")
    ap.add_argument("--ddof", type=int, default=0, help="std over seeds: 0 = population (default), 1 = sample")
    ap.add_argument("--out", default=None, help="write the report to PATH (markdown; a .json path writes the tables)")
    ap.add_argument("--json", default=None, help="(alias) also write the tables as JSON")
    args = ap.parse_args(argv)
    if args.paper and args.no_stored:
        ap.error("--paper and --no-stored exclude each other")
    args.tables = sorted({FIGURES[f] for f in args.figure})
    args.results = args.results or []
    return args


def main(argv=None):
    args = parse_args(argv)
    runs = {} if args.paper else S.aggregate_runs(args.results, ddof=args.ddof, include_partial=args.include_partial,
                                                  min_partial_trials=args.min_partial_trials)
    stored = None if args.no_stored or not os.path.exists(args.stored) else S.load_stored(args.stored)
    targets = S.load_targets()
    dump = {}
    lines = []

    def emit(text=""):
        lines.append(text)

    if "2" in args.tables:
        t2 = S.table2(runs, stored)
        emit("## Table 2: D3IL-Avoiding, SSucc (mean ± std over seeds) / #Vio\n")
        emit(S.markdown(t2, ROW_NAMES_T2))
        emit()
        if args.include_partial:
            emit(PARTIAL_FOOTNOTE)
            emit()
        dump["table2"] = {r: {s: c for s, c in cells.items()} for r, cells in t2.items()}
        if args.check_paper:
            emit("### Table 2 vs paper\n")
            emit(S.format_check(S.check_paper(t2, targets["table2"], "table2", targets.get("printed_decimals"))))
            emit()
        repro = S.repro_table(runs, stored, targets["table2"])
        emit("### Reproduction checks: DPCC and Ours, re-run vs stored vs paper\n")
        emit("re-run: mean ± std [min, max] over seeds (number of seeds); stored: mean [min, max] (std not stored); "
             "paper: mean ± std. Cells: SSucc / #Vio.\n")
        emit(S.markdown_repro(repro, ROW_NAMES_T2))
        emit()
        dump["repro"] = repro
        if args.check_paper:
            reruns = {r: {s: c for s, c in by_src["rerun"].items()} for r, by_src in repro.items()}
            if any(c is not None for cells in reruns.values() for c in cells.values()):
                emit("### Re-run DPCC / Ours vs paper\n")
                emit(S.format_check(S.check_paper(reruns, targets["table2"], "table2_rerun",
                                                  targets.get("printed_decimals"))))
                emit()
    if "3" in args.tables:
        t3 = S.table3(runs)
        emit("## Table 3: JM2D u-step ablation (--u-steps u), SSucc (mean ± std) / #Vio\n")
        emit(S.markdown(t3, {u: f"u = {u}" for u in t3}, first_col="u"))
        emit()
        if args.include_partial:
            emit(PARTIAL_FOOTNOTE)
            emit()
        dump["table3"] = t3
        if args.check_paper:
            emit("### Table 3 vs paper\n")
            emit(S.format_check(S.check_paper(t3, targets["table3_u_steps"], "table3_u_steps",
                                              targets.get("printed_decimals"))))
            emit()

    report = "\n".join(lines)
    print(report)
    json_paths = ([args.json] if args.json else []) + ([args.out] if args.out and args.out.endswith(".json") else [])
    for path in json_paths:
        with open(path, "w") as f:
            json.dump(dump, f, indent=1)
    if args.out and not args.out.endswith(".json"):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
