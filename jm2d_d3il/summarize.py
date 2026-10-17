"""Table 2 / Table 3 from run results (mean +- std over checkpoint seeds) and the comparison with the paper.

Per (label, scenario): every seed's run gives ssucc_k = mean over its trials of ``n_success_and_constraints`` and
n_vio_k = mean of ``n_violations``. A table cell is mean_k(ssucc_k) +- std_k(ssucc_k) and mean_k(n_vio_k)
(population std, ddof = 0, unless ``ddof`` is given).

Sources:
  * reruns: ``results/<dir>/<label>/<scenario>/seed<k>.json`` (``jm2d_d3il/results.py``);
  * stored paper runs: ``results/paper/cummulative_data.json``, the aggregate behind Table 2 DPCC / Ours
    (SafeRobomimic@d3il b13ba1a4 ``scripts/notebooks/assets``). It keeps [mean, max, min] over seeds of the per-seed
    means; the per-seed values (and therefore the std) were not kept. DPCC = 'dpcc-c-tightened',
    Ours = 'model_based-dpcc-c-tightened'. Ours-Cluttered aggregates 3 seeds (value granularity, see below).

``aggregate_runs(..., include_partial=True)`` also includes seeds whose run is unfinished: a
``seed<k>.trials.jsonl`` with no finished ``seed<k>.json`` and >= ``min_partial_trials`` trials is aggregated from the
trials finished so far (same metric formulas, ``results.summarize_trials``). Such cells get ``partial=True`` and
``n_trials_per_seed``.
"""
import glob
import json
import os

import numpy as np

from .paths import PAPER_DIR, RESULTS_DIR
from .results import load_runs, read_trials, summarize_trials

SCENARIOS = ("top_left", "top_right", "both", "cluttered")
LEGACY_SCENARIOS = {"top_left": "halfspace_top-left-hard", "top_right": "halfspace_top-right-hard",
                    "both": "halfspace_both-hard", "cluttered": "halfspace_cluttered"}
STORED_VARIANTS = {"dpcc": "dpcc-c-tightened", "jm2d": "model_based-dpcc-c-tightened"}
TABLE2_ROWS = ("dpcc", "safediffuser", "mpd", "jm2d")
TABLE3_U = (0, 1, 2, 5, 10)
DEFAULT_STORED = os.path.join(RESULTS_DIR, "paper", "cummulative_data.json")


def load_targets(path=None):
    with open(path or os.path.join(PAPER_DIR, "targets.json")) as f:
        return json.load(f)


def infer_n_seeds(means, n_trials=100, max_seeds=5, tol=1e-6):
    """Smallest n <= max_seeds such that every mean * n_trials * n is integral.

    ``means`` are averages over n seeds x n_trials trials of integer per-trial counts (success flags, number of
    violating steps), so their sums are integers.
    """
    for n in range(1, max_seeds + 1):
        if all(abs(m * n_trials * n - round(m * n_trials * n)) < tol * max(1.0, m * n_trials * n) for m in means):
            return n
    return None


def load_stored(path=DEFAULT_STORED):
    """{(row, scenario): cell} from the stored aggregate (mean over seeds; max / min kept, std not stored)."""
    with open(path) as f:
        data = json.load(f)
    cells = {}
    for row, variant in STORED_VARIANTS.items():
        for scen, key in LEGACY_SCENARIOS.items():
            d = data[key][variant]
            ss_mean, ss_max, ss_min = d["n_success_and_constraints"]
            n = infer_n_seeds([ss_mean, d["n_violations"][0], d["n_success"][0]])
            cells[(row, scen)] = dict(ssucc=ss_mean, ssucc_std=None, ssucc_max=ss_max, ssucc_min=ss_min,
                                      n_vio=d["n_violations"][0], n_seeds=n, source="stored", label=row,
                                      variant=variant, time_per_step_s=d["avg_time"][0])
    return cells


def find_partial_seeds(results_dir, min_trials=1):
    """{(label, scenario, seed): row} for jobs with a trials.jsonl but no finished seed<k>.json (>= min_trials trials).

    ``row`` has the same shape (label, scenario, seed, n_trials, metrics) as a finished ``load_runs`` row, with
    ``metrics`` computed from the trials finished so far via ``results.summarize_trials``.
    """
    out = {}
    for jsonl_path in sorted(glob.glob(os.path.join(results_dir, "**", "seed*.trials.jsonl"), recursive=True)):
        json_path = jsonl_path[: -len(".trials.jsonl")] + ".json"
        if os.path.exists(json_path):
            continue
        seed = int(os.path.basename(jsonl_path)[len("seed"):-len(".trials.jsonl")])
        scenario = os.path.basename(os.path.dirname(jsonl_path))
        label = os.path.basename(os.path.dirname(os.path.dirname(jsonl_path)))
        trials = read_trials(jsonl_path)
        if len(trials) < min_trials:
            continue
        _, _, metrics = summarize_trials(trials)
        out[(label, scenario, seed)] = dict(label=label, scenario=scenario, seed=seed, n_trials=len(trials),
                                            metrics=metrics)
    return out


def aggregate_runs(results_dirs, ddof=0, include_partial=False, min_partial_trials=1):
    """{(label, scenario): cell} over the seeds found in the result directories.

    ``include_partial``: also use seeds with only a (sufficiently long) ``seed<k>.trials.jsonl``, see
    ``find_partial_seeds``; cells that include such a seed get ``partial=True`` and ``n_trials_per_seed``.
    """
    per = {}
    for d in results_dirs:
        for row in load_runs(d):
            per.setdefault((row["label"], row["scenario"]), {})[row["seed"]] = row
    partial = set()
    if include_partial:
        for d in results_dirs:
            for (label, scen, seed), row in find_partial_seeds(d, min_partial_trials).items():
                seeds = per.setdefault((label, scen), {})
                if seed in seeds:
                    continue
                seeds[seed] = row
                partial.add((label, scen))
    cells = {}
    for (label, scen), seeds in per.items():
        ss = np.array([seeds[k]["metrics"]["ssucc"] for k in sorted(seeds)])
        vio = np.array([seeds[k]["metrics"]["n_vio"] for k in sorted(seeds)])
        cell = dict(ssucc=float(ss.mean()), ssucc_std=float(ss.std(ddof=ddof)) if len(ss) > ddof else None,
                    ssucc_max=float(ss.max()), ssucc_min=float(ss.min()), n_vio=float(vio.mean()),
                    n_seeds=len(ss), seeds=sorted(seeds), source="rerun", label=label,
                    n_trials=sorted({seeds[k]["n_trials"] for k in seeds}),
                    time_per_step_s=float(np.mean([seeds[k]["metrics"]["time_per_step_s"] for k in seeds])),
                    per_seed_ssucc=[float(x) for x in ss], per_seed_n_vio=[float(x) for x in vio])
        if (label, scen) in partial:
            cell["partial"] = True
            cell["n_trials_per_seed"] = {k: seeds[k]["n_trials"] for k in sorted(seeds)}
        cells[(label, scen)] = cell
    return cells


def better(a, b):
    """The better SafeDiffuser cell: higher ssucc, then fewer violations."""
    if a is None:
        return b
    if b is None:
        return a
    return a if (a["ssucc"], -a["n_vio"]) >= (b["ssucc"], -b["n_vio"]) else b


def table2(runs, stored=None, prefer_stored=("dpcc", "jm2d")):
    """{row: {scenario: cell}} for DPCC, SafeDiffuser (best of ReS / TVS per scenario), MPD, Ours."""
    table = {}
    for row in TABLE2_ROWS:
        table[row] = {}
        for scen in SCENARIOS:
            if row == "safediffuser":
                cell = better(runs.get(("safediffuser-res", scen)), runs.get(("safediffuser-tvs", scen)))
            elif stored is not None and row in prefer_stored and (row, scen) in stored:
                cell = stored[(row, scen)]
            else:
                cell = runs.get((row, scen))
            table[row][scen] = cell
    return table


def table3(runs):
    return {str(u): {scen: runs.get((f"jm2d-u{u}", scen)) for scen in SCENARIOS} for u in TABLE3_U}


def repro_table(runs, stored, targets_table2, rows=("dpcc", "jm2d")):
    """{row: {source: {scenario: cell}}} for the reproduction checks of Table 2 DPCC and Ours.

    source = 'rerun' (results of this code; label ``dpcc`` resp. ``jm2d`` = u full), 'stored' (results/paper aggregate),
    'paper' (paper/targets.json).
    """
    out = {}
    for row in rows:
        out[row] = {
            "rerun": {scen: runs.get((row, scen)) for scen in SCENARIOS},
            "stored": {scen: (stored or {}).get((row, scen)) for scen in SCENARIOS},
            "paper": {scen: (dict(targets_table2[row][scen], source="paper", label=row, n_seeds=5)
                             if scen in targets_table2.get(row, {}) else None) for scen in SCENARIOS},
        }
    return out


def fmt_repro_cell(cell):
    if cell is None:
        return "-"
    if cell.get("source") == "stored":
        txt = f"{cell['ssucc']:.3f} [{cell['ssucc_min']:.2f}, {cell['ssucc_max']:.2f}] / {cell['n_vio']:.3f}"
    elif cell.get("source") == "paper":
        txt = f"{cell['ssucc']:.2f}±{cell['ssucc_std']:.2f} / {cell['n_vio']}"
    else:
        std = "n/a" if cell.get("ssucc_std") is None else f"{cell['ssucc_std']:.2f}"
        txt = f"{cell['ssucc']:.3f}±{std} [{cell['ssucc_min']:.2f}, {cell['ssucc_max']:.2f}] / {cell['n_vio']:.3f}"
    n = cell.get("n_seeds")
    return txt + (f" ({n}s)" if n not in (None, 5) or cell.get("source") == "rerun" else "")


def markdown_repro(repro, row_names):
    head = "| row | source | " + " | ".join(f"{s} SSucc / #Vio" for s in SCENARIOS) + " |"
    lines = [head, "|" + "---|" * (len(SCENARIOS) + 2)]
    for row, by_src in repro.items():
        for src in ("rerun", "stored", "paper"):
            lines.append(f"| {row_names.get(row, row)} | {src} | "
                         + " | ".join(fmt_repro_cell(by_src[src][s]) for s in SCENARIOS) + " |")
    return "\n".join(lines)


def fmt_cell(cell):
    if cell is None:
        return "-"
    std = "n/a" if cell.get("ssucc_std") is None else f"{cell['ssucc_std']:.2f}"
    tag = cell["label"] if cell["label"].startswith("safediffuser") else ""
    if cell.get("partial"):
        n_tot = sum(cell["n_trials_per_seed"].values())
        seeds = f" [{cell['n_seeds']}s, {n_tot}t, partial]"
    else:
        seeds = f" [{cell['n_seeds']}s]" if cell.get("n_seeds") not in (None, 5) else ""
    return f"{cell['ssucc']:.2f}±{std} / {cell['n_vio']:.2f}{seeds}{' (' + tag + ')' if tag else ''}"


def markdown(table, row_names, first_col="Algorithm"):
    head = f"| {first_col} | " + " | ".join(f"{s} SSucc±std / #Vio" for s in SCENARIOS) + " |"
    lines = [head, "|" + "---|" * (len(SCENARIOS) + 1)]
    for row, cells in table.items():
        lines.append(f"| {row_names.get(row, row)} | " + " | ".join(fmt_cell(cells[s]) for s in SCENARIOS) + " |")
    return "\n".join(lines)


def compare(value, target, decimals=2):
    """EXACT if the value rounds to the printed number, CLOSE if within 0.01, else DIFF."""
    if value is None:
        return "MISSING"
    if round(value, decimals) == round(target, decimals):
        return "EXACT"
    if abs(value - target) <= 0.01 + 1e-9:
        return "CLOSE"
    return "DIFF"


def check_paper(table, targets_section, name, printed_decimals=None):
    """Rows of (cell id, ours, paper, status, source, n_seeds) for ssucc, its std and n_vio.

    ``printed_decimals`` lists cells printed with one decimal in the paper (e.g. Table 3 u=10 TopLeft #Vio "1.6").
    """
    printed_decimals = printed_decimals or {}
    out = []
    for row, cells in table.items():
        for scen in SCENARIOS:
            tgt = targets_section.get(row, {}).get(scen)
            cell = cells.get(scen)
            if tgt is None:
                continue
            for key in ("ssucc", "ssucc_std", "n_vio"):
                val = None if cell is None else cell.get(key)
                cid = f"{name}.{row}.{scen}.{key}"
                dec = printed_decimals.get(cid, 2) if key == "n_vio" else 2
                status = compare(val, tgt[key], dec) if val is not None else ("NOT STORED" if cell else "MISSING")
                src = "" if cell is None else cell["source"]
                out.append((cid, val, tgt[key], status, src,
                            None if cell is None else cell.get("n_seeds")))
    return out


def format_check(rows):
    lines = ["| cell | this | paper | status | source | seeds |", "|---|---|---|---|---|---|"]
    for cid, val, tgt, status, src, n in rows:
        v = "-" if val is None else f"{val:.4f}"
        lines.append(f"| {cid} | {v} | {tgt} | {status} | {src} | {n if n is not None else '-'} |")
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    lines.append("")
    lines.append("tally: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)
