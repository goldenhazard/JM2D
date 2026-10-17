"""Aggregate PointMaze runs, reproduce Fig. 4 / Table 1, and diff them against ``paper/targets.json``.

A *run* is one planner x task x checkpoint over env seeds (a JSONL file of ``scripts/run.py`` or one stored
research-code folder in ``results/paper``). Metric definitions are those of the research-code scripts
(``generate_paper_plot.py`` for Fig. 4, ``print_stats.py`` for Table 1):

Fig. 4 (per run, then mean / min / max over runs = the paper's line and band):
    safe_success  #(success and no collision) / #episodes
    intervention  sum(interventions) / sum(replanning cycles), successful episodes only
    horizon_s     mean control steps of successful episodes x 0.01 s
Fig. 4 sources: the stored DP and RAIL runs were made with research code that conditioned the DP on
the current frame twice; their curves are re-run with this code (``scripts/sweep_fig4.py`` -> results/fig4_rerun) and
used where complete (``select_fig4_runs``); JM2D and Gibbs (never affected) use the stored runs.

Table 1 (one run per cell):
    ssucc         #(success and no collision) / #episodes
    shor          mean control steps of successful episodes (steps)
    int           sum(interventions) / sum(replanning cycles), all episodes
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from jm2d_maze.results import legacy_stats_to_records, load_results

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_RESULTS = os.path.join(REPO_ROOT, "results", "paper")
RERUN_RESULTS = os.path.join(REPO_ROOT, "results", "fig4_rerun")
FIG4_CKPTS = ("ep1250", "ep1500", "ep1750", "ep1900", "ep2000")   # the paper's "5 random seeds"
RERUN_PLANNERS = ("dp", "rail")                                   # curves taken from the released-code re-run
TARGETS = os.path.join(REPO_ROOT, "paper", "targets.json")
PLANNERS = ["dp", "rail", "gibbs", "jm2d"]
DISPLAY = {"jm2d": "Ours (JM2D)", "gibbs": "Gibbs", "rail": "RAIL", "dp": "DP"}
# paper styling (legend order Ours, Gibbs, RAIL, DP)
STYLE = {"jm2d": dict(color="royalblue", linestyle="-", linewidth=4),
         "gibbs": dict(color="limegreen", linestyle="-.", linewidth=2),
         "rail": dict(color="darkorange", linestyle="--", linewidth=2),
         "dp": dict(color="crimson", linestyle=":", linewidth=2)}


# --------------------------------------------------------------------------------------------- loading
def load_paper_runs(results_dir=PAPER_RESULTS, figure=None):
    """Stored research-code runs (``results/paper/index.json``) -> list of runs (dict with ``records``)."""
    with open(os.path.join(results_dir, "index.json")) as f:
        index = json.load(f)["runs"]
    runs = []
    for entry in index:
        if figure is not None and entry["figure"] != figure:
            continue
        run_dir = os.path.join(results_dir, entry["path"])
        stats_path = os.path.join(run_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            stats = {}
            for p in glob.glob(os.path.join(run_dir, "seed*", "stats.json")):
                with open(p) as f:
                    stats[os.path.basename(os.path.dirname(p))[len("seed"):]] = json.load(f)
        meta = dict(task=f"maze_pad{int(round(entry['pad'] * 100)):03d}_{entry['backup']}", planner=entry["planner"],
                    pad=entry["pad"], backup=entry["backup"], ckpt=entry["ckpt"], source="paper_stored",
                    run=entry["path"])
        runs.append(dict(meta, figure=entry["figure"], records=legacy_stats_to_records(stats, **meta)))
    return runs


def load_rerun_runs(rerun_dir=RERUN_RESULTS, allow_partial=False):
    """Fig. 4 DP/RAIL re-runs of ``scripts/sweep_fig4.py`` (same per-seed schema as the stored runs).

    A run is complete when its ``stats.json`` holds all seeds; with ``allow_partial`` the finished seeds of running or
    interrupted jobs are used as well (``seed<k>/stats.json``).
    """
    runs = []
    for meta_path in sorted(glob.glob(os.path.join(rerun_dir, "large_pad*", "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        stats = None
        stats_path = os.path.join(run_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
        complete = stats is not None and stats.get("total_runs") == meta["n_seeds"]
        if not complete:
            if not allow_partial:
                continue
            stats = {}
            for p in glob.glob(os.path.join(run_dir, "seed*", "stats.json")):
                with open(p) as f:
                    stats[os.path.basename(os.path.dirname(p))[len("seed"):]] = json.load(f)
            if not stats:
                continue
        info = dict(task=f"maze_pad{int(round(meta['pad'] * 100)):03d}_{meta['backup']}", planner=meta["planner"],
                    pad=meta["pad"], backup=meta["backup"], ckpt=meta["ckpt"], source="rerun",
                    run=os.path.relpath(run_dir, rerun_dir))
        runs.append(dict(info, figure="fig4", complete=complete, n_seeds=meta["n_seeds"],
                         records=legacy_stats_to_records(stats, **info)))
    return runs


def select_fig4_runs(stored_runs, rerun_runs=(), dp_rail_source="rerun", allow_partial=False):
    """Runs used for Fig. 4 and the source label of every (planner, pad) cell.

    JM2D and Gibbs: stored runs. DP and RAIL (``dp_rail_source="rerun"``): the re-runs of a cell once all five
    checkpoints are complete (with ``allow_partial``: whatever re-run episodes exist), otherwise the stored runs.
    Returns (runs, {(planner, pad): label}).
    """
    rerun_by_cell = defaultdict(list)
    for r in rerun_runs:
        if r["backup"] == "hq":
            rerun_by_cell[(r["planner"], round(r["pad"], 2))].append(r)
    selected, sources = [], {}
    stored_by_cell = defaultdict(list)
    for r in stored_runs:
        stored_by_cell[(r["planner"], round(r["pad"], 2))].append(r)
    for cell in sorted(set(stored_by_cell) | set(rerun_by_cell)):
        planner, pad = cell
        reruns = rerun_by_cell.get(cell, [])
        n_complete = len({r["ckpt"] for r in reruns if r.get("complete")})
        use_rerun = planner in RERUN_PLANNERS and dp_rail_source == "rerun" and reruns and (
            allow_partial or n_complete >= len(FIG4_CKPTS))
        if use_rerun:
            selected += reruns
            n_eps = sum(len(r["records"]) for r in reruns)
            full = n_complete >= len(FIG4_CKPTS) and all(r.get("complete") for r in reruns)
            sources[cell] = "rerun" if full else (f"rerun (partial: {len({r['ckpt'] for r in reruns})}/"
                                                  f"{len(FIG4_CKPTS)} ckpts, {n_eps} episodes)")
        else:
            selected += stored_by_cell.get(cell, [])
            note = ""
            if planner in RERUN_PLANNERS and dp_rail_source == "rerun":
                note = f" (rerun {n_complete}/{len(FIG4_CKPTS)} ckpts complete)"
            sources[cell] = "stored" + note if stored_by_cell.get(cell) else "none"
    return selected, sources


def curve_source(sources, planner):
    """'rerun', 'stored' or 'mixed' for the legend of one planner's curve."""
    kinds = {v.split(" ")[0] for (p, _), v in sources.items() if p == planner}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def sources_text(sources, pads=None, dp_rail_source="rerun"):
    head = ("DP/RAIL from the released-code re-run where complete, JM2D/Gibbs stored runs"
            if dp_rail_source == "rerun" else "all curves from the stored research-code runs")
    lines = [f"\nFig. 4 curve sources ({head})"]
    for planner in ["jm2d", "gibbs", "rail", "dp"]:
        cells = sorted((pad, v) for (p, pad), v in sources.items() if p == planner and (pads is None or pad in pads))
        if cells:
            lines.append(f"  {planner:6s} " + "; ".join(f"w={pad:.2f}: {v}" for pad, v in cells))
    return "\n".join(lines)


def load_runs(results_dir):
    """JSONL results of ``scripts/run.py`` -> list of runs grouped by (task, planner, ckpt)."""
    groups = defaultdict(list)
    for row in load_results(results_dir):
        groups[(row["task"], row["planner"], row["ckpt"])].append(row)
    runs = []
    for (task, planner, ckpt), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["env_seed"])
        runs.append(dict(task=task, planner=planner, pad=rows[0]["pad"], backup=rows[0]["backup"], ckpt=ckpt,
                         source=rows[0]["source"], run=rows[0]["run"], figure=None, records=rows))
    return runs


# --------------------------------------------------------------------------------------------- metrics
def safe_success(records):
    return float(np.mean([r["success"] and r["n_unsafe"] == 0 for r in records]))


def intervention_successful(records):
    succ = [r for r in records if r["success"]]
    plans = sum(r["n_plan"] for r in succ)
    return sum(r["n_intervention"] for r in succ) / plans if plans else float("nan")


def horizon_successful_steps(records):
    succ = [r["horizon_steps"] for r in records if r["success"]]
    return float(np.mean(succ)) if succ else float("nan")


def intervention_all(records):
    plans = sum(r["n_plan"] for r in records)
    return sum(r["n_intervention"] for r in records) / plans if plans else 0.0


FIG4_METRICS = {"safe_success": safe_success, "intervention": intervention_successful,
                "horizon_s": lambda rec: horizon_successful_steps(rec) * 0.01}


def fig4_summary(runs, pads=None):
    """{metric: {planner: {pad: dict(mean, min, max, n_runs, n_episodes, runs)}}} for HQ runs."""
    out = {m: defaultdict(dict) for m in FIG4_METRICS}
    by_cell = defaultdict(list)
    for run in runs:
        if run["backup"] == "hq":
            by_cell[(run["planner"], round(run["pad"], 2))].append(run)
    for (planner, pad), cell_runs in sorted(by_cell.items()):
        if pads is not None and pad not in pads:
            continue
        for metric, fn in FIG4_METRICS.items():
            if metric == "intervention" and planner == "dp":
                continue
            vals = np.array([fn(r["records"]) for r in cell_runs])
            out[metric][planner][pad] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)),
                                             max=float(np.max(vals)), n_runs=len(cell_runs),
                                             n_episodes=[len(r["records"]) for r in cell_runs],
                                             runs=[r["ckpt"] for r in cell_runs])
    return out


def table1_summary(runs):
    """{planner: {backup: dict(ssucc, shor, int, n, ...)}} for the w = 0.20 runs (one run per cell)."""
    out = defaultdict(dict)
    for run in runs:
        if round(run["pad"], 2) != 0.20 or run["planner"] not in ("rail", "jm2d"):
            continue
        rec = run["records"]
        out[run["planner"]][run["backup"]] = dict(
            ssucc=safe_success(rec), shor=horizon_successful_steps(rec), int=intervention_all(rec), n=len(rec),
            n_safe_success=int(sum(r["success"] and r["n_unsafe"] == 0 for r in rec)),
            n_intervention=int(sum(r["n_intervention"] for r in rec)), n_plan=int(sum(r["n_plan"] for r in rec)),
            run=run["run"])
    return out


# --------------------------------------------------------------------------------------------- tables
def _fmt(x, nd=3):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def verdict(metric, diff):
    tol = (0.15, 0.35) if metric == "horizon_s" else (0.015, 0.05)
    a = abs(diff)
    return "MATCH" if a <= tol[0] else ("CLOSE" if a <= tol[1] else "MISMATCH")


def fig4_text(summary, targets=None, sources=None):
    lines = []
    for metric in FIG4_METRICS:
        lines.append(f"\nFig. 4 {metric} (mean over runs [min, max]; n runs)")
        header = f"{'planner':8s} {'w':>5s} {'value':>7s} {'min':>7s} {'max':>7s} {'runs':>4s}"
        if targets:
            header += f" {'paper':>7s} {'diff':>7s}  verdict"
        lines.append(header)
        for planner in ["jm2d", "gibbs", "rail", "dp"]:
            for pad, s in sorted(summary[metric].get(planner, {}).items()):
                row = f"{planner:8s} {pad:5.2f} {_fmt(s['mean']):>7s} {_fmt(s['min']):>7s} {_fmt(s['max']):>7s} {s['n_runs']:4d}"
                if targets:
                    tv = None
                    tpads = targets["fig4"]["pads"]
                    vals = targets["fig4"]["values"][metric].get(planner)
                    if vals is not None and pad in tpads:
                        tv = vals[tpads.index(pad)]
                    if tv is None:
                        row += f" {'-':>7s} {'-':>7s}  (not in paper)"
                    else:
                        d = s["mean"] - tv
                        row += f" {tv:7.3f} {d:+7.3f}  {verdict(metric, d)}"
                if sources is not None:
                    row += f"  [{sources.get((planner, pad), '?')}]"
                lines.append(row)
    return "\n".join(lines)


def table1_text(summary, targets=None):
    lines = ["\nTable 1 (w = 0.20): SSucc | SHor [steps] | Int"]
    header = f"{'planner':8s} {'backup':6s} {'n':>3s} {'SSucc':>6s} {'SHor':>7s} {'Int':>7s}"
    if targets:
        header += "   paper: SSucc   SHor    Int   exact (paper rounding)"
    lines.append(header)
    n_exact = n_cells = 0
    for planner in ["rail", "jm2d"]:
        for backup in ["hq", "xpyp", "xmym"]:
            s = summary.get(planner, {}).get(backup)
            if s is None:
                continue
            row = (f"{planner:8s} {backup:6s} {s['n']:3d} {s['ssucc']:6.2f} {s['shor']:7.1f} {s['int']:7.3f}"
                   f" ({s['n_intervention']}/{s['n_plan']})")
            if targets:
                t = targets["table1"]["values"][planner][backup]
                ok = [round(s["ssucc"], 2) == t["ssucc"], round(s["shor"], 1) == t["shor"], round(s["int"], 2) == t["int"]]
                n_exact += sum(ok)
                n_cells += 3
                flags = "".join("=" if o else "x" for o in ok)
                row += f"   {t['ssucc']:5.2f} {t['shor']:6.1f} {t['int']:6.2f}   {flags}"
                if not ok[2] and int(s["int"] * 100) / 100 == t["int"]:
                    row += "  (Int: paper truncates)"
            lines.append(row)
    if targets and n_cells:
        lines.append(f"exact: {n_exact}/{n_cells} numbers (= : equal at the paper's precision)")
    return "\n".join(lines)


def load_targets(path=TARGETS):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------- stored-run check
MATCH_KEYS = ("success", "n_unsafe", "n_intervention", "n_plan", "horizon_steps")


def match_stored(runs, paper_runs):
    """Per-episode comparison of re-runs with the stored run of the same planner / pad / backup / checkpoint.

    Returns rows (planner, pad, backup, ckpt, stored run, n compared, n identical on all of MATCH_KEYS, per-key
    identical counts, aggregate metrics of both).
    """
    stored = {(r["planner"], round(r["pad"], 2), r["backup"], r["ckpt"]): r for r in paper_runs}
    rows = []
    for run in runs:
        key = (run["planner"], round(run["pad"], 2), run["backup"], run["ckpt"])
        ref = stored.get(key)
        if ref is None:
            continue
        ref_by_seed = {r["env_seed"]: r for r in ref["records"]}
        pairs = [(r, ref_by_seed[r["env_seed"]]) for r in run["records"] if r["env_seed"] in ref_by_seed]
        per_key = {k: sum(a[k] == b[k] for a, b in pairs) for k in MATCH_KEYS}
        rows.append(dict(planner=key[0], pad=key[1], backup=key[2], ckpt=key[3], stored=ref["run"], n=len(pairs),
                         n_identical=sum(all(a[k] == b[k] for k in MATCH_KEYS) for a, b in pairs), per_key=per_key,
                         rerun=dict(safe_success=safe_success([a for a, _ in pairs]),
                                    intervention=intervention_all([a for a, _ in pairs]),
                                    horizon=float(np.mean([a["horizon_steps"] for a, _ in pairs]))),
                         stored_same_seeds=dict(safe_success=safe_success([b for _, b in pairs]),
                                                intervention=intervention_all([b for _, b in pairs]),
                                                horizon=float(np.mean([b["horizon_steps"] for _, b in pairs]))),
                         seeds=[a["env_seed"] for a, _ in pairs],
                         differing=[dict(env_seed=a["env_seed"], rerun={k: a[k] for k in MATCH_KEYS},
                                         stored={k: b[k] for k in MATCH_KEYS})
                                    for a, b in pairs if not all(a[k] == b[k] for k in MATCH_KEYS)]))
    return rows


def match_stored_text(rows):
    lines = ["\nRe-run vs stored run, per episode (identical = same Success, Unsafe, Intervention, Plan, Horizon)",
             f"{'planner':7s} {'w':>5s} {'ckpt':7s} {'n':>3s} {'ident':>5s}  succ unsafe int plan hor   "
             f"rerun SS/Int/Hor        stored SS/Int/Hor       stored run"]
    for r in rows:
        pk = r["per_key"]
        lines.append(f"{r['planner']:7s} {r['pad']:5.2f} {r['ckpt']:7s} {r['n']:3d} {r['n_identical']:5d}  "
                     f"{pk['success']:4d} {pk['n_unsafe']:6d} {pk['n_intervention']:3d} {pk['n_plan']:4d} "
                     f"{pk['horizon_steps']:3d}   "
                     f"{r['rerun']['safe_success']:.2f}/{r['rerun']['intervention']:.3f}/{r['rerun']['horizon']:6.1f}   "
                     f"{r['stored_same_seeds']['safe_success']:.2f}/{r['stored_same_seeds']['intervention']:.3f}/"
                     f"{r['stored_same_seeds']['horizon']:6.1f}   {r['stored']}")
    tot = sum(r["n"] for r in rows)
    lines.append(f"total: {sum(r['n_identical'] for r in rows)}/{tot} episodes identical")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- figure
def plot_fig4(summary, path, targets=None, title=None, sources=None):
    """Three panels (safe success, intervention, horizon) with min-max bands; paper values as crosses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("safe_success", "Safe Success Rate", (0.2, 1.05)), ("intervention", "Intervention Rate", (-0.05, 1.05)),
              ("horizon_s", "Task Horizon (sec)", (2, 7))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, label, ylim) in zip(axes, panels):
        for planner in ["jm2d", "gibbs", "rail", "dp"]:
            cells = summary[metric].get(planner, {})
            if not cells:
                continue
            pads = sorted(cells)
            mean = [cells[p]["mean"] for p in pads]
            legend = DISPLAY[planner] if sources is None else f"{DISPLAY[planner]} [{curve_source(sources, planner)}]"
            ax.plot(pads, mean, marker="o", markersize=8, label=legend, **STYLE[planner])
            ax.fill_between(pads, [cells[p]["min"] for p in pads], [cells[p]["max"] for p in pads], alpha=0.2,
                            color=STYLE[planner]["color"])
            if targets:
                vals = targets["fig4"]["values"][metric].get(planner)
                if vals:
                    tp = [p for p, v in zip(targets["fig4"]["pads"], vals) if v is not None]
                    tv = [v for v in vals if v is not None]
                    ax.scatter(tp, tv, marker="x", s=60, color=STYLE[planner]["color"], zorder=5,
                               label=f"{DISPLAY[planner]} (paper)")
        ax.set_xlabel("Test-time Pad")
        ax.set_ylabel(label)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.5)
    axes[2].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
