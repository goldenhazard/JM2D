"""Aggregate results over seeds, compare with the paper, and draw one panel per sampler (CPU only).

Standard deviations are sample standard deviations over runs (ddof = 1; 0 for a single run).
"""
import json
import os
from collections import OrderedDict

import numpy as np

from jm2d_toy.results import load_results
from jm2d_toy.tasks import PAPER_FIGURE, canonical

SAMPLER_ORDER = {"donut_joint": ["sequential", "gibbs", "jm2d"], "donut_constrained": ["projection", "jm2d"]}
TASKS = tuple(SAMPLER_ORDER)
LABELS = {"sequential": "Sequential", "gibbs": "Gibbs", "jm2d": "JM2D (ours)", "projection": "Projection"}
TARGETS_PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "targets.json")


def load_targets(path=TARGETS_PATH):
    with open(path) as f:
        return json.load(f)


def aggregate(rows, task):
    """{sampler: {n, runs, run_ids, n_valid[], n_ood[], valid_mean, valid_std, ood_mean, ood_std}} (fractions of n)."""
    out = OrderedDict()
    for sampler in SAMPLER_ORDER[task] + sorted({r["sampler"] for r in rows} - set(SAMPLER_ORDER[task])):
        sel = [r for r in rows if r["task"] == task and r["sampler"] == sampler]
        if not sel:
            continue
        ns = {r["n"] for r in sel}
        if len(ns) != 1:
            raise ValueError(f"{task}/{sampler}: mixed sample counts {sorted(ns)}")
        n = ns.pop()
        valid = np.array([r["n_valid"] for r in sel], dtype=float)
        ood = np.array([r["n_ood"] for r in sel], dtype=float)
        ddof = 1 if len(sel) > 1 else 0
        out[sampler] = dict(n=n, runs=len(sel), run_ids=[r["run_id"] for r in sel],
                            n_valid=valid.astype(int).tolist(), n_ood=ood.astype(int).tolist(),
                            valid_mean=float(valid.mean() / n), valid_std=float(valid.std(ddof=ddof) / n),
                            ood_mean=float(ood.mean() / n), ood_std=float(ood.std(ddof=ddof) / n))
    return out


def format_table(agg, task):
    valid_name = "compatible" if task == "donut_joint" else "valid"
    lines = [f"{task}: mean +- sample std over runs (fraction of n); per-run counts in brackets",
             f"{'sampler':<12} {'runs':>4} {'n':>4}  {valid_name:<34} {'ood':<34}"]
    for sampler, a in agg.items():
        v = f"{a['valid_mean']:.3f} +- {a['valid_std']:.3f} {a['n_valid']}"
        o = f"{a['ood_mean']:.3f} +- {a['ood_std']:.3f} {a['n_ood']}"
        lines.append(f"{sampler:<12} {a['runs']:>4} {a['n']:>4}  {v:<34} {o:<34}")
    return "\n".join(lines)


def paper_check(agg, task, targets=None):
    """Rows (sampler, metric, paper count, ours mean count, delta, ok) and an ordering check."""
    targets = load_targets() if targets is None else targets
    t = targets[PAPER_FIGURE[task]]
    n_paper = t["n"]
    rows, checks = [], {}
    if task == "donut_joint":
        for sampler, paper in t["values"].items():
            if sampler in agg:
                rows.append((sampler, "n_compatible", paper, agg[sampler]["valid_mean"] * n_paper))
        if all(s in agg for s in SAMPLER_ORDER["donut_joint"]):
            checks["ordering jm2d > gibbs > sequential (compatible)"] = \
                agg["jm2d"]["valid_mean"] > agg["gibbs"]["valid_mean"] > agg["sequential"]["valid_mean"]
    else:
        for sampler, vals in t["values"].items():
            if sampler in agg:
                for metric, key in (("n_valid", "valid_mean"), ("n_ood", "ood_mean")):
                    rows.append((sampler, metric, vals[metric], agg[sampler][key] * n_paper))
        if "jm2d" in agg and "projection" in agg:
            checks["jm2d has no OOD samples"] = agg["jm2d"]["ood_mean"] == 0.0
            checks["projection has OOD samples"] = agg["projection"]["ood_mean"] > 0.0
    rows = [(s, m, p, o, o - p, bool(np.isclose(o, p))) for s, m, p, o in rows]
    return rows, checks


def format_paper_check(rows, checks, task, targets=None):
    targets = load_targets() if targets is None else targets
    figure = PAPER_FIGURE[task]
    lines = [f"{task}: paper Fig. {figure[3:]} ({targets['paper']}) vs ours "
             f"(mean over runs, scaled to n={targets[figure]['n']})",
             f"{'sampler':<12} {'metric':<13} {'paper':>6} {'ours':>8} {'delta':>8}  exact"]
    for sampler, metric, paper, ours, delta, exact in rows:
        lines.append(f"{sampler:<12} {metric:<13} {paper:>6} {ours:>8.2f} {delta:>+8.2f}  {'yes' if exact else 'no'}")
    for name, ok in checks.items():
        lines.append(f"check: {name}: {'PASS' if ok else 'FAIL'}")
    notes = targets[figure].get("notes", {})
    if targets[figure].get("note"):
        lines.append("note: " + targets[figure]["note"])
    for sampler, note in notes.items():
        lines.append(f"note ({sampler}): {note}")
    return "\n".join(lines)


def draw_figure(rows, task, out_path, run_id=None):
    """One panel per sampler, for ``run_id`` (default: the first run of each sampler, e.g. seed0 / paper).

    Runs on the CPU: the problem is only needed for its obstacle geometry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jm2d_toy import tasks
    from jm2d_toy.plotting import plot_constrained_samples, plot_joint_samples

    problem = tasks.make(task, device="cpu")
    samplers = [s for s in SAMPLER_ORDER[task] if any(r["sampler"] == s and r["task"] == task for r in rows)]
    fig, axes = plt.subplots(1, len(samplers), figsize=(5 * len(samplers), 5.2))
    axes = np.atleast_1d(axes)
    for ax, sampler in zip(axes, samplers):
        sel = [r for r in rows if r["task"] == task and r["sampler"] == sampler and "x" in r]
        if run_id is not None:
            sel = [r for r in sel if r["run_id"] == run_id]
        if not sel:
            ax.set_axis_off()
            continue
        r = sel[0]
        n = r["n"]
        if task == "donut_joint":
            title = f"{LABELS[sampler]}: {r['n_valid']}/{n} compatible ({r['run_id']})"
            plot_joint_samples(ax, problem, r["x"], r["k"], r["valid"], title=title, marker_scale=0.2)
        else:
            title = f"{LABELS[sampler]}: {r['n_valid']}/{n} valid, {r['n_ood']}/{n} OOD ({r['run_id']})"
            plot_constrained_samples(ax, problem, r["x"], r["valid"], title=title, legend=(sampler == samplers[-1]))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)  # keep the panel titles
    plt.close(fig)
    return out_path


def compare(results_dir, task, check_paper=False, out=None, run_id=None):
    """Table (and paper check) for ``task`` ("donut_joint" | "donut_constrained"); draws it to ``out`` if given."""
    task = canonical(task)
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    rows = load_results(results_dir, task=task)
    if not rows:
        raise FileNotFoundError(f"no {task} results under {results_dir}")
    agg = aggregate(rows, task)
    text = [format_table(agg, task)]
    if check_paper:
        prow, checks = paper_check(agg, task)
        text.append(format_paper_check(prow, checks, task))
    if out:
        text.append(f"figure: {draw_figure(rows, task, out, run_id=run_id)}")
    return agg, "\n\n".join(text)
