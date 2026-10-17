"""Top view of D3IL-Avoiding trials (as paper Fig. 11): the scenario's constraints, the D3IL obstacles, the start, the
finish line and the end-effector path of every trial, colored by outcome.

Draws the trajectory files written by ``scripts/run.py --save-traj / --plot`` (``results.write_traj``). The
constraints are the rows the runner counts violations of (``AvoidingTask.xy_constraints``): halfspaces are shaded
where they are violated, inside the workspace bounds; discs are drawn with their radius. matplotlib only, through
``matplotlib.figure.Figure`` (no pyplot state, no backend change).
"""
import numpy as np

from .results import read_traj

# D3IL Avoiding environment (fixed): the six cylinders of avoiding_objects.get_obj_list() as (x, y, radius) (touching
# one ends the episode as a failure) and the finish line (ObstacleAvoidanceEnv.goal_ypos; success iff y > 0.35).
ENV_OBSTACLES = ((0.5, -0.1, 0.03), (0.425, 0.08, 0.025), (0.575, 0.08, 0.025), (0.35, 0.26, 0.025),
                 (0.5, 0.26, 0.025), (0.65, 0.26, 0.025))
FINISH_Y = 0.35

OUTCOMES = {  # outcome -> (legend label, color, line style); drawn in this order (rarer outcomes on top)
    "ssucc": ("success, no violation", "#2a78d6", "-"),
    "violated": ("success with violation", "#eb6834", "-"),
    "failure": ("failure (end: x)", "#52514e", "--"),
}
CONSTRAINT_FACE, CONSTRAINT_EDGE, FILL_ALPHA = "#9085e9", "#4a3aa7", 0.3
OBSTACLE_COLOR = "#d03b3b"
GOAL_COLOR = "#0ca30c"
SURFACE = "#fcfcfb"


def outcome(row):
    return "ssucc" if row["ssucc"] else ("violated" if row["success"] else "failure")


def count_outcomes(rows):
    counts = {k: 0 for k in OUTCOMES}
    for row in rows.values():
        counts[outcome(row)] += 1
    return counts


def violated_part(poly, a, b, d):
    """Part of the convex polygon ``poly`` [(x, y), ...] where a x + b y >= d (one Sutherland-Hodgman step)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b * p[1] - d, a * q[0] + b * q[1] - d
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def draw_scene(ax, meta):
    """Workspace, goal region, constraints and D3IL obstacles of a trajectory file's ``meta``."""
    from matplotlib.patches import Circle, Polygon
    (x0, x1), (y0, y1) = np.asarray(meta["xlim"], dtype=float), np.asarray(meta["ylim"], dtype=float)
    ax.set_facecolor(SURFACE)
    ax.axhspan(FINISH_Y, y1, color=GOAL_COLOR, alpha=0.10, lw=0, zorder=0)
    ax.axhline(FINISH_Y, color=GOAL_COLOR, lw=2.5, zorder=1)
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for a, b, d in np.asarray(meta["halfspaces"], dtype=float).reshape(-1, 3):
        poly = violated_part(box, a, b, d)
        if len(poly) >= 3:
            ax.add_patch(Polygon(poly, closed=True, fc=CONSTRAINT_FACE, alpha=FILL_ALPHA, lw=0, zorder=1))
        p, q = ((0.0, d / b), (1.0, (d - a) / b)) if b != 0 else ((d / a, 0.0), (d / a, 1.0))
        ax.axline(p, q, color=CONSTRAINT_EDGE, lw=1, zorder=1)
    for cx, cy, r in np.asarray(meta["discs"], dtype=float).reshape(-1, 3):
        ax.add_patch(Circle((cx, cy), r, fc=CONSTRAINT_FACE, alpha=FILL_ALPHA, lw=0, zorder=1))
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=CONSTRAINT_EDGE, lw=1, zorder=1))
    for cx, cy, r in ENV_OBSTACLES:
        ax.add_patch(Circle((cx, cy), r, fc=OBSTACLE_COLOR, ec="none", zorder=2))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")


def draw_paths(ax, rows, lw=1.2):
    """One line per trial, colored by outcome; failures end with an x. Returns the outcome counts."""
    for key, (_, color, ls) in OUTCOMES.items():
        for t in sorted(rows):
            if outcome(rows[t]) != key:
                continue
            path = rows[t]["path"]
            ax.plot(path[:, 0], path[:, 1], color=color, ls=ls, lw=lw, alpha=0.85, zorder=3)
            if key == "failure":
                ax.plot(path[-1, 0], path[-1, 1], "x", color=color, ms=6, mew=1.5, zorder=3)
    if rows:
        start = rows[min(rows)]["path"][0]
        ax.plot(start[0], start[1], "o", ms=7, mfc="white", mec="#0b0b0b", mew=1.5, zorder=4)
    return count_outcomes(rows)


def legend_handles(counts=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=color, ls=ls, lw=1.5,
                      label=label if counts is None else f"{label}: {counts[key]}")
               for key, (label, color, ls) in OUTCOMES.items()]
    handles += [Patch(fc=CONSTRAINT_FACE, alpha=FILL_ALPHA, ec=CONSTRAINT_EDGE, label="constraint (violated region)"),
                Line2D([], [], ls="none", marker="o", ms=7, mfc=OBSTACLE_COLOR, mec="none", label="D3IL obstacle"),
                Line2D([], [], color=GOAL_COLOR, lw=2.5, label="finish line (y = 0.35)"),
                Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec="#0b0b0b", mew=1.5, label="start")]
    return handles


def run_title(meta, rows):
    c = count_outcomes(rows)
    return (f"{meta['label']} | {meta['scenario']} | seed {meta['seed']}\n"
            f"{len(rows)} trials: {c['ssucc']} safe success, {c['violated']} success with violation, "
            f"{c['failure']} failure")


def plot_run(traj_file, png_path, trials=None, dpi=150):
    """Figure of one trajectory file (only ``trials`` if given) -> ``png_path``; returns the number of trials drawn."""
    from matplotlib.figure import Figure
    meta, rows = read_traj(traj_file)
    if trials is not None:
        keep = set(trials)
        rows = {t: r for t, r in rows.items() if t in keep}
    fig = Figure(figsize=(4.6, 6.2))
    ax = fig.subplots()
    draw_scene(ax, meta)
    counts = draw_paths(ax, rows)
    ax.set_title(run_title(meta, rows), fontsize=9)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.legend(handles=legend_handles(counts), loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7,
              frameon=False)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return len(rows)
