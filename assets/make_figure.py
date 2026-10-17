"""Draw ``assets/trajectories.png`` (README Overview) from the end-effector paths in ``assets/traj/``.

    python assets/make_figure.py [--results results/runs] [--out assets/trajectories.png]

``assets/traj/<label>_<scenario>_seed0.traj.npz`` are the trajectory files ``results/runs/<label>/<scenario>/
seed0.traj.npz`` of the runs below, copied here; ``--results DIR`` reads ``DIR/<label>/<scenario>/seed0.traj.npz``
instead. No GPU needed.

    python scripts/run.py --planner jm2d --scenario top_left top_right both cluttered --seed 0 --n-trials 10 --plot
    python scripts/run.py --planner dpcc --scenario cluttered --seed 0 --n-trials 10 --plot
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from matplotlib.figure import Figure  # noqa: E402

from jm2d_d3il.plotting import draw_paths, draw_scene, legend_handles  # noqa: E402
from jm2d_d3il.results import read_traj  # noqa: E402

PANELS = [("jm2d", "top_left"), ("jm2d", "top_right"), ("jm2d", "both"), ("jm2d", "cluttered"), ("dpcc", "cluttered")]
NAMES = {"jm2d": "JM2D (Ours)", "dpcc": "DPCC"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="output directory of scripts/run.py (default: assets/traj/)")
    ap.add_argument("--out", default=os.path.join(HERE, "trajectories.png"))
    args = ap.parse_args()

    fig = Figure(figsize=(12.5, 4.1))
    grid = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.12, 1], left=0.04, right=0.995, bottom=0.2, top=0.9,
                            wspace=0.08)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 3, 5)]
    for i, (ax, (label, scenario)) in enumerate(zip(axes, PANELS)):
        path = (os.path.join(args.results, label, scenario, "seed0.traj.npz") if args.results else
                os.path.join(HERE, "traj", f"{label}_{scenario}_seed0.traj.npz"))
        meta, rows = read_traj(path)
        draw_scene(ax, meta)
        counts = draw_paths(ax, rows)
        ax.set_title(f"{NAMES[label]}, {scenario}\n{counts['ssucc']}/{len(rows)} safe success", fontsize=9.5)
        ax.set_xticks([0.3, 0.5, 0.7])
        ax.set_xlabel("x [m]", fontsize=8)
        if i == 0 or label != PANELS[i - 1][0]:
            ax.set_ylabel("y [m]", fontsize=8)
        else:
            ax.tick_params(labelleft=False)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
