"""Draw saved end-effector paths (``scripts/run.py --plot / --save-traj``) as a top view of the task.

    python scripts/plot_trajectories.py --results results/runs
    python scripts/plot_trajectories.py --results results/runs --label jm2d --scenario cluttered --seed 0

Every ``<results>/<label>/<scenario>/seed<k>.traj.npz`` that matches the filters is drawn to ``seed<k>.png`` next to
it (jm2d_d3il/plotting.py): the scenario's constraints, the D3IL obstacles, the start, the finish line and each trial's
path colored by outcome (success without violation / success with violation / failure). No GPU needed.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jm2d_d3il.plotting import plot_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="output directory of scripts/run.py (--out)")
    ap.add_argument("--label", nargs="+", default=None, help="labels, e.g. jm2d dpcc (default: all)")
    ap.add_argument("--scenario", nargs="+", default=None, help="scenarios (default: all)")
    ap.add_argument("--seed", nargs="+", type=int, default=None, help="checkpoint seeds (default: all)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results, "*", "*", "seed*.traj.npz")))
    n = 0
    for path in files:
        label, scenario = path.split(os.sep)[-3:-1]
        seed = int(os.path.basename(path)[len("seed"):-len(".traj.npz")])
        if ((args.label and label not in args.label) or (args.scenario and scenario not in args.scenario)
                or (args.seed is not None and seed not in args.seed)):
            continue
        png = path[:-len(".traj.npz")] + ".png"
        n_trials = plot_run(path, png)
        print(f"wrote {png} ({n_trials} trials)")
        n += 1
    if n == 0:
        sys.exit(f"no trajectory files (<label>/<scenario>/seed<k>.traj.npz) match in {args.results}; "
                 f"run scripts/run.py with --plot or --save-traj")


if __name__ == "__main__":
    main()
