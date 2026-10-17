"""Run D3IL-Avoiding evaluation episodes for one planner.

    python scripts/run.py --planner jm2d --scenario top_left --seed 0 --n-trials 100
    python scripts/run.py --planner jm2d --u-steps 5 --scenario cluttered --seed 0-4
    python scripts/run.py --planner safediffuser --variant tvs --scenario both --seed 0 --n-trials 10
    python scripts/run.py --planner dpcc --scenario top_left top_right --seed 0 --ckpt-dir /path/to/H8_K20_Dmodels.GaussianDiffusion

Writes ``<out>/<label>/<scenario>/seed<k>.json`` (summary, see jm2d_d3il/results.py) and ``seed<k>.trials.jsonl``
(one line per trial). An interrupted run resumes from the trials file (``--no-resume`` starts over).
Label = planner (+ ``-res``/``-tvs`` for SafeDiffuser, ``-u<u>`` for JM2D with --u-steps).
``--save-traj`` also writes the end-effector path of every trial to ``seed<k>.traj.npz``; ``--plot`` does that and
draws them to ``seed<k>.png`` (top view, jm2d_d3il/plotting.py; redraw with scripts/plot_trajectories.py). Neither
changes the trials or their results.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jm2d_d3il  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG and sys.path before torch / d3il are imported)
from jm2d_d3il.experiment import make_planner, run_job  # noqa: E402
from jm2d_d3il.paths import DEFAULT_CKPT_DIR, RESULTS_DIR  # noqa: E402
from jm2d_d3il.planners import REGISTRY  # noqa: E402
from jm2d_d3il.results import read_traj, traj_path  # noqa: E402
from jm2d_d3il.tasks import load_scenarios  # noqa: E402


def parse_seeds(items):
    seeds = []
    for item in items:
        for part in str(item).split(","):
            if "-" in part:
                a, b = part.split("-")
                seeds.extend(range(int(a), int(b) + 1))
            elif part:
                seeds.append(int(part))
    return seeds


def parse_u(value):
    return value if value == "full" else int(value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--variant", choices=["res", "tvs"], help="SafeDiffuser invariance variant (default: config)")
    ap.add_argument("--u-steps", type=parse_u, default=None,
                    help="JM2D clean-sample estimate: 'full' (Table 2, default) or an integer u (Table 3)")
    ap.add_argument("--scenario", nargs="+", required=True, choices=list(load_scenarios()["scenarios"]),
                    help="scenarios of configs/scenarios.yaml")
    ap.add_argument("--seed", nargs="+", default=["0"], help="checkpoint seeds, e.g. 0 or 0-4 or 0,2")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR, help="directory with <seed>/state_best.pt + config pickles")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "runs"))
    ap.add_argument("--config", default=None, help="planner YAML (default: configs/planners/<planner>.yaml)")
    ap.add_argument("--batch-size", type=int, default=None, help="override the planner batch size")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="JM2D: print ESS / valid ratio every denoising step")
    ap.add_argument("--save-traj", action="store_true", help="also save the end-effector paths (seed<k>.traj.npz)")
    ap.add_argument("--plot", action="store_true", help="save the paths and draw them (seed<k>.png)")
    args = ap.parse_args(argv)

    for scenario in args.scenario:
        for seed in parse_seeds(args.seed):
            extra = {"batch_size": args.batch_size}
            if args.planner == "jm2d" and args.verbose:
                extra["verbose"] = True
            planner = make_planner(args.planner, variant=args.variant, u_steps=args.u_steps, config=args.config,
                                   **extra)
            run_job(planner, scenario, seed, n_trials=args.n_trials, out_dir=args.out, ckpt_dir=args.ckpt_dir,
                    resume=not args.no_resume, save_traj=args.save_traj or args.plot)
            if args.save_traj or args.plot:
                traj_file = traj_path(args.out, planner.label, scenario, seed)
                missing = [t for t in range(args.n_trials) if t not in read_traj(traj_file)[1]]
                if missing:
                    print(f"[run] {traj_file}: no path for trials {missing[:5]}{'...' if len(missing) > 5 else ''} "
                          f"(run without --save-traj / --plot; --no-resume re-runs them)", flush=True)
                if args.plot and len(missing) < args.n_trials:
                    from jm2d_d3il.plotting import plot_run
                    png = traj_file[:-len(".traj.npz")] + ".png"
                    plot_run(traj_file, png, trials=range(args.n_trials))
                    print(f"[run] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
