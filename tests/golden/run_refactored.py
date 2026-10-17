"""Re-run one golden case (``tests/golden/manifest.json``) through the refactored ``jm2d_maze`` API.

    python tests/golden/run_refactored.py --case jm2d_pad025_hq --ckpt checkpoints/maze2d_large_ep1900.pth --out DIR

Same protocol as ``capture_legacy.py``: set_random_seed(42) -> build_planner -> rollout_seeds(seeds, horizon),
with the same recorder around the planner's ``__call__``; writes ``<out>/<case>.npz`` and ``<out>/<case>.json``.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from capture_legacy import episodes_to_arrays, install_recorder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    args = parser.parse_args()

    with open(args.manifest) as f:
        info = json.load(f)["cases"][args.case]

    import d4rl
    import robomimic
    import safediffusion.algo.jm2d_policy as JM2DPolicyModule
    import jm2d_maze
    from jm2d_maze.planners import build_planner
    from jm2d_maze.rollout import rollout_seeds
    from jm2d_maze.seeding import set_random_seed

    modules = {"jm2d_maze": jm2d_maze.__file__, "safediffusion": JM2DPolicyModule.__file__,
               "robomimic": robomimic.__file__, "d4rl": d4rl.__file__}
    for key in ("jm2d_maze", "safediffusion", "robomimic"):
        assert os.path.abspath(modules[key]).startswith(REPO_ROOT), f"{key} resolved to {modules[key]}"

    set_random_seed(42)
    planner, env, eval_env = build_planner(info["planner"], os.path.abspath(args.ckpt), pad=info["pad"],
                                           backup=info["backup"], num_workers=args.workers)
    episodes = []
    install_recorder(planner, episodes)
    t0 = time.time()
    stats = rollout_seeds(planner, env, eval_env, horizon=info["horizon"], seeds=np.array(info["seeds"]))
    elapsed = time.time() - t0
    if hasattr(planner, "pool"):
        planner.pool.close()
        planner.pool.join()

    os.makedirs(args.out, exist_ok=True)
    arrays = episodes_to_arrays(episodes)
    np.savez(os.path.join(args.out, args.case + ".npz"), **arrays)
    per_seed = {str(s): {k: v for k, v in stats[str(s)].items() if k != "Time"} for s in info["seeds"]}
    replan_times = [r["time_s"] for ep in episodes for r in ep["replans"]]
    with open(os.path.join(args.out, args.case + ".json"), "w") as f:
        json.dump(dict(stats=per_seed, mean_replan_time_s=float(np.mean(replan_times)),
                       replan_times_s=replan_times, wall_time_s=round(elapsed, 1), modules=modules,
                       pool_workers=getattr(planner, "num_processes", None)), f, indent=2)


if __name__ == "__main__":
    main()
