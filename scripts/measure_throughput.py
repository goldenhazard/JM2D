"""Measure the cost of one planner: seconds per planning step, steps per trial and peak GPU memory.

    python scripts/measure_throughput.py --label safediffuser-tvs --scenario both --seed 0 --n-trials 1

Runs ``--n-trials`` trials (default 1) and writes ``<out>/<label>_<scenario>_seed<k>.json`` with the per-trial
times, the torch peak memory (``max_memory_reserved``) and the process GPU memory reported by nvidia-smi.
Used to size the sweep (scripts/sweep.py); does not write result files.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jm2d_d3il  # noqa: E402,F401
import torch  # noqa: E402

from jm2d_d3il.experiment import setup  # noqa: E402
from jm2d_d3il.paths import DEFAULT_CKPT_DIR  # noqa: E402
from jm2d_d3il.planners import from_label  # noqa: E402
from jm2d_d3il.runner import run_episodes  # noqa: E402


def process_gpu_mib(pid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # pragma: no cover
        return None
    for line in out.strip().splitlines():
        p, mem = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(mem)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenario", default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    planner = from_label(args.label)
    t0 = time.time()
    task, experiment, env = setup(planner, args.scenario, args.seed, ckpt_dir=args.ckpt_dir)
    t_setup = time.time() - t0

    peak_smi = [0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            m = process_gpu_mib(os.getpid())
            if m is not None:
                peak_smi[0] = max(peak_smi[0], m)
            stop.wait(5.0)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t1 = time.time()
    res = run_episodes(planner, task, env, list(range(args.n_trials)))
    wall = time.time() - t1
    stop.set()
    th.join()
    env.close()

    steps = [r["n_steps"] + 1 for r in res.values()]   # policy calls per trial
    out = dict(label=args.label, scenario=args.scenario, seed=args.seed, n_trials=args.n_trials,
               setup_s=round(t_setup, 1), wall_s=round(wall, 1), policy_calls=steps,
               s_per_step=round(wall / sum(steps), 3), per_trial=res,
               torch_max_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20),
               torch_max_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
               nvidia_smi_peak_mib=peak_smi[0])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.label}_{args.scenario}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_trial"}), flush=True)


if __name__ == "__main__":
    main()
