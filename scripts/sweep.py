"""Resumable job queue for the D3IL re-runs (Table 2 SafeDiffuser / MPD rows, Table 3, reproduction checks).

    python scripts/sweep.py plan   --out results/rerun                 # write results/rerun/queue.json
    python scripts/sweep.py work   --out results/rerun --workers 8     # run (use scripts/launch_sweep.sh to detach)
    python scripts/sweep.py status --out results/rerun                 # progress + ETA

One job = (label, scenario, checkpoint seed) x n trials (``plan --n-trials``, default 100) = one ``scripts/run.py``
subprocess writing ``<out>/<label>/<scenario>/seed<k>.json`` (+ ``.trials.jsonl``, one line per finished trial).
A job is done when its summary JSON exists (whatever its number of trials: use a new ``--out`` to run more trials);
an interrupted or crashed job resumes from its trials file (trials are independent, see
``tests/test_equivalence.py::test_resume_trial_alone``), so ``work`` can be stopped and restarted at any time.
Create ``<out>/STOP`` to stop launching new jobs (running jobs finish). Jobs are started in queue order, subject to a
GPU-memory budget (``--gpu-budget-mib``, per-label estimates in the queue file). Groups, in launch order:

    table2              safediffuser-res, safediffuser-tvs, mpd   Table 2 rows without stored per-seed results
    table2_dpcc_repro   dpcc (um_dpcc)                            reproduction check of the stored Table 2 DPCC row
    table3              jm2d-u0, -u1, -u2, -u5, -u10              Table 3
    table2_ours_repro   jm2d (u = full)                           reproduction check of the stored Table 2 Ours row

Within a group jobs are seed-major (seed 0 of every cell first). Re-planning keeps every trial file: a job is
identified by (label, scenario, seed) only.
Logs: ``<out>/logs/<label>/<scenario>_seed<k>.log`` per job, ``<out>/logs/sweep_master.log``.
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from jm2d_d3il.results import read_trials, run_paths  # noqa: E402
from jm2d_d3il.tasks import load_scenarios  # noqa: E402

SCENARIOS = ["top_left", "top_right", "both", "cluttered"]
GROUPS = [  # (group, labels, purpose) in launch order
    ("table2", ["safediffuser-res", "safediffuser-tvs", "mpd"], "Table 2 rows without stored per-seed results"),
    ("table2_dpcc_repro", ["dpcc"], "reproduction check of the stored Table 2 DPCC row (um_dpcc DPCC)"),
    ("table3", [f"jm2d-u{u}" for u in (0, 1, 2, 5, 10)], "Table 3 (JM2D u-step ablation)"),
    ("table2_ours_repro", ["jm2d"], "reproduction check of the stored Table 2 Ours row (JM2D, u = full)"),
]
# Per label: GPU memory per worker (MiB, nvidia-smi peak incl. CUDA context), seconds per planning step and planning
# steps per trial. s/step: scripts/measure_throughput.py (scenario 'both', 1 trial) with 22 concurrent workers covering
# all labels on an otherwise idle RTX 4090 + 32 CPUs (results/throughput/w20/; GPU at 99 % utilisation, so per-worker
# speed depends on the worker count). steps/trial: stored runs (results/paper, seed-0 partial runs) and the
# measurements. Used for the GPU-memory budget and the ETA only; `status` prefers the timings of finished trials.
DEFAULT_COST = {
    "safediffuser-res": {"gpu_mib": 870, "s_per_step": 9.4, "steps_per_trial": 61},
    "safediffuser-tvs": {"gpu_mib": 870, "s_per_step": 8.5, "steps_per_trial": 61},
    "mpd": {"gpu_mib": 900, "s_per_step": 3.5, "steps_per_trial": 57},
    "dpcc": {"gpu_mib": 740, "s_per_step": 1.36, "steps_per_trial": 63},
    "jm2d-u0": {"gpu_mib": 742, "s_per_step": 2.27, "steps_per_trial": 55},
    "jm2d-u1": {"gpu_mib": 742, "s_per_step": 3.47, "steps_per_trial": 55},
    "jm2d-u2": {"gpu_mib": 742, "s_per_step": 3.56, "steps_per_trial": 55},
    "jm2d-u5": {"gpu_mib": 742, "s_per_step": 3.98, "steps_per_trial": 55},
    "jm2d-u10": {"gpu_mib": 742, "s_per_step": 4.64, "steps_per_trial": 55},
    "jm2d": {"gpu_mib": 742, "s_per_step": 5.24, "steps_per_trial": 53},
}


def label_args(label):
    if label.startswith("safediffuser-"):
        return ["--planner", "safediffuser", "--variant", label.split("-")[1]]
    if label.startswith("jm2d-u"):
        return ["--planner", "jm2d", "--u-steps", label[len("jm2d-u"):]]
    if label == "jm2d":
        return ["--planner", "jm2d", "--u-steps", "full"]
    return ["--planner", label]


def job_id(job):
    return f"{job['label']}/{job['scenario']}/seed{job['seed']}"


def cmd_plan(args):
    unknown = sorted(set(args.scenarios or []) - set(load_scenarios()["scenarios"]))
    if unknown:
        sys.exit(f"unknown scenarios {unknown}: not in configs/scenarios.yaml")
    scenarios = SCENARIOS + [s for s in (args.scenarios or []) if s not in SCENARIOS]  # + scenarios added by the user
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for group, labels, purpose in GROUPS:
        if args.groups and group not in args.groups:
            continue
        for seed in args.seeds:
            for scen in scenarios:
                if args.scenarios and scen not in args.scenarios:
                    continue
                for label in labels:
                    if args.labels and label not in args.labels:
                        continue
                    jobs.append(dict(group=group, label=label, scenario=scen, seed=seed, n_trials=args.n_trials))
    cost = dict(DEFAULT_COST)
    if args.cost and os.path.exists(args.cost):
        with open(args.cost) as f:
            cost.update(json.load(f))
    queue = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), n_trials=args.n_trials,
                 ckpt_dir=args.ckpt_dir, groups={g: dict(labels=l, purpose=p) for g, l, p in GROUPS}, cost=cost,
                 jobs=jobs)
    path = os.path.join(args.out, "queue.json")
    with open(path, "w") as f:
        json.dump(queue, f, indent=1)
    done = sum(job_state(args.out, j)[1] for j in jobs)
    print(f"wrote {path}: {len(jobs)} jobs ({sum(j['n_trials'] for j in jobs)} trials, {done} already finished)")


def load_queue(out):
    with open(os.path.join(out, "queue.json")) as f:
        return json.load(f)


def job_state(out, job):
    json_path, jsonl_path = run_paths(out, job["label"], job["scenario"], job["seed"])
    if os.path.exists(json_path):
        return "done", job["n_trials"]
    return "pending", len(read_trials(jsonl_path))


def gpu_free_mib():
    """Free memory of GPU 0 from nvidia-smi (None if unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def log(out, msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)


def cmd_work(args):
    queue = load_queue(args.out)
    cost = queue["cost"]
    ckpt_dir = args.ckpt_dir or queue.get("ckpt_dir")
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    running = {}     # job id -> (Popen, job, log file, start time)
    attempts = {}    # job id -> consecutive attempts without a new finished trial
    started_with = {}
    retry_after = {}
    stop_file = os.path.join(args.out, "STOP")
    with open(os.path.join(args.out, "sweep_master.pid"), "w") as f:
        f.write(str(os.getpid()))
    log(args.out, f"master pid {os.getpid()} workers {args.workers} budget {args.gpu_budget_mib} MiB "
                  f"jobs {len(queue['jobs'])} python {sys.executable}")

    stopping = {"flag": False}

    def on_signal(signum, frame):
        stopping["flag"] = True
        log(args.out, f"signal {signum}: stop launching, terminating {len(running)} running jobs")
        for proc, _, _, _ in running.values():
            proc.terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    while True:
        # reap finished workers
        for jid in list(running):
            proc, job, fh, t0 = running[jid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[jid]
            state, n_done = job_state(args.out, job)
            log(args.out, f"exit {rc} {jid} after {time.time() - t0:.0f}s: {state} ({n_done}/{job['n_trials']} trials)")
            if state != "done":
                # count only attempts without progress (e.g. a crash or OOM before the first new trial)
                attempts[jid] = 0 if n_done > started_with.get(jid, 0) else attempts.get(jid, 0) + 1
                retry_after[jid] = time.time() + args.retry_delay_s
        if stopping["flag"] and not running:
            break
        pending = []
        for job in queue["jobs"]:
            jid = job_id(job)
            if jid in running:
                continue
            state, _ = job_state(args.out, job)
            if state == "done":
                continue
            if attempts.get(jid, 0) >= args.max_attempts:
                continue
            if retry_after.get(jid, 0) > time.time():
                continue
            pending.append(job)
        if not pending and not running:
            log(args.out, "all jobs finished" + ("" if all(job_state(args.out, j)[0] == "done" for j in queue["jobs"])
                                                 else " (some exceeded --max-attempts; see logs)"))
            break
        if os.path.exists(stop_file) and not running:
            log(args.out, f"{stop_file} present and no job running: exiting")
            break
        launch_ok = not stopping["flag"] and not os.path.exists(stop_file)
        used = sum(cost[j["label"]]["gpu_mib"] for _, j, _, _ in running.values())
        free = gpu_free_mib()
        # jobs started in the last 2 minutes may not have allocated their memory yet
        warming = sum(cost[j["label"]]["gpu_mib"] for _, j, _, t in running.values() if time.time() - t < 120)
        for job in pending:
            if not launch_ok or len(running) >= args.workers:
                break
            need = cost[job["label"]]["gpu_mib"]
            if used + need > args.gpu_budget_mib:
                continue
            if free is not None and free - warming - need < args.gpu_reserve_mib:
                break   # wait for memory (other GPU users); retried at the next poll
            jid = job_id(job)
            log_dir = os.path.join(args.out, "logs", job["label"])
            os.makedirs(log_dir, exist_ok=True)
            fh = open(os.path.join(log_dir, f"{job['scenario']}_seed{job['seed']}.log"), "a")
            cmd = [sys.executable, os.path.join(REPO, "scripts", "run.py"), *label_args(job["label"]),
                   "--scenario", job["scenario"], "--seed", str(job["seed"]), "--n-trials", str(job["n_trials"]),
                   "--out", args.out]
            if ckpt_dir:
                cmd += ["--ckpt-dir", ckpt_dir]
            fh.write(f"\n### {datetime.datetime.now().isoformat(timespec='seconds')} attempt {attempts.get(jid, 0) + 1}: "
                     f"{' '.join(cmd)}\n")
            fh.flush()
            # cwd = logs dir: MuJoCo writes MUJOCO_LOG.TXT into the working directory
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.join(args.out, "logs"),
                                    env=env)
            running[jid] = (proc, job, fh, time.time())
            started_with[jid] = job_state(args.out, job)[1]
            used += need
            warming += need
            log(args.out, f"start {jid} pid {proc.pid} (attempt {attempts.get(jid, 0) + 1}, gpu est {used} MiB)")
        time.sleep(args.poll_s)


def observed_s_per_trial(out, jobs):
    """{label: mean wall seconds per finished trial} from the trials files (avg_time x n_steps per trial)."""
    acc = {}
    for job in jobs:
        _, jsonl_path = run_paths(out, job["label"], job["scenario"], job["seed"])
        for r in read_trials(jsonl_path).values():
            acc.setdefault(job["label"], []).append(r["avg_time"] * max(r["n_steps"], 1))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def cmd_status(args):
    queue = load_queue(args.out)
    cost = queue["cost"]
    purpose = {g: v["purpose"] for g, v in queue.get("groups", {}).items()}
    observed = observed_s_per_trial(args.out, queue["jobs"])
    rows = {}
    order = []
    remaining_s = 0.0
    total_trials = done_trials = 0
    for job in queue["jobs"]:
        group = job.get("group", f"table{job.get('table', '')}")
        state, n = job_state(args.out, job)
        key = (group, job["label"])
        if key not in rows:
            order.append(key)
        r = rows.setdefault(key, dict(jobs=0, done=0, trials=0, trials_done=0))
        r["jobs"] += 1
        r["done"] += state == "done"
        r["trials"] += job["n_trials"]
        r["trials_done"] += n
        total_trials += job["n_trials"]
        done_trials += n
        c = cost[job["label"]]
        per_trial = observed.get(job["label"]) if args.use_observed else None
        per_trial = per_trial or c["s_per_step"] * c.get("steps_per_trial", 60)
        remaining_s += (job["n_trials"] - n) * per_trial
    print(f"{'group':<18} {'label':<18} {'jobs done':>10} {'trials done':>14}  {'s/trial est':>11}")
    for key in order:
        group, label = key
        r = rows[key]
        c = cost[label]
        est = observed.get(label) if args.use_observed and observed.get(label) else c["s_per_step"] * c.get("steps_per_trial", 60)
        src = "obs" if args.use_observed and observed.get(label) else "cfg"
        print(f"{group:<18} {label:<18} {r['done']:>4}/{r['jobs']:<5} {r['trials_done']:>6}/{r['trials']:<7}  "
              f"{est:>7.0f} {src}")
    for g, p in purpose.items():
        print(f"  {g}: {p}")
    workers = args.workers
    print(f"total trials {done_trials}/{total_trials}; ETA with {workers} workers: {remaining_s / workers / 3600:.1f} h "
          f"(remaining worker-hours {remaining_s / 3600:.0f}; s/trial from finished trials where available ('obs'), "
          f"else the queue cost table; assumes per-worker speed stays as measured)")
    pid_file = os.path.join(args.out, "sweep_master.pid")
    if os.path.exists(pid_file):
        pid = int(open(pid_file).read().strip())
        alive = os.path.exists(f"/proc/{pid}")
        print(f"master pid {pid}: {'running' if alive else 'not running'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=os.path.join(REPO, "results", "rerun"))
    p.add_argument("--groups", nargs="*", default=None, choices=[g for g, _, _ in GROUPS],
                   help="restrict to these groups (default: all, in launch order)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--labels", nargs="*", default=None, help="restrict to these labels (default: all labels of the groups)")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="restrict to these scenarios (default: the four paper scenarios; "
                        "any entry of configs/scenarios.yaml)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--cost", default=None, help="JSON {label: {gpu_mib, s_per_step}} overriding the defaults")
    w = sub.add_parser("work")
    w.add_argument("--out", default=os.path.join(REPO, "results", "rerun"))
    w.add_argument("--workers", type=int, default=8)
    w.add_argument("--gpu-budget-mib", type=int, default=15000)
    w.add_argument("--max-attempts", type=int, default=20,
                   help="consecutive attempts without progress per job (crashed jobs resume from their trials file)")
    w.add_argument("--retry-delay-s", type=float, default=120.0, help="wait before relaunching a failed job")
    w.add_argument("--gpu-reserve-mib", type=int, default=2500,
                   help="launch only if the GPU keeps this much free memory after the new job (shared GPU)")
    w.add_argument("--poll-s", type=float, default=10.0)
    w.add_argument("--ckpt-dir", default=None)
    s = sub.add_parser("status")
    s.add_argument("--out", default=os.path.join(REPO, "results", "rerun"))
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--no-observed", dest="use_observed", action="store_false",
                   help="ETA from the queue cost table only (ignore the timings of finished trials)")
    args = ap.parse_args()
    # the run.py jobs start in <out>/logs (cmd_work): relative paths would resolve there
    args.out = os.path.abspath(args.out)
    if getattr(args, "ckpt_dir", None):
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    {"plan": cmd_plan, "work": cmd_work, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
