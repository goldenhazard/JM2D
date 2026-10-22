#!/usr/bin/env python
"""Resumable re-run of the Fig. 4 DP and RAIL curves with the released code.

    python scripts/sweep_fig4.py plan   --out results/fig4_rerun                # write results/fig4_rerun/queue.json
    python scripts/sweep_fig4.py work   --out results/fig4_rerun --workers 4    # run (scripts/launch_fig4.sh detaches it)
    python scripts/sweep_fig4.py status --out results/fig4_rerun                # progress + ETA
    python scripts/sweep_fig4.py job    --out results/fig4_rerun --planner rail --ckpt ep1900 --pad 0.25   # one job

Why: the stored Fig. 4 DP and RAIL runs (results/paper) were made with research code that conditioned the DP on the
current frame twice instead of (previous, current) frame; the released code has the fix.
JM2D and Gibbs were never affected and keep their stored runs.

Jobs: planner {rail, dp} x checkpoint {ep1900, ep2000, ep1250, ep1500, ep1750} x pad {0, .05, .10, .15, .20, .25}
= 60 jobs x 100 env seeds. Protocol of the stored runs: ``set_random_seed(42)`` -> build -> env seeds 0..99 in order
(``set_random_seed(seed)`` per episode), horizon 2000, HQ backup, ``configs/maze2d_large.json``.

Output per job (same schema as the stored runs, no videos / pickles)::

    <out>/large_pad<www>/<planner>_<ckpt>/seed<k>/stats.json   per episode, written when the episode ends
    <out>/large_pad<www>/<planner>_<ckpt>/stats.json           per-seed entries + aggregate, written at the end
    <out>/large_pad<www>/<planner>_<ckpt>/{config.json, meta.json, episodes.jsonl}   config, provenance, timings

Resume: a job is done when its ``stats.json`` holds all seeds. An interrupted job restarts after the last finished
seed: the environment RNG chain is fast-forwarded through the finished seeds (two ``env.reset()`` per seed, exactly
what ``rollout_seeds`` does) without simulating them; in the smoke test (dp and rail, ep1900, w = 0.25) a job stopped
after 2 seeds and resumed to 5 gave the same per-seed stats as an uninterrupted run (5/5 each).
Results depend on the CPU thread count at round-off level (RAIL: IPOPT and CPU zonotope algebra), so the launcher fixes
1 thread per job (recorded in meta.json); the golden tests were captured with the default thread count. ``touch <out>/STOP`` stops launching new jobs (running jobs finish);
``kill $(cat <out>/sweep_master.pid)`` stops everything (jobs resume on the next launch).
Logs: ``<out>/logs/<planner>_<ckpt>_pad<www>.log`` per job (episode lines; ``--verbose`` adds the planner output),
``<out>/logs/sweep_master.log``.
"""
import argparse
import contextlib
import datetime
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

PLANNERS = ["rail", "dp"]
CKPTS = {  # tag -> file in checkpoints/ (scripts/download_checkpoints.py)
    "ep1900": "maze2d_large_ep1900.pth", "ep2000": "maze2d_large_ep2000.pth", "ep1250": "maze2d_large_ep1250.pth",
    "ep1500": "maze2d_large_ep1500.pth", "ep1750": "maze2d_large_ep1750.pth",
}
PADS = [0.25, 0.20, 0.15, 0.10, 0.05, 0.0]
N_SEEDS = 100
HORIZON = 2000
PURPOSE = "Fig. 4 DP/RAIL re-run with the released code"  # recorded in queue.json and meta.json
# Cost model, measured in the smoke test (RTX 4090 shared with a D3IL sweep at ~99 % GPU utilisation, 3 jobs at
# a time, OMP/MKL threads = 1): wall seconds per replanning cycle including the 15 other control steps of the cycle;
# GPU memory per job process (nvidia-smi): steady state after the build (torch.cuda.empty_cache) and the transient
# peak while building (the checkpoint is loaded twice, as in the research code). Workload per job = replanning cycles
# of the stored run of the same cell (meant as an upper bound for RAIL: the 20-episode comparison in results/repro
# had ~15-20 % shorter re-run horizons; the full re-run turned out at 0.95-1.04x the stored horizons).
COST = {
    "dp": {"s_per_replan": 0.07, "s_per_step": 0.0, "gpu_mib": 1550, "gpu_peak_mib": 3000},
    "rail": {"s_per_replan": 0.62, "s_per_step": 0.0, "gpu_mib": 1550, "gpu_peak_mib": 3000},
}
WARMUP_S = 90  # a job counts with its peak memory for this long after its start


def pad_dir(pad):
    return f"large_pad{int(round(float(pad) * 100)):03d}"


def run_name(job):
    return f"{job['planner']}_{job['ckpt']}"


def job_id(job):
    return f"{run_name(job)}/{pad_dir(job['pad'])}"


def job_dir(out, job):
    return os.path.join(out, pad_dir(job["pad"]), run_name(job))


def log_path(out, job):
    return os.path.join(out, "logs", f"{run_name(job)}_{pad_dir(job['pad'])}.log")


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def finished_seeds(run_dir, n_seeds):
    """Leading run of finished seeds 0, 1, ... (episodes must run in order)."""
    k = 0
    while k < n_seeds and _load_json(os.path.join(run_dir, f"seed{k}", "stats.json")) is not None:
        k += 1
    return k


def job_state(out, job):
    run_dir = job_dir(out, job)
    agg = _load_json(os.path.join(run_dir, "stats.json"))
    if agg is not None and agg.get("total_runs") == job["n_seeds"]:
        return "done", job["n_seeds"]
    return "pending", finished_seeds(run_dir, job["n_seeds"])


# ------------------------------------------------------------------------------------------------ cost / ETA
def stored_plans(planner, pad, ckpt):
    """(sum of replanning cycles, sum of control steps) of the stored run of this cell (workload proxy), or None."""
    index = _load_json(os.path.join(REPO, "results", "paper", "index.json")) or {"runs": []}
    for e in index["runs"]:
        if e["figure"] == "fig4" and e["planner"] == planner and abs(e["pad"] - pad) < 1e-9 and e["ckpt"] == ckpt:
            stats = _load_json(os.path.join(REPO, "results", "paper", e["path"], "stats.json"))
            seeds = [stats[k] for k in stats if k.isdigit()]
            return sum(s["Plan"] for s in seeds), sum(s["Horizon"] for s in seeds), len(seeds)
    return None


def estimate_job_s(job, cost):
    c = cost[job["planner"]]
    ref = stored_plans(job["planner"], job["pad"], job["ckpt"])
    if ref is None:  # fall back to the mean over checkpoints of this cell
        refs = [stored_plans(job["planner"], job["pad"], ck) for ck in CKPTS]
        refs = [r for r in refs if r]
        ref = tuple(sum(r[i] for r in refs) / len(refs) for i in range(3))
    plans, steps, n = ref
    per_episode = (plans * c["s_per_replan"] + (steps - plans) * c["s_per_step"]) / n
    return per_episode * job["n_seeds"]


# ------------------------------------------------------------------------------------------------ plan
def cmd_plan(args):
    os.makedirs(args.out, exist_ok=True)
    jobs = [dict(planner=p, ckpt=ck, pad=pad, n_seeds=args.n_seeds, horizon=args.horizon)
            for p in args.planners for ck in args.ckpts for pad in args.pads]
    for j in jobs:
        j["est_s"] = round(estimate_job_s(j, COST), 1)
    jobs.sort(key=lambda j: -j["est_s"])  # longest first (shorter tail)
    queue = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), purpose=PURPOSE,
                 protocol="set_random_seed(42) -> build_planner -> env seeds 0..n-1 in order, HQ backup, "
                          "configs/maze2d_large.json", cost=COST, jobs=jobs)
    path = os.path.join(args.out, "queue.json")
    with open(path, "w") as f:
        json.dump(queue, f, indent=1)
    done = sum(job_state(args.out, j)[0] == "done" for j in jobs)
    print(f"wrote {path}: {len(jobs)} jobs, {sum(j['n_seeds'] for j in jobs)} episodes, {done} jobs already done; "
          f"estimated {sum(j['est_s'] for j in jobs) / 3600:.1f} worker-hours")


def load_queue(out):
    with open(os.path.join(out, "queue.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ job
def cmd_job(args):
    import jm2d_maze  # noqa: F401  (CUBLAS_WORKSPACE_CONFIG)
    import torch
    from jm2d_maze.config import load_config
    from jm2d_maze.planners import build_planner
    from jm2d_maze.results import EpisodeResult, md5, versions
    from jm2d_maze.rollout import compute_aggregate_stats, rollout_seeds
    from jm2d_maze.seeding import set_random_seed

    job = dict(planner=args.planner, ckpt=args.ckpt, pad=float(args.pad), n_seeds=args.n_seeds, horizon=args.horizon)
    run_dir = job_dir(args.out, job)
    os.makedirs(run_dir, exist_ok=True)
    state, n_done = job_state(args.out, job)
    if state == "done":
        print(f"{job_id(job)}: already done", flush=True)
        return
    ckpt_path = os.path.join(args.ckpt_dir, CKPTS[args.ckpt])
    config = load_config(args.config, planner=args.planner)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(config.dump())
    meta_path = os.path.join(run_dir, "meta.json")
    meta = _load_json(meta_path) or dict(
        figure="fig4", purpose=PURPOSE, planner=args.planner, pad=job["pad"], backup="hq", ckpt=args.ckpt,
        ckpt_file=CKPTS[args.ckpt], ckpt_md5=md5(ckpt_path), n_seeds=args.n_seeds, horizon=args.horizon,
        protocol="set_random_seed(42) -> build_planner -> env seeds 0..n-1 in order (set_random_seed(seed) each)",
        seed_gym_spaces=True, versions=versions(), torch_threads=torch.get_num_threads(), attempts=[])
    meta["attempts"].append(dict(start=datetime.datetime.now().isoformat(timespec="seconds"), resume_from_seed=n_done,
                                 host_threads=torch.get_num_threads()))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    seeds = list(range(args.n_seeds))
    todo = seeds[n_done:]
    episodes_path = os.path.join(run_dir, "episodes.jsonl")
    t_last = [time.time()]

    def on_episode(seed, stat, traj):
        wall = time.time() - t_last[0]
        seed_dir = os.path.join(run_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "stats.json.tmp"), "w") as f:
            json.dump(stat, f, indent=4)
        os.replace(os.path.join(seed_dir, "stats.json.tmp"), os.path.join(seed_dir, "stats.json"))
        rec = EpisodeResult.from_legacy_stat(stat, task=f"maze_{pad_dir(job['pad'])[len('large_'):]}_hq",
                                             planner=args.planner, pad=job["pad"], backup="hq", ckpt=args.ckpt,
                                             env_seed=int(seed), source="fig4_rerun", extra={"wall_s": round(wall, 2)})
        with open(episodes_path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
        print(f"{job_id(job)} seed{seed}: success={stat['Success']} unsafe={stat['Unsafe']} "
              f"int={stat['Intervention']}/{stat['Plan']} horizon={stat['Horizon']} wall={wall:.1f}s",
              file=sys.__stdout__, flush=True)
        t_last[0] = time.time()

    sink = None if args.verbose else open(os.devnull, "w")
    with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
        set_random_seed(42)
        planner, env, eval_env = build_planner(args.planner, ckpt_path, pad=job["pad"], backup="hq", config=config)
        # release the CUDA cache of the build (the checkpoint is loaded twice, as in the research code): ~3 GB -> see
        # COST; allocator-only, no effect on the computation
        torch.cuda.empty_cache()
        # fast-forward the environment RNG through the finished seeds (rollout_seeds: reset + get_goal, rollout: reset)
        for seed in seeds[:n_done]:
            set_random_seed(seed, deterministic_torch=False)
            env.reset()
            env.get_goal()
            env.reset()
        if n_done:
            print(f"{job_id(job)}: resuming at seed {n_done}", file=sys.__stdout__, flush=True)
        t_last[0] = time.time()
        if todo:
            rollout_seeds(planner, env, eval_env, horizon=args.horizon, seeds=todo, on_episode=on_episode)
        if hasattr(planner, "pool"):
            planner.pool.close()
            planner.pool.join()
    stats = {str(s): _load_json(os.path.join(run_dir, f"seed{s}", "stats.json")) for s in seeds}
    missing = [s for s, v in stats.items() if v is None]
    if missing:
        raise SystemExit(f"{job_id(job)}: missing seeds {missing[:5]}")
    stats.update(compute_aggregate_stats(stats))
    with open(os.path.join(run_dir, "stats.json.tmp"), "w") as f:
        json.dump(stats, f, indent=4)
    os.replace(os.path.join(run_dir, "stats.json.tmp"), os.path.join(run_dir, "stats.json"))
    meta["attempts"][-1]["end"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{job_id(job)}: done, success_and_safe_rate {stats['success_and_safe_rate']:.3f}, "
          f"intervention_rate {stats['intervention_rate']:.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ work
def gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def log(msg):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def cmd_work(args):
    queue = load_queue(args.out)
    cost = queue["cost"]
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, str(args.threads))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("MPLBACKEND", "Agg")
    running, attempts, started_with, retry_after = {}, {}, {}, {}
    stop_file = os.path.join(args.out, "STOP")
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    with open(os.path.join(args.out, "sweep_master.pid"), "w") as f:
        f.write(str(os.getpid()))
    log(f"master pid {os.getpid()} workers {args.workers} gpu budget {args.gpu_budget_mib} MiB threads "
        f"{env['OMP_NUM_THREADS']} jobs {len(queue['jobs'])} python {sys.executable}")
    stopping = {"flag": False}

    def on_signal(signum, frame):
        stopping["flag"] = True
        log(f"signal {signum}: stop launching, terminating {len(running)} running jobs")
        for proc, _, _, _ in running.values():
            proc.terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while True:
        for jid in list(running):
            proc, job, fh, t0 = running[jid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[jid]
            state, n_done = job_state(args.out, job)
            log(f"exit {rc} {jid} after {time.time() - t0:.0f}s: {state} ({n_done}/{job['n_seeds']} seeds)")
            if state != "done":
                attempts[jid] = 0 if n_done > started_with.get(jid, 0) else attempts.get(jid, 0) + 1
                retry_after[jid] = time.time() + args.retry_delay_s
        if stopping["flag"] and not running:
            break
        unfinished = [j for j in queue["jobs"] if job_id(j) not in running and job_state(args.out, j)[0] != "done"
                      and attempts.get(job_id(j), 0) < args.max_attempts]
        pending = [j for j in unfinished if retry_after.get(job_id(j), 0) <= time.time()]
        if not unfinished and not running:
            all_done = all(job_state(args.out, j)[0] == "done" for j in queue["jobs"])
            log("all jobs finished" + ("" if all_done else " (some exceeded --max-attempts; see logs)"))
            break
        if os.path.exists(stop_file) and not running:
            log(f"{stop_file} present and no job running: exiting")
            break
        launch_ok = not stopping["flag"] and not os.path.exists(stop_file)
        # budget: steady memory of running jobs, peak memory of jobs still building (and of the job to start)
        now = time.time()
        used = sum(cost[j["planner"]]["gpu_peak_mib" if now - t < WARMUP_S else "gpu_mib"]
                   for _, j, _, t in running.values())
        free = gpu_free_mib()
        warming = sum(cost[j["planner"]]["gpu_peak_mib"] for _, j, _, t in running.values() if now - t < WARMUP_S)
        for job in pending:
            if not launch_ok or len(running) >= args.workers:
                break
            need = cost[job["planner"]]["gpu_peak_mib"]
            if used + need > args.gpu_budget_mib:
                break
            if free is not None and free - warming - need < args.gpu_reserve_mib:
                break
            jid = job_id(job)
            fh = open(log_path(args.out, job), "a")
            cmd = [sys.executable, os.path.abspath(__file__), "job", "--out", args.out, "--planner", job["planner"],
                   "--ckpt", job["ckpt"], "--pad", str(job["pad"]), "--n-seeds", str(job["n_seeds"]),
                   "--horizon", str(job["horizon"]), "--ckpt-dir", args.ckpt_dir]
            fh.write(f"\n### {datetime.datetime.now().isoformat(timespec='seconds')} attempt "
                     f"{attempts.get(jid, 0) + 1}: {' '.join(cmd)}\n")
            fh.flush()
            # cwd = logs dir: MuJoCo writes MUJOCO_LOG.TXT into the working directory
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.join(args.out, "logs"),
                                    env=env, start_new_session=False)
            running[jid] = (proc, job, fh, time.time())
            started_with[jid] = job_state(args.out, job)[1]
            used += need
            warming += need
            log(f"start {jid} pid {proc.pid} (attempt {attempts.get(jid, 0) + 1}, from seed {started_with[jid]}, "
                f"gpu est {used} MiB)")
        time.sleep(args.poll_s)


# ------------------------------------------------------------------------------------------------ status
def observed_episode_s(out, jobs):
    """{(planner, pad): mean wall seconds per finished episode} from episodes.jsonl."""
    acc = {}
    for job in jobs:
        path = os.path.join(job_dir(out, job), "episodes.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                acc.setdefault((job["planner"], job["pad"]), []).append(r["extra"]["wall_s"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def cmd_status(args):
    queue = load_queue(args.out)
    observed = observed_episode_s(args.out, queue["jobs"])
    rows, remaining_s, n_eps, n_done_eps = {}, 0.0, 0, 0
    for job in queue["jobs"]:
        state, n = job_state(args.out, job)
        r = rows.setdefault((job["planner"], job["pad"]), dict(jobs=0, done=0, eps=0, eps_done=0))
        r["jobs"] += 1
        r["done"] += state == "done"
        r["eps"] += job["n_seeds"]
        r["eps_done"] += n
        n_eps += job["n_seeds"]
        n_done_eps += n
        per_ep = observed.get((job["planner"], job["pad"])) or job["est_s"] / job["n_seeds"]
        remaining_s += (job["n_seeds"] - n) * per_ep
    print(f"{'planner':8s} {'pad':>5s} {'jobs done':>10s} {'episodes':>10s} {'s/episode':>10s}")
    for (planner, pad), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        obs = observed.get((planner, pad))
        print(f"{planner:8s} {pad:5.2f} {r['done']:>4d}/{r['jobs']:<5d} {r['eps_done']:>4d}/{r['eps']:<5d} "
              f"{(obs if obs else float('nan')):>7.1f} {'obs' if obs else 'est'}")
    print(f"episodes {n_done_eps}/{n_eps}; remaining {remaining_s / 3600:.1f} worker-hours; ETA with {args.workers} "
          f"workers {remaining_s / args.workers / 3600:.1f} h (s/episode observed where available, else the queue "
          f"estimate; whole-job granularity not modelled)")
    pid_file = os.path.join(args.out, "sweep_master.pid")
    if os.path.exists(pid_file):
        pid = int(open(pid_file).read().strip())
        print(f"master pid {pid}: {'running' if os.path.exists(f'/proc/{pid}') else 'not running'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_out = os.path.join(REPO, "results", "fig4_rerun")
    default_ckpt_dir = os.path.join(REPO, "checkpoints")
    p = sub.add_parser("plan")
    p.add_argument("--out", default=default_out)
    p.add_argument("--planners", nargs="+", default=PLANNERS, choices=PLANNERS)
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS), choices=list(CKPTS))
    p.add_argument("--pads", nargs="+", type=float, default=PADS)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--horizon", type=int, default=HORIZON)
    w = sub.add_parser("work")
    w.add_argument("--out", default=default_out)
    w.add_argument("--workers", type=int, default=4)
    w.add_argument("--threads", type=int, default=1, help="OMP/MKL/OpenBLAS threads per job process")
    w.add_argument("--gpu-budget-mib", type=int, default=7900,
                   help="cap on this sweep's GPU memory: steady + building-peak estimates of the queue cost table")
    w.add_argument("--gpu-reserve-mib", type=int, default=2500,
                   help="launch only if the GPU keeps this much free memory after the new job (shared GPU)")
    w.add_argument("--max-attempts", type=int, default=5, help="consecutive attempts without a new finished seed")
    w.add_argument("--retry-delay-s", type=float, default=120.0)
    w.add_argument("--poll-s", type=float, default=10.0)
    w.add_argument("--ckpt-dir", default=default_ckpt_dir)
    j = sub.add_parser("job")
    j.add_argument("--out", default=default_out)
    j.add_argument("--planner", required=True, choices=PLANNERS)
    j.add_argument("--ckpt", required=True, choices=list(CKPTS))
    j.add_argument("--pad", type=float, required=True)
    j.add_argument("--n-seeds", type=int, default=N_SEEDS)
    j.add_argument("--horizon", type=int, default=HORIZON)
    j.add_argument("--ckpt-dir", default=default_ckpt_dir)
    j.add_argument("--config", default=None)
    j.add_argument("--verbose", action="store_true", help="keep the planners' console output in the job log")
    s = sub.add_parser("status")
    s.add_argument("--out", default=default_out)
    s.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if getattr(args, "out", None):
        args.out = os.path.abspath(args.out)
    {"plan": cmd_plan, "work": cmd_work, "job": cmd_job, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
