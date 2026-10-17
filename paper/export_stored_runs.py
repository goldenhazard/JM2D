#!/usr/bin/env python
"""Build ``results/paper/`` from the research-code result folders (run once by the authors; kept for provenance).

    python paper/export_stored_runs.py --src <SafeRobomimic>/safediffusion_d4rl/pointmaze/results/large

Copies, for every run behind Fig. 4 and Table 1, only ``stats.json`` (aggregate + one entry per env seed) and
``config.json`` (absolute paths redacted); runs without an aggregate ``stats.json`` (the Table 1 RAIL runs) keep
their ``seed<k>/stats.json``. Videos and ``rollout_results.pkl`` are not copied. ``results/paper/index.json``
lists every run with its figure, planner, padding, backup and (inferred) checkpoint. Run folders of JM2D are renamed
from the research-code prefix ``rail_aligned`` to ``jm2d`` (``path``); ``legacy_path`` keeps the original name.

Run selection = the committed plotting/stat scripts of the research code:
* Fig. 4: ``generate_paper_plot.py`` globs ``large_pad<www>/*<pattern>*`` with patterns ``rail_aligned_2025`` (JM2D),
  ``rail_2025`` (RAIL), ``gibbs`` (Gibbs), ``diffusion`` (DP) and keeps folders that have a ``stats.json``.
* Table 1: ``print_stats.py`` folders ``large_pad020/{rail,rail_aligned}_simple2d{,_xpyp,_xmym}_2025_04_28_*``.
"""
import argparse
import glob
import json
import os
import re
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PADS = {"large_pad000": 0.0, "large_pad005": 0.05, "large_pad010": 0.10, "large_pad015": 0.15,
        "large_pad020": 0.20, "large_pad025": 0.25}
FIG4_PATTERNS = [("rail_aligned_2025", "jm2d"), ("rail_2025", "rail"), ("gibbs", "gibbs"), ("diffusion", "dp")]
TABLE1_RUNS = {
    ("rail", "hq"): "rail_simple2d_2025_04_28_15_02_iter0",
    ("rail", "xpyp"): "rail_simple2d_xpyp_2025_04_28_15_55_iter0",
    ("rail", "xmym"): "rail_simple2d_xmym_2025_04_28_16_46_iter0",
    ("jm2d", "hq"): "rail_aligned_simple2d_2025_04_28_15_02_iter0",
    ("jm2d", "xpyp"): "rail_aligned_simple2d_xpyp_2025_04_28_16_38_iter0",
    ("jm2d", "xmym"): "rail_aligned_simple2d_xmym_2025_04_28_18_29_iter0",
}


def infer_ckpt(planner, run, pad_dir, src):
    """Checkpoint of a stored run (not recorded in the run folders; inferred from names and timestamps).

    * ``gibbs_<epoch>``: epoch from the name (paper/gibbs run scripts loop over 1250..2000).
    * dated runs of 2025-04-21..24: ``experiments/run_{dp,rail,rail_v2}.sh`` (ad701a1c) loop over epochs
      1250/1500/1750/2000 of maze2d_large_DP/20250416144223 in this order -> rank of the timestamp.
    * runs of 2025-04-12/13 and the undated ``*_iter0`` runs: maze2d_large/model_epoch_1900.pth (run_rail.sh at
      28e0a36d / a7b7bb47).
    * Table 1 runs (2025-04-28): epoch 2000 (run_rail{,_v2}_biased_backup.sh).
    """
    m = re.match(r"gibbs_(\d+)$", run)
    if m:
        return f"ep{m.group(1)}", "folder name"
    if "simple2d" in run:
        return "ep2000", "run_rail_v2_biased_backup.sh (inferred)"
    m = re.search(r"_2025_(\d\d)_(\d\d)_(\d\d)_(\d\d)_iter", run)
    if m is None:
        return "ep1900", "undated *_iter0 run (inferred)"
    month, day = int(m.group(1)), int(m.group(2))
    if (month, day) <= (4, 13):
        return "ep1900", "2025-04-12/13 run (inferred)"
    prefix = run[:m.start()]
    dated = sorted(os.path.basename(d) for d in glob.glob(os.path.join(src, pad_dir, prefix + "_2025_04_2*_iter0"))
                   if os.path.exists(os.path.join(d, "stats.json")) and re.search(r"_2025_04_2[1-4]_", d))
    epochs = [1250, 1500, 1750, 2000]
    if len(dated) == 4 and run in dated:
        return f"ep{epochs[dated.index(run)]}", "timestamp rank among the 04-21..24 runs (inferred)"
    return "unknown", "not inferable"


def release_name(run):
    """Folder name in results/paper: the research-code JM2D prefix ``rail_aligned`` becomes ``jm2d``."""
    return "jm2d" + run[len("rail_aligned"):] if run.startswith("rail_aligned") else run


def redact(text, src):
    root = src.split("/safediffusion_d4rl/")[0]
    return text.replace(root, "<SafeRobomimic>")


def copy_run(src, pad_dir, run, dst_root):
    src_dir = os.path.join(src, pad_dir, run)
    dst_dir = os.path.join(dst_root, pad_dir, release_name(run))
    os.makedirs(dst_dir, exist_ok=True)
    files = []
    for name in ("stats.json", "config.json"):
        p = os.path.join(src_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                txt = redact(f.read(), src)
            with open(os.path.join(dst_dir, name), "w") as f:
                f.write(txt)
            files.append(name)
    if "stats.json" not in files:
        for seed_dir in sorted(glob.glob(os.path.join(src_dir, "seed*"))):
            p = os.path.join(seed_dir, "stats.json")
            if os.path.exists(p):
                os.makedirs(os.path.join(dst_dir, os.path.basename(seed_dir)), exist_ok=True)
                shutil.copy(p, os.path.join(dst_dir, os.path.basename(seed_dir), "stats.json"))
                files.append(os.path.join(os.path.basename(seed_dir), "stats.json"))
    seeds = [x for x in files if x.startswith("seed")]
    return [x for x in files if not x.startswith("seed")] + ([f"seed<k>/stats.json ({len(seeds)} seeds)"] if seeds else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="<SafeRobomimic>/safediffusion_d4rl/pointmaze/results/large")
    parser.add_argument("--dst", default=os.path.join(REPO_ROOT, "results", "paper"))
    args = parser.parse_args()
    src = os.path.abspath(args.src)

    index = []
    for pad_dir, pad in PADS.items():
        for pattern, planner in FIG4_PATTERNS:
            runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(src, pad_dir, f"*{pattern}*"))
                          if os.path.exists(os.path.join(d, "stats.json")))
            for run in runs:
                files = copy_run(src, pad_dir, run, args.dst)
                ckpt, how = infer_ckpt(planner, run, pad_dir, src)
                index.append(dict(figure="fig4", planner=planner, pad=pad, backup="hq", ckpt=ckpt, ckpt_source=how,
                                  path=f"{pad_dir}/{release_name(run)}", legacy_path=f"{pad_dir}/{run}", files=files))
    for (planner, backup), run in TABLE1_RUNS.items():
        files = copy_run(src, "large_pad020", run, args.dst)
        ckpt, how = infer_ckpt(planner, run, "large_pad020", src)
        index.append(dict(figure="table1", planner=planner, pad=0.20, backup=backup, ckpt=ckpt, ckpt_source=how,
                          path=f"large_pad020/{release_name(run)}", legacy_path=f"large_pad020/{run}", files=files))
    with open(os.path.join(args.dst, "index.json"), "w") as f:
        json.dump({"source": "SafeRobomimic/safediffusion_d4rl/pointmaze/results/large (research-code runs)",
                   "renamed": "JM2D run folders: research-code prefix 'rail_aligned' -> 'jm2d' (legacy_path = original)",
                   "runs": index}, f, indent=1)
    print(f"{len(index)} runs -> {args.dst}")


if __name__ == "__main__":
    main()
