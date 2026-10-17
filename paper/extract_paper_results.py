#!/usr/bin/env python
"""Convert the stored legacy runs behind paper Fig. 2 and Fig. 9(b) into ``results/paper/`` (SampleResult schema).

    python paper/extract_paper_results.py --legacy-results <SafeRobomimic>/dev/results

Sources (legacy ``histories.pkl`` + ``important_stats.json``):
    donut_joint/sequential        donut_joint/sequential/20250427_010251   (SequentialGDPExact)
    donut_joint/gibbs             donut_joint/gibbs_exact/20250427_011204  (GibbsAlternatingGDPExact)
    donut_joint/jm2d              donut_joint/yongxin/20250427_005601      (YongxinExactGDP)
    donut_constrained/projection  donut2d/projection/20250409_171730       (ProjectionDP, classic)

Joint runs: final x = noisy_samples[-1], k = clean_params[-1] (recorded after each step).
donut_constrained projection: histories are recorded *before* each step, so the final sample is rebuilt by the last
step: at t = 0 the DDPM step is deterministic and equals the stored Tweedie estimate clean_samples[:, -1]
(coefficients 1 and 0, no noise), followed by the projection, which is re-applied here.
The recomputed counts are asserted against the stored ``important_stats.json``.
"""
import argparse
import json
import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jm2d_toy import samplers, tasks  # noqa: E402
from jm2d_toy.config import load_config  # noqa: E402
from jm2d_toy.results import SampleResult, md5, save_result  # noqa: E402

RUNS = [
    ("donut_joint", "sequential", "donut_joint/sequential/20250427_010251", "SequentialGDPExact", "sequential"),
    ("donut_joint", "gibbs", "donut_joint/gibbs_exact/20250427_011204", "GibbsAlternatingGDPExact", "gibbs_exact"),
    ("donut_joint", "jm2d", "donut_joint/yongxin/20250427_005601", "YongxinExactGDP", "yongxin"),
    ("donut_constrained", "projection", "donut2d/projection/20250409_171730", "ProjectionDP", "projection"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-results", required=True, help="<SafeRobomimic>/dev/results")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper"))
    args = parser.parse_args()

    for task, sampler, rel, legacy_cls, section in RUNS:
        run_dir = os.path.join(args.legacy_results, rel)
        with open(os.path.join(run_dir, "histories.pkl"), "rb") as f:
            hist = pickle.load(f)
        with open(os.path.join(run_dir, "important_stats.json")) as f:
            stats = json.load(f)
        with open(os.path.join(run_dir, "guide_config.json")) as f:
            guide_config = json.load(f)
        with open(os.path.join(run_dir, "args.json")) as f:
            run_args = json.load(f)
        n = run_args["N_samples_eval"]
        problem = tasks.make(task)
        cfg = load_config(task)

        if task == "donut_joint":
            x = np.asarray(hist["noisy_samples"][-1])
            k = np.asarray(hist["clean_params"][-1])
            xt, kt = problem.to_tensor(x), problem.to_tensor(k)
            valid = problem.is_valid(xt, kt).cpu().numpy()
            objective = problem.objective_function(xt, kt).cpu().numpy()
        else:
            k = None
            x0_last = torch.tensor(np.asarray(hist["clean_samples"])[:, -1], device="cuda")
            proj = samplers.from_config(cfg, "projection").build(problem)
            x = proj.project(x0_last).cpu().numpy()
            xt = problem.to_tensor(x)
            valid = problem.is_valid(xt).cpu().numpy()
            objective = problem.objective_function(xt).cpu().numpy()
        ood = problem.is_ood(xt).cpu().numpy()

        # recomputed metrics must match the stored legacy stats
        assert int(valid.sum()) == round(stats["num_valid"] * n), (rel, int(valid.sum()), stats["num_valid"])
        assert int(ood.sum()) == round(stats["num_ood"] * n), (rel, int(ood.sum()), stats["num_ood"])
        assert np.isclose(objective.mean(), stats["objective_mean"], atol=1e-5), (rel, objective.mean())

        legacy_section = guide_config["gibbs_exact"] if section == "sequential" else guide_config[section]
        res = SampleResult(
            task=task, sampler=sampler, n=n, seed=None, x=x, k=k, valid=valid, ood=ood, objective=objective,
            config={"class": legacy_cls, "legacy_section": section, "legacy_base": guide_config["base"],
                    "legacy_params": {kk: vv for kk, vv in legacy_section.items() if kk != "enabled"}},
            use_constraints=bool(run_args["use_constraints"]), global_optimum=float(stats["global_optima"]),
            ckpt=os.path.basename(cfg["checkpoint"]), ckpt_md5=md5(cfg["checkpoint"]), time_s=0.0,
            source=f"stored legacy run SafeRobomimic/dev/results/{rel} (unseeded; code {legacy_cls})",
        )
        path = save_result(res, args.out)
        print(f"{task}/{sampler}: valid {int(valid.sum())}/{n}, ood {int(ood.sum())}/{n} -> {path}")


if __name__ == "__main__":
    main()
