"""Result schema shared by ``scripts/run.py``, ``scripts/compare.py`` and the notebook.

One run = one ``SampleResult`` = ``<out>/<task>/<sampler>/<run_id>.json`` (summary, schema below) plus
``<run_id>.npz`` (x, k, per-sample valid/ood/objective; history arrays when requested).

JSON fields: schema, task, sampler, run_id, seed, n, n_valid, n_ood, n_compatible (joint tasks: samples
whose (x, k) satisfy the constraint, i.e. n_valid), objective_{mean,min,max}, global_optimum,
use_constraints, config, ckpt, ckpt_md5, time_s, versions, source.
"""
import glob
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jm2d_toy.tasks import canonical

SCHEMA = "jm2d_toy/sample_result/v1"
HISTORY_KEYS_SAVED = ("noisy_samples", "clean_samples", "clean_params")


@dataclass
class SampleResult:
    task: str
    sampler: str
    n: int
    seed: Optional[int]
    x: np.ndarray
    valid: np.ndarray
    ood: np.ndarray
    objective: np.ndarray
    k: Optional[np.ndarray] = None
    history: Optional[dict] = None
    config: dict = field(default_factory=dict)
    use_constraints: bool = True
    global_optimum: Optional[float] = None
    ckpt: str = ""
    ckpt_md5: str = ""
    time_s: float = 0.0
    source: str = "jm2d_toy"

    @property
    def joint(self):
        return self.k is not None

    @property
    def compatible(self):
        """Joint tasks: per-sample flag that (x, k) satisfies the constraint (same as ``valid``)."""
        return self.valid if self.joint else None

    @property
    def run_id(self):
        return "paper" if self.seed is None else f"seed{self.seed}"

    def summary(self):
        return dict(
            schema=SCHEMA, task=self.task, sampler=self.sampler, run_id=self.run_id, seed=self.seed, n=int(self.n),
            n_valid=int(np.sum(self.valid)), n_ood=int(np.sum(self.ood)),
            n_compatible=int(np.sum(self.valid)) if self.joint else None,
            objective_mean=float(np.mean(self.objective)), objective_min=float(np.min(self.objective)),
            objective_max=float(np.max(self.objective)),
            global_optimum=None if self.global_optimum is None else float(self.global_optimum),
            use_constraints=bool(self.use_constraints), config=self.config, ckpt=self.ckpt,
            ckpt_md5=self.ckpt_md5, time_s=round(float(self.time_s), 2), versions=versions(), source=self.source,
        )


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import scipy
    import torch
    import diffusers
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "scipy": scipy.__version__, "diffusers": diffusers.__version__}


def save_result(result: SampleResult, out_dir, save_history=False):
    """Write ``<out_dir>/<task>/<sampler>/<run_id>.{json,npz}``; returns the JSON path."""
    run_dir = os.path.join(out_dir, result.task, result.sampler)
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.join(run_dir, result.run_id)
    arrays = dict(x=result.x, valid=result.valid, ood=result.ood, objective=result.objective)
    if result.k is not None:
        arrays["k"] = result.k
    if save_history and result.history is not None:
        for key in HISTORY_KEYS_SAVED:
            if key in result.history:
                arrays["hist_" + key] = result.history[key]
    np.savez(base + ".npz", **arrays)
    summary = result.summary()
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    return base + ".json"


def load_results(results_dir, task=None):
    """Load every result JSON under ``results_dir`` (optionally one task); adds ``x``/``k`` from the npz."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        with open(path) as f:
            row = json.load(f)
        if row.get("schema") != SCHEMA:
            continue
        row["task"] = canonical(row.get("task"))
        if task is not None and row["task"] != canonical(task):
            continue
        npz = path[:-len(".json")] + ".npz"
        if os.path.exists(npz):
            data = np.load(npz)
            row["x"] = data["x"]
            row["k"] = data["k"] if "k" in data.files else None
            row["valid"] = data["valid"]
            row["ood"] = data["ood"]
        row["path"] = path
        rows.append(row)
    return rows
