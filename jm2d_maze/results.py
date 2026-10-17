"""Per-episode result schema (shared by ``scripts/run.py``, ``scripts/compare.py`` and the stored paper runs).

One run = one planner x task x checkpoint x env seeds = ``<out>/<task>/<planner>/<ckpt>.jsonl``, one JSON line
per episode::

    {"schema": "jm2d_maze/episode/v1", "task": "maze_pad015_hq", "planner": "jm2d", "pad": 0.15, "backup": "hq",
     "ckpt": "ep1900", "env_seed": 3, "success": true, "safe": true, "success_and_safe": true, "n_unsafe": 0,
     "n_intervention": 3, "n_plan": 19, "intervention_rate": 0.158, "horizon_steps": 290, "horizon_s": 2.9,
     "stuck": false, "time_per_replan_s": 4.1, "source": "jm2d_maze", ...}

``legacy_stats_to_records`` converts a stored research-code ``stats.json`` (per-seed dicts keyed by the seed) to
the same records, so stored and re-run results go through the same aggregation (``jm2d_maze.compare``).
"""
import glob
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

SCHEMA = "jm2d_maze/episode/v1"
DT_CONTROL = 0.01  # s per control step (D4RL maze timestep 0.01, frame_skip 1)


@dataclass
class EpisodeResult:
    task: str
    planner: str
    pad: float
    backup: str
    ckpt: str
    env_seed: int
    success: bool
    n_unsafe: int
    n_intervention: int
    n_plan: int
    horizon_steps: int
    stuck: bool = False
    n_unsafe_head: Optional[int] = None
    n_unsafe_backup: Optional[int] = None
    time_per_replan_s: Optional[float] = None
    ckpt_md5: str = ""
    source: str = "jm2d_maze"
    run: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def safe(self):
        return self.n_unsafe == 0

    def to_dict(self):
        d = asdict(self)
        d.update(schema=SCHEMA, safe=self.safe, success_and_safe=bool(self.success and self.safe),
                 intervention_rate=(self.n_intervention / self.n_plan) if self.n_plan else 0.0,
                 horizon_s=round(self.horizon_steps * DT_CONTROL, 4))
        order = ["schema", "task", "planner", "pad", "backup", "ckpt", "env_seed", "success", "safe",
                 "success_and_safe", "n_unsafe", "n_intervention", "n_plan", "intervention_rate", "horizon_steps",
                 "horizon_s", "stuck", "n_unsafe_head", "n_unsafe_backup", "time_per_replan_s", "ckpt_md5",
                 "source", "run", "extra"]
        return {k: d[k] for k in order}

    @classmethod
    def from_legacy_stat(cls, stat, **meta):
        """Episode record from one legacy per-seed stats dict (``rollout`` output)."""
        detail = stat.get("SafetyDetail", {}) or {}
        t = stat.get("Time")
        return cls(success=bool(stat["Success"]), n_unsafe=int(stat["Unsafe"]),
                   n_intervention=int(stat["Intervention"]), n_plan=int(stat["Plan"]),
                   horizon_steps=int(stat["Horizon"]), stuck=bool(stat.get("Stuck", False)),
                   n_unsafe_head=detail.get("HeadPlan"), n_unsafe_backup=detail.get("BackupPlan"),
                   time_per_replan_s=None if t is None or (isinstance(t, float) and np.isnan(t)) else float(t),
                   **meta)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__}


def run_path(out_dir, task, planner, ckpt):
    return os.path.join(out_dir, task, planner, f"{ckpt}.jsonl")


class JSONLWriter:
    """Append episode records to ``<out>/<task>/<planner>/<ckpt>.jsonl`` (one line per episode)."""

    def __init__(self, path, overwrite=True):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if overwrite and os.path.exists(path):
            os.remove(path)

    def write(self, record):
        row = record.to_dict() if isinstance(record, EpisodeResult) else record
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(results_dir):
    """All episode records under ``results_dir`` (``**/*.jsonl``)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.jsonl"), recursive=True)):
        for row in load_jsonl(path):
            if row.get("schema") == SCHEMA:
                row.setdefault("run", os.path.relpath(path, results_dir))
                rows.append(row)
    return rows


def legacy_stats_to_records(stats, **meta):
    """Stored research-code ``stats.json`` -> list of episode-record dicts (seeds in numeric order)."""
    seeds = sorted((k for k in stats if k.isdigit()), key=int)
    return [EpisodeResult.from_legacy_stat(stats[k], env_seed=int(k), **meta).to_dict() for k in seeds]
