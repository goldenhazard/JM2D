"""Experiment configs (``configs/*.json``).

Schema::

    {"task": "donut_joint" | "donut_constrained", "checkpoint": "<path relative to the repository root>", "n": N,
     "use_constraints": true,
     "base": {<BaseSampler / BaseJointSampler kwargs>},
     "samplers": {"<name>": {<sampler kwargs>}, ...}}

Keys were renamed from the original upstream config schema (old -> new).
"""
import json
import os

from jm2d_toy.tasks import canonical

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
DEFAULT_CONFIGS = {"donut_joint": "donut_joint.json", "donut_constrained": "donut_constrained.json"}


def load_config(task_or_path):
    """Load ``configs/<task>.json`` (``"donut_joint"`` / ``"donut_constrained"``) or an explicit JSON path."""
    path = task_or_path
    if canonical(task_or_path) in DEFAULT_CONFIGS:
        path = os.path.join(CONFIG_DIR, DEFAULT_CONFIGS[canonical(task_or_path)])
    with open(path) as f:
        cfg = json.load(f)
    if not os.path.isabs(cfg["checkpoint"]):
        cfg["checkpoint"] = os.path.join(REPO_ROOT, cfg["checkpoint"])
    return cfg


def sampler_kwargs(cfg, name):
    """``base`` kwargs merged with the ``samplers[name]`` section."""
    if name not in cfg["samplers"]:
        raise KeyError(f"sampler {name!r} not in config (available: {sorted(cfg['samplers'])})")
    kwargs = dict(cfg.get("base", {}))
    kwargs.update(cfg["samplers"][name])
    return kwargs
