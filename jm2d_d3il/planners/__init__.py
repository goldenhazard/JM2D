"""Planner registry: ``make(name, **settings)`` / ``from_config(name, **overrides)``.

    dpcc          DPCC (projection), legacy 'dpcc-c-tightened'                       origin/um_dpcc
    safediffuser  SafeDiffuser, variant res | tvs, legacy 'invariance-{ReS,TVS}-dpcc-c'  origin/mpd-improved
    mpd           MPD (noisy-space cost guidance), legacy 'mpd-gradient'              origin/mpd-improved
    jm2d          JM2D ("Ours"), legacy 'model_based-dpcc-c-tightened', u_steps       origin/um_dpcc
"""
import os

import yaml

from ..paths import PLANNER_CONFIG_DIR
from .base import Planner
from .dpcc import DPCCPlanner
from .jm2d import JM2DPlanner, resolve_u_steps
from .mpd import MPDPlanner
from .safediffuser import SafeDiffuserPlanner

REGISTRY = {"dpcc": DPCCPlanner, "safediffuser": SafeDiffuserPlanner, "mpd": MPDPlanner, "jm2d": JM2DPlanner}

# result labels -> (planner, overrides); the labels used in results/ and in the tables
LABELS = {
    "dpcc": ("dpcc", {}),
    "safediffuser-res": ("safediffuser", {"variant": "res"}),
    "safediffuser-tvs": ("safediffuser", {"variant": "tvs"}),
    "mpd": ("mpd", {}),
    "jm2d": ("jm2d", {"u_steps": "full"}),
}


def load_planner_config(name, path=None):
    with open(path or os.path.join(PLANNER_CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def make(name, **settings):
    if name not in REGISTRY:
        raise KeyError(f"unknown planner {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**settings)


def from_config(name, config_path=None, **overrides):
    """Planner with the settings of ``configs/planners/<name>.yaml`` (``overrides`` win; None values are ignored)."""
    cfg = load_planner_config(name, config_path)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return make(name, **cfg)


def from_label(label, **overrides):
    """'safediffuser-tvs', 'jm2d', 'jm2d-u5', ... -> planner."""
    if label.startswith("jm2d-u"):
        return from_config("jm2d", u_steps=int(label[len("jm2d-u"):]), **overrides)
    name, extra = LABELS[label]
    return from_config(name, **{**extra, **overrides})


__all__ = ["Planner", "REGISTRY", "LABELS", "make", "from_config", "from_label", "load_planner_config",
           "DPCCPlanner", "SafeDiffuserPlanner", "MPDPlanner", "JM2DPlanner", "resolve_u_steps"]
