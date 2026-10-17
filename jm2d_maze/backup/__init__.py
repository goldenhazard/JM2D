"""Backup planners: ``make_backup(name, **safety_config)``.

    hq    Simple2DPlanner                    k in [-1, 1]^2   (paper "HQ"; legacy backup_policy_type simple2d)
    xpyp  Simple2DPlannerUnderAct            k in [ 0, 1]^2   (Table 1 "x+y+"; legacy simple2d_xpyp)
    xmym  Simple2DPlannerUnderAct            k in [-1, 0]^2   (Table 1 "x-y-"; legacy simple2d_xmym)
"""
from jm2d_maze.backup.simple2d import Simple2DPlanner, Simple2DPlannerUnderAct

BACKUPS = {
    "hq": (Simple2DPlanner, {}),
    "xpyp": (Simple2DPlannerUnderAct, {"k_lb": [0, 0], "k_ub": [1, 1]}),
    "xmym": (Simple2DPlannerUnderAct, {"k_lb": [-1, -1], "k_ub": [0, 0]}),
}
LEGACY_NAMES = {"hq": "simple2d", "xpyp": "simple2d_xpyp", "xmym": "simple2d_xmym"}


def make_backup(name, **safety_config):
    """Instantiate backup planner ``name`` (``hq``, ``xpyp``, ``xmym``) from the ``safety`` config block."""
    if name not in BACKUPS:
        raise KeyError(f"unknown backup planner {name!r}; choose from {sorted(BACKUPS)}")
    cls, kwargs = BACKUPS[name]
    return cls(**safety_config, **kwargs)


__all__ = ["BACKUPS", "LEGACY_NAMES", "make_backup", "Simple2DPlanner", "Simple2DPlannerUnderAct"]
