"""Planner registry.

    REGISTRY = {"dp": DPPlanner, "rail": RAILPlanner, "gibbs": GibbsPlanner, "jm2d": JM2DPlanner}

    planner = make("jm2d", rollout_policy=dp, backup_policy=backup, predictor=predictor, dt_action=0.01,
                   **config.safety)                        # low level: wrap an existing DP
    planner, env, eval_env = build_planner("jm2d", ckpt="checkpoints/maze2d_large_ep1900.pth", pad=0.15,
                                           backup="hq")    # high level: checkpoint + config + padded maze
"""
from jm2d_maze.planners.dp import DPPlanner
from jm2d_maze.planners.gibbs import GibbsPlanner
from jm2d_maze.planners.jm2d import JM2DPlanner
from jm2d_maze.planners.rail import RAILPlanner

REGISTRY = {"dp": DPPlanner, "rail": RAILPlanner, "gibbs": GibbsPlanner, "jm2d": JM2DPlanner}


def make(name, rollout_policy, backup_policy, predictor, dt_action, **safety_config):
    """Wrap ``rollout_policy`` (a robomimic DP ``RolloutPolicy``) with planner ``name``."""
    from safediffusion.algo.jm2d_policy import GUIDANCE_MODES
    from jm2d_maze.config import PLANNER_ALIGN_MODE

    if name not in REGISTRY:
        raise KeyError(f"unknown planner {name!r}; choose from {sorted(REGISTRY)}")
    if name in PLANNER_ALIGN_MODE:
        mode = GUIDANCE_MODES[safety_config["align"]["mode"]]
        if mode != PLANNER_ALIGN_MODE[name]:
            raise ValueError(f"planner {name!r} needs safety.align.mode={PLANNER_ALIGN_MODE[name]!r}, got {mode!r} "
                             "(use jm2d_maze.config.load_config(planner=...))")
    return REGISTRY[name](rollout_policy=rollout_policy, backup_policy=backup_policy, predictor=predictor,
                          dt_action=dt_action, **safety_config)


def build_planner(name, ckpt, pad=0.0, backup="hq", config=None, num_workers=None, seed_gym_spaces=True,
                  maze_layout=None):
    """Checkpoint + config -> (planner, env, eval_env), with the legacy build order (see ``jm2d_maze.build``).

    Call ``jm2d_maze.seeding.set_random_seed(42)`` first to reproduce the paper protocol (``scripts/run.py`` does).
    ``seed_gym_spaces=False`` keeps the research code's unseeded construction-time action (non-reproducible).
    ``pad`` is any wall padding w >= 0; ``maze_layout`` replaces the maze2d-large layout
    (``jm2d_maze.tasks.get_maze_spec``).
    """
    from safediffusion.algo.jm2d_policy import GUIDANCE_MODES
    from jm2d_maze.build import build
    from jm2d_maze.config import PLANNER_ALIGN_MODE, load_config

    if name not in REGISTRY:
        raise KeyError(f"unknown planner {name!r}; choose from {sorted(REGISTRY)}")
    if config is None or isinstance(config, str):
        config = load_config(config, planner=name, num_workers=num_workers)
    if name in PLANNER_ALIGN_MODE and GUIDANCE_MODES[config.safety.align.mode] != PLANNER_ALIGN_MODE[name]:
        raise ValueError(f"planner {name!r} needs safety.align.mode={PLANNER_ALIGN_MODE[name]!r}")
    return build(REGISTRY[name], ckpt, config, pad=pad, backup=backup, seed_gym_spaces_=seed_gym_spaces,
                 maze_layout=maze_layout)


__all__ = ["REGISTRY", "make", "build_planner", "DPPlanner", "RAILPlanner", "GibbsPlanner",
           "JM2DPlanner"]
