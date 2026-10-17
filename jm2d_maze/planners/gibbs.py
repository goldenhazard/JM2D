"""Gibbs baseline on PointMaze (Fig. 4): alternating action / backup-parameter updates."""
from jm2d_maze.planners.jm2d import JM2DPlanner


class GibbsPlanner(JM2DPlanner):
    """Was ``SafeDiffusionPolicyMaze`` with ``align.mode=gibbs_sampling`` (SafeRobomimic@paper/gibbs, be46370b).

    Same class as JM2D; ``jm2d_maze.planners.make`` sets ``safety.align.mode = "gibbs"``, which dispatches every
    guided denoising step to ``JM2DPolicy.multistep_gibbs_sampling``.
    """
    pass
