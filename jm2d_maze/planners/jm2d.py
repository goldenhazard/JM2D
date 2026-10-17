"""JM2D on PointMaze: joint sampling of the DP actions and the backup parameter."""
from safediffusion.algo.jm2d_policy import JM2DPolicy
from jm2d_maze.planners.maze_filter import SafetyFilterMaze


class JM2DPlanner(SafetyFilterMaze, JM2DPolicy):
    """Was ``SafeDiffusionPolicyMaze`` in the research code.

    The guidance mode is read from ``safety.align.mode``; ``jm2d_maze.planners.make`` sets it to ``jm2d``.
    """
    pass
