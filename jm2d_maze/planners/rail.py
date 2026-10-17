"""RAIL baseline on PointMaze: DP plan + head-plan verification + IPOPT backup trajectory optimisation."""
from safediffusion.algo.safety_filter import RAILPolicy
from jm2d_maze.planners.maze_filter import SafetyFilterMaze


class RAILPlanner(SafetyFilterMaze, RAILPolicy):
    """Was ``ExplicitSafetyFilterMaze`` (legacy ``policy_mode=rail``)."""
    pass
