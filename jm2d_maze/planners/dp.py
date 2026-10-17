"""DP baseline on PointMaze: the diffusion policy without a safety filter."""
from safediffusion.algo.safety_filter import DPPolicy
from jm2d_maze.planners.maze_filter import SafetyFilterMaze


class DPPlanner(SafetyFilterMaze, DPPolicy):
    """Was ``IdentityDiffusionPolicyMaze`` (legacy ``policy_mode=diffusion``)."""
    pass
