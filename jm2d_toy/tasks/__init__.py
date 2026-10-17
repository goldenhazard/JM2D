"""Problems for the two toy tasks.

    tasks.make("donut_joint")        -> DonutJointProblem with the four circular obstacles (paper Fig. 2)
    tasks.make("donut_constrained")  -> DonutConstrainedProblem, scenario ``great_wall_rectangle`` (paper Fig. 9)
"""
import sys

import torch

from jm2d_toy.tasks import donut_constrained
from jm2d_toy.tasks.donut_constrained import SCENARIOS, Donut2DProblem, DonutConstrainedProblem
from jm2d_toy.tasks.donut_joint import FIG2_OBS_CENTERS, FIG2_OBS_RADIUS, DonutJointProblem
from jm2d_toy.tasks.problem import TFGJointProblem, TFGProblem

TASKS = ("donut_joint", "donut_constrained")
JOINT_TASKS = ("donut_joint",)
# task -> figure id in the paper (the keys of paper/targets.json)
PAPER_FIGURE = {"donut_joint": "fig2", "donut_constrained": "fig9"}
_ALIASES = {figure: task for task, figure in PAPER_FIGURE.items()}

sys.modules[__name__ + ".donut2d"] = donut_constrained


def canonical(task):
    """Task name for ``task`` (the paper figure ids are accepted too)."""
    return _ALIASES.get(task, task)


def make(task, device=torch.device('cuda')):
    """Build the problem of a task."""
    task = canonical(task)
    if task == "donut_joint":
        problem = DonutJointProblem(obs_centers=FIG2_OBS_CENTERS, obs_radius=FIG2_OBS_RADIUS, device=device)
    elif task == "donut_constrained":
        problem = DonutConstrainedProblem.from_scenario("great_wall_rectangle", device=device)
    else:
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
    problem.task = task
    return problem


__all__ = ["make", "canonical", "TASKS", "JOINT_TASKS", "PAPER_FIGURE", "DonutJointProblem",
           "DonutConstrainedProblem", "SCENARIOS", "TFGProblem", "TFGJointProblem"]
