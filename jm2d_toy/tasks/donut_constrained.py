"""Constrained generation on the donut (task ``donut_constrained``, paper Fig. 9): x in R^2 must avoid a
wall-shaped obstacle.

Moved from legacy ``dev/problems/donut2D_problem.py`` (``SCENARIO_DICT`` L14, ``Donut2DProblem`` L86).
Objective and constraint are unchanged. Changes: the constructor no longer builds a 10k dataset (it only
read the radii, and the data-manifold constraints of the removed ``classic_informed`` projection mode);
``is_ood`` uses the dataset radii as constants instead of regenerating a dataset per call. Plotting moved to
``jm2d_toy.plotting``. Removed: scenarios other than ``great_wall_rectangle`` and ``Donut2DJointProblem``.
"""
import numpy as np
import torch

from jm2d_toy.constraints import CircleConstraint, SupEllipseConstraint
from jm2d_toy.data import R_INNER, R_OUTER
from jm2d_toy.tasks.problem import TFGProblem

SCENARIOS = {
    # Fig. 9: a super-ellipse (p = q = 4) wall covering the left part of the donut; no objective (center None)
    "great_wall_rectangle": {
        "center": None,
        "obs_types": ['rectangle'],
        "obs_centers": np.array([[-0.20, 0.00]]),
        "obs_radius": np.array([[1.00, 5.00]])
    }
}


class DonutConstrainedProblem(TFGProblem):
    """Objective: distance to ``center`` (zero if ``center`` is None). Constraint: sum of obstacle violations."""
    def __init__(self,
                center=np.array([0, 0.1]),
                obs_types = ['circle', 'circle'],
                obs_centers=np.array([[0.00, 0.00],[-0.5, 0.5]]),
                obs_radius=np.array([0.9, 0.2]),
                device=torch.device('cuda')):
        self.device = device
        if center is not None:
            self.center = torch.tensor(center, dtype=torch.float32, device=self.device)
        else:
            self.center = None

        # define obstacle of shape circle with center at obs_center and raidus length obs_radius
        assert len(obs_centers) == len(obs_radius)

        self.obstacles = []
        for obs_center, obs_radius, obs_type in zip(obs_centers, obs_radius, obs_types):
            if obs_type == 'circle':
                self.obstacles.append(CircleConstraint(obs_center, obs_radius, device=self.device))
            elif obs_type == 'rectangle':
                self.obstacles.append(SupEllipseConstraint(obs_center, obs_radius, p=4, q=4, device=self.device))
            else:
                raise ValueError(f"Unknown obstacle type: {obs_type}")

    @classmethod
    def from_scenario(cls, scenario="great_wall_rectangle", device=torch.device('cuda')):
        return cls(**SCENARIOS[scenario], device=device)

    @property
    def input_dim(self):
        return 2

    def objective_function(self, x):
        """
        Objective function for the donut problem

        Args:
            x (torch.Tensor): (B, 2) The input to the objective function

        Returns:
            torch.Tensor: (B,) The value of the objective function
        """
        x1 = x[...,:2]

        if self.center is not None:
            # Minimize distance to center
            obj = torch.linalg.norm(x1 - self.center, axis=-1)
        else:
            # No objective alignment
            obj = torch.zeros_like(x[..., 0])

        return obj

    def constraint_function(self, x):
        """
        Constraint function for the donut problem, positive for infeasible samples

        Args:
            x (torch.Tensor): (B1, B2, ..., Bn, 2) The input to the constraint function

        Returns:
            torch.Tensor: (B1, B2, ..., Bn) summed constraint violation
        """
        constraints = torch.zeros(x.shape[:-1] + (len(self.obstacles),), device=x.device)

        for i, constraint in enumerate(self.obstacles):
            cons = constraint.evaluate(x) # positive for feasible region
            constraints[..., i] = -cons.squeeze(-1)

        constraints = torch.relu(constraints)
        constraints = torch.sum(constraints, dim=-1)

        return constraints

    def global_optimal_value(self, num_samples=10000, seed=0):
        """Minimum objective over a seeded donut dataset (legacy: unseeded ``get_dataloader()``)."""
        from jm2d_toy.data import DonutConstrainedDataset
        dataset = DonutConstrainedDataset(num_samples, rng=np.random.RandomState(seed))
        global_optima = self.objective_function(self.to_tensor(dataset.all_samples)).min()
        return global_optima.cpu().numpy()

    def is_ood(self, x):
        r0 = R_INNER
        r1 = R_OUTER

        point1_ood = torch.logical_or(torch.linalg.norm(x[...,:2], axis=-1) < r0, torch.linalg.norm(x[...,:2], axis=-1) > r1)

        return point1_ood


Donut2DProblem = DonutConstrainedProblem
