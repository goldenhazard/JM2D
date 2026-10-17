"""Joint donut problem (Fig. 2): x = (p1, p2) in R^4 from the donut model, k = (r, theta) in R^2.

Moved from legacy ``dev/problems/donut_problem.py`` (``line_intersects_circle`` L13,
``sample_points_from_circle`` L74, ``DonutJointProblem`` L243). Objective and constraint are unchanged.
Changes: ``is_ood`` uses the dataset radii as constants instead of regenerating a 100k dataset per call,
and ``global_optimal_value`` uses a seeded local generator (both now deterministic, sampler RNG untouched).
Plotting moved to ``jm2d_toy.plotting``. Removed: ``DonutProblem`` (4-D, non-joint, unused by the paper).
"""
import numpy as np
import torch

from jm2d_toy.data import R_INNER, R_OUTER
from jm2d_toy.tasks.problem import TFGJointProblem

# Fig. 2 obstacles (legacy dev/paper_scripts/02_toy_joint_diffusion.py:207-208)
FIG2_OBS_CENTERS = np.array([[-0.50, 0.50], [-0.50, -0.50], [0.50, -0.50], [0.50, 0.50]])
FIG2_OBS_RADIUS = np.array([0.30, 0.50, 0.50, 0.50])


def line_intersects_circle(x1, x2, center, radius, return_penalty=False):
    """
    Check if the line connecting x1 and x2 intersects with the circle with center and radius

    Args:
        x1 (torch.Tensor): (B, 2) The first point of the line
        x2 (torch.Tensor): (B, 2) The second point of the line
        center (torch.Tensor): (2,) The center of the obstacle
        radius (float): The radius of the obstacle
        return_penalty (bool): Whether to return the penalty value

    Returns:
        if return_penalty is True:
            torch.Tensor: (B,) How much penetration of the line into the circle
        else:
            torch.Tensor: (B,) Whether the line intersects with the circle
    """
    radius_sq = radius ** 2

    # Check if either endpoint is inside the circle
    d1_sq = torch.sum((x1 - center) ** 2, dim=1)
    d2_sq = torch.sum((x2 - center) ** 2, dim=1)
    endpoint_inside = (d1_sq <= radius_sq) | (d2_sq <= radius_sq)

    # Compute direction vectors of the line segments
    dx = x2[:, 0] - x1[:, 0]
    dy = x2[:, 1] - x1[:, 1]

    # Vector from x1 to center
    ax = x1[:, 0]
    ay = x1[:, 1]
    cx, cy = center
    numerator = (cx - ax) * dx + (cy - ay) * dy

    # Squared length of the line segment (avoid division by zero)
    denominator = dx ** 2 + dy ** 2

    # Compute t (projection parameter) and clamp to [0, 1]
    t = torch.zeros_like(numerator)
    mask = denominator != 0
    t[mask] = numerator[mask] / denominator[mask]
    t = torch.clamp(t, 0.0, 1.0)

    # Find the closest point on the line segment
    closest_x = ax + t * dx
    closest_y = ay + t * dy

    # Squared distance from closest point to center
    closest_dist_sq = (closest_x - cx) ** 2 + (closest_y - cy) ** 2

    # Check if closest point is within the circle
    closest_intersects = closest_dist_sq <= radius_sq

    if return_penalty:
        # Penalty is max(0, radius^2 - distance^2) for penetration
        penalty = torch.relu(radius_sq - closest_dist_sq)
        return penalty

    # Final result: either endpoint inside or closest point intersects
    return (endpoint_inside | closest_intersects).to(int)


def sample_points_from_circle(r, theta, radius_scale=0.6):
    """
    Map k = (r, theta) in [-1, 1]^2 to a point in the plane: |r| * radius_scale * (cos 2 pi theta, sin 2 pi theta)

    Args:
        r (torch.Tensor): (B,) The radius of the circle
        theta (torch.Tensor): (B,) The angle of the points

    Returns:
        torch.Tensor: (B, 2) The sampled points
    """
    if type(r) == np.ndarray:
        r = torch.tensor(r, dtype=torch.float32)
    if type(theta) == np.ndarray:
        theta = torch.tensor(theta, dtype=torch.float32)

    r = torch.abs(r)*radius_scale
    theta = theta*2*np.pi
    x = r*torch.cos(theta)
    y = r*torch.sin(theta)

    return torch.stack([x, y], dim=-1)


class DonutJointProblem(TFGJointProblem):
    """
    Donut problem with two points x = (p1, p2) and one parameter k (a third point inside r <= 0.6).

    Objective: maximise the perimeter of the triangle (p1, p2, k). Constraint: the segments k-p1 and
    k-p2 must not intersect any circular obstacle (penetration penalty > 0 means infeasible).
    """
    def __init__(self,
                obs_centers=FIG2_OBS_CENTERS,
                obs_radius=FIG2_OBS_RADIUS,
                device=torch.device('cuda')):
        self.device = device

        # define obstacle of shape circle with center at obs_center and raidus length obs_radius
        assert len(obs_centers) == len(obs_radius)
        self.obstacles = []
        for obs_center, obs_radius in zip(obs_centers, obs_radius):
            self.obstacles.append((torch.tensor(obs_center, dtype=torch.float32, device=self.device), obs_radius))

    @property
    def input_dim(self):
        return 4

    @property
    def param_dim(self):
        return 2

    def objective_function(self, x, k):
        """
        Maximize the perimeter of the triangle formed by the points x1, x2, and k

        Args:
            x (torch.Tensor): (B, input_dim) The input to the objective function
            k (torch.Tensor): (B, param_dim) The parameter to the objective function

        Returns:
            torch.Tensor: (B,) The value of the objective function
        """
        if x.dim() != k.dim():
            x = x.unsqueeze(1).repeat(1, k.shape[1], 1)
        x1 = x[...,:2]
        x2 = x[...,2:]

        ks = sample_points_from_circle(k[...,0], k[...,1]).to(x.device)

        kx1 = torch.linalg.norm(x1 - ks, axis=-1)
        kx2 = torch.linalg.norm(x2 - ks, axis=-1)
        x1x2 = torch.linalg.norm(x1 - x2, axis=-1)

        Obj = -(kx1 + kx2 + x1x2)

        return Obj

    def constraint_function(self, x, k):
        """
        Constraint function for the donut problem >0 if invalid

        Args:
            x (torch.Tensor): (B, 4) The input to the constraint function
            k (torch.Tensor): (B, 2) The parameter to the constraint function

        Returns:
            torch.Tensor: (B,) The value of the constraint function
        """
        if x.dim() != k.dim():
            x = x.unsqueeze(1).repeat(1, k.shape[1], 1)

        x1 = x[...,:2]
        x2 = x[...,2:]

        ks = sample_points_from_circle(k[...,0], k[...,1]).to(x.device)

        penalty = torch.zeros(x.shape[0]).to(x.device)
        for obs_center, obs_radius in self.obstacles:
            penalty += line_intersects_circle(x1, ks, obs_center, obs_radius, return_penalty=True)
            penalty += line_intersects_circle(x2, ks, obs_center, obs_radius, return_penalty=True)

        return penalty

    def global_optimal_value(self, num_samples=10000, seed=0):
        """
        The global optimal should exist at x=(r1, theta1), (r1, theta2) and k=(r2, 0).
        Estimated with ``num_samples`` random angle pairs from a local generator seeded with ``seed``.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        theta_samples = torch.rand(size=(num_samples, 2), device=self.device, generator=gen)

        r_out = self.to_tensor(R_OUTER)

        x1 = sample_points_from_circle(self.to_tensor(1.0), theta_samples[...,0], radius_scale=r_out)
        x2 = sample_points_from_circle(self.to_tensor(1.0), theta_samples[...,1], radius_scale=r_out)
        xs = torch.cat([x1, x2], dim=-1)
        k = self.to_tensor(np.array([1.0, 0.0])).unsqueeze(0)

        objectives = self.objective_function(xs, k)

        return objectives.min().detach().cpu().numpy()

    def is_ood(self, x):
        r0 = R_INNER
        r1 = R_OUTER

        point1_ood = torch.logical_or(torch.linalg.norm(x[...,:2], axis=-1) < r0, torch.linalg.norm(x[...,:2], axis=-1) > r1)
        point2_ood = torch.logical_or(torch.linalg.norm(x[...,2:], axis=-1) < r0, torch.linalg.norm(x[...,2:], axis=-1) > r1)

        return torch.logical_or(point1_ood, point2_ood)
