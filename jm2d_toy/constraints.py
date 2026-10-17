"""Obstacle constraints g(x) >= 0 (feasible), used by ``DonutConstrainedProblem`` and the projection baseline.

Moved from legacy ``dev/constraints/{base,obstacles}.py``: ``Constraint``, ``EllipseConstraint``,
``CircleConstraint`` and ``SupEllipseConstraint`` (the "rectangle" of ``great_wall_rectangle``).
Removed: unused ``InCircle``/``Wavy``/``Kidney``/``DonutEllipse`` constraints and ``normalize*`` helpers.
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch


class Constraint(ABC):
    def __init__(self, name="Constraint", device=torch.device("cuda"), dtype=torch.float32):
        """
        Base class for constraints with evaluation, gradient, and visualization.

        Parameters:
            name: Name of the constraint.
        """
        self.name = name
        self.device = device
        self.dtype = dtype
        self.normalized = False

    @abstractmethod
    def evaluate(self, x):
        """
        Evaluates g(x) >= 0, non-negative for feasible region

        Args:
            x: torch.Tensor of shape (..., n_dim)

        Returns:
            torch.Tensor of shape (..., 1)
        """
        pass

    def gradient(self, x):
        """ Computes the gradient of g(x). """
        with torch.enable_grad():
            x = x.requires_grad_(True)
            g = self.evaluate(x)
            grad = torch.autograd.grad(g.sum(), x)[0]

        return grad

    def evaluate_np(self, x):
        assert isinstance(x, np.ndarray)
        x = self.to_tensor(x)
        return self.evaluate(x).cpu().numpy()

    def gradient_np(self, x):
        assert isinstance(x, np.ndarray)
        x = self.to_tensor(x)
        return self.gradient(x).cpu().numpy()

    def plot(self, ax=None, xlim=(-2, 2), ylim=(-2, 2), grid_size=100, swap_xy=False, show=False, color='red', alpha=0.2):
        """
        Plots the level set g(x, y) = 0 and fills the region where g(x,y) < 0.

        Args:
            ax: matplotlib axis to plot on. If None, uses current axis
            xlim: x-axis limits
            ylim: y-axis limits
            grid_size: number of points in grid
            swap_xy: if True, plots (y,x,g) instead of (x,y,g)
            color: color of the contour and fill
            alpha: transparency of the fill (0 is transparent, 1 is solid)
        """
        if ax is None:
            ax = plt.gca()

        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)

        sample = torch.tensor(np.stack([X, Y], axis=-1)).float()

        sample = self.to_tensor(sample)
        G = self.evaluate(sample)
        G = G.squeeze(-1)
        G = G.cpu().numpy()

        if swap_xy:
            ax.contourf(Y, X, G, levels=[-np.inf, 0], colors=[color], alpha=alpha)
            ax.contour(Y, X, G, levels=[0], colors=color)
        else:
            ax.contourf(X, Y, G, levels=[-np.inf, 0], colors=[color], alpha=alpha)
            ax.contour(X, Y, G, levels=[0], colors=color)

        if show:
            ax.axis('equal')
            ax.set_title(self.name)
            plt.show()

        return ax

    def to_tensor(self, x):
        """ Converts x to a tensor. """
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        elif isinstance(x, list):
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")


class EllipseConstraint(Constraint):
    def __init__(self, center, radius, device=torch.device("cuda"), dtype=torch.float32, name="Ellipse"):
        super().__init__(name=name, device=device, dtype=dtype)

        self.center = self.to_tensor(center)
        self.radius = self.to_tensor(radius)

        self.center_np = center
        self.radius_np = radius

    def evaluate(self, x):
        assert x.shape[-1] == 2
        Cons = ((x[..., 0] - self.center[0]) / self.radius[0])**2 + ((x[..., 1] - self.center[1]) / self.radius[1])**2 - 1
        Cons = Cons.unsqueeze(-1)

        if torch.any(torch.isnan(Cons)):
            print(f"Warning: NaN values detected in {self.name} constraint evaluation")

        return Cons

    def gradient(self, x):
        Grad = 2*(x - self.center)/(self.radius ** 2)
        return Grad

    def evaluate_np(self, x):
        Cons = ((x[..., 0] - self.center_np[0]) / self.radius_np[0])**2 + ((x[..., 1] - self.center_np[1]) / self.radius_np[1])**2 - 1
        return Cons

    def gradient_np(self, x):
        Grad = 2*(x - self.center_np)/(self.radius_np ** 2)
        return Grad


class CircleConstraint(EllipseConstraint):
    def __init__(self, center, radius, device=torch.device("cuda"), dtype=torch.float32):
        assert isinstance(radius, float) or isinstance(radius, int)
        radius = np.array(radius).repeat(center.shape[-1])
        super().__init__(name="Circle", device=device, dtype=dtype, center=center, radius=radius)


class SupEllipseConstraint(Constraint):
    """Super-ellipse |dx/rx|^p + |dy/ry|^q - 1 >= 0 (p = q = 4 approximates a rectangle)."""
    def __init__(self, center, radius, p=4, q=4, device=torch.device("cuda"), dtype=torch.float32):
        super().__init__(name="SupEllipse", device=device, dtype=dtype)

        self.center = self.to_tensor(center)
        self.radius = self.to_tensor(radius)
        self.p = p
        self.q = q

    def evaluate(self, x):
        assert x.shape[-1] == 2
        Cons = ((x[..., 0] - self.center[0]) / self.radius[0])**self.p + ((x[..., 1] - self.center[1]) / self.radius[1])**self.q - 1
        Cons = Cons.unsqueeze(-1)
        return Cons
