"""Problem interfaces (legacy ``dev/problems/problem.py``: ``TFGProblem`` L6, ``TFGJointProblem`` L68).

A problem defines an objective J (minimised) and a constraint function (> 0 means infeasible) on clean
samples x (and parameters k for joint problems). Removed from the abstract interface: ``get_dataloader``
and ``plot_results`` (plotting lives in ``jm2d_toy.plotting``; data generation in ``jm2d_toy.data``).
"""
from abc import ABC, abstractmethod

import numpy as np
import torch


class TFGProblem(ABC):
    """Training-free guided sampling problem over x."""

    @abstractmethod
    def objective_function(self, x):
        """ Minimize this function

        Args:
            x (torch.Tensor): (..., input_dim)

        Returns:
            torch.Tensor: (...,) objective values
        """
        pass

    @abstractmethod
    def constraint_function(self, x):
        """(..., input_dim) -> (...,); > 0 means infeasible."""
        pass

    @abstractmethod
    def is_ood(self, x):
        """(..., input_dim) -> (...,) bool: outside the data support."""
        pass

    @property
    @abstractmethod
    def input_dim(self):
        pass

    def to_tensor(self, x):
        """
        Convert x to a tensor
        """
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, (int, float)):
            return torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    def is_valid(self, x):
        """
        Check if the sample x is valid
        """
        Cons = self.constraint_function(self.to_tensor(x)) # (B,)
        is_valid = (Cons <= 0.0)

        return is_valid


class TFGJointProblem(TFGProblem):
    """
    A problem that jointly samples the data x and the parameter k
    """
    @abstractmethod
    def objective_function(self, x, k):
        """
        Minimize this function

        Args:
            x (torch.Tensor): (B, input_dim) The input to the objective function
            k (torch.Tensor): (B, param_dim) The parameter to the objective function

        Returns:
            torch.Tensor: (B,) The value of the objective function
        """
        pass

    @abstractmethod
    def constraint_function(self, x, k):
        """
        Penalty function for the constraint

        Args:
            x (torch.Tensor): (B, input_dim) The input to the objective function
            k (torch.Tensor): (B, param_dim) The parameter to the objective function

        Returns:
            torch.Tensor: (B,) The value of the constraint function (> 0 means infeasible)
        """
        pass

    @property
    @abstractmethod
    def param_dim(self):
        pass

    def is_valid(self, x, k):
        """
        Check if the sample x and parameter k are compatible (constraint satisfied)

        Args:
            x (torch.Tensor): (B, input_dim)
            k (torch.Tensor): (B, param_dim)

        Returns:
            torch.Tensor: (B,) bool
        """
        Cons = self.constraint_function(self.to_tensor(x), self.to_tensor(k)) # (B,)
        is_valid = (Cons <= 0.0)

        return is_valid
