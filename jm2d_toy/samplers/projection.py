"""Projection baseline for constrained generation (Fig. 9).

Was ``ProjectionDP`` (legacy ``dev/policies/projection_guided_policy.py:8``), ``classic`` mode only: every
x_{t-1} from the unguided DDPM step is projected onto the feasible set with scipy ``trust-constr``.
Removed: the ``classic_informed`` / ``classic_clean`` modes and a leftover debug print.
"""
import numpy as np
import torch
from scipy.optimize import Bounds, minimize

from jm2d_toy.samplers.base import BaseSampler
from jm2d_toy.tasks.problem import TFGProblem


class ProjectionSampler(BaseSampler):
    """DDPM step followed by an L2 projection of x_{t-1} onto {x : obstacle constraints satisfied}."""
    def __init__(self,
                 model_path: str,
                 problem: TFGProblem,
                 solver='trust-constr',
                 max_iter=100,
                 tol=1e-5,
                 **base_kwargs
                 ):
        """
        Args:
            model_path: Path to pretrained diffusion model
            problem: Problem instance with ``obstacles`` (``Constraint`` objects, >= 0 feasible)
            solver: scipy.optimize.minimize method (trust-constr for strict constraint satisfaction)
            max_iter: solver iteration limit
            tol: solver tolerance
        """
        super().__init__(model_path, problem, **base_kwargs)

        self.solver = solver
        self.solver_max_iter = max_iter
        self.solver_tol = tol

    def _name(self):
        return "Projection"

    def project(self, naction):
        """
        Project the noisy action to the feasible set

        min 0.5 * (x-naction)^T Q (x-naction)
        s.t. g_i(x) >= 0 for every obstacle i

        Args:
            naction: (B, A) Noisy action tensor
        """
        assert hasattr(self.problem, 'obstacles'), "Now only supporting obstacle avoidance with sphere-shaped obstacles"

        B, A = naction.shape

        obstacles = self.problem.obstacles

        naction_np = naction.detach().cpu().numpy()

        # Build projection cost
        Q = np.eye(A)
        r = -naction_np @ Q

        # Build constraints
        constraints = ()

        for obstacle in obstacles:
            constraints += ({
                'type': 'ineq',
                'fun': lambda x, obstacle=obstacle: obstacle.evaluate_np(x),
                'jac': lambda x, obstacle=obstacle: obstacle.gradient_np(x)
            },)

        # Solve the projection problem
        sol = np.zeros_like(naction_np)
        for i in range(B):
            cost_fun = lambda x: 0.5 * x @ Q @ x + r[i] @ x
            jac_cost_fun = lambda x: Q @ x + r[i]

            # -5 is selected to be safe for all "noisy samples"
            res = minimize(fun = cost_fun,
                           jac = jac_cost_fun,
                           method=self.solver,
                           x0  = naction_np[i],
                           constraints = constraints,
                           bounds=Bounds(-5 * np.ones_like(naction_np[i]), 5 * np.ones_like(naction_np[i])),
                           tol=self.solver_tol,
                           options={'maxiter': self.solver_max_iter, 'disp': False}
            )

            sol[i] = res.x

            if not res.success:
                if self.verbose:
                    print(f"Projection failed for sample {i}, msg: {res.message}")
                    print(sol[i] - naction_np[i])
            elif self.verbose:
                for constraint in constraints:
                    if constraint['fun'](sol[i]) < -0.01:
                        print(f"Constraint {constraint} violated for sample {i}")

        naction_next = torch.tensor(sol, device=naction.device, dtype=naction.dtype)

        return naction_next

    def guided_step(self, nets, naction, timestep, eta=0, verbose=True, use_constraints=False):
        """DDPM step x_t -> x_{t-1}, then project x_{t-1} (``use_constraints`` is not used).

        Args:
            nets: Noise prediction model network
            naction: Noisy action tensor (B, A)
            timestep: Current diffusion timestep (t-1)

        Returns:
            naction_next: Updated noisy action
        """
        naction_next = self.step_normal(nets, naction, timestep)
        naction_next = self.project(naction_next)

        return naction_next
