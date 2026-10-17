"""Sequential baseline for joint x/k sampling (Fig. 2).

Moved from the original code. The original class read the Gibbs
config section but only ever used its temperature; it now takes ``temperature`` directly (0.1 in both
original sections, so behaviour is unchanged).
"""
import torch

from jm2d_toy.samplers.base import BaseJointSampler
from jm2d_toy.tasks.problem import TFGJointProblem


class SequentialSampler(BaseJointSampler):
    """Unguided DDPM sampling of x, then sampling-based optimisation of k given the final x (t = 0 only)."""
    def __init__(self,
                 model_path: str,
                 problem: TFGJointProblem,
                 temperature=0.1,
                 **base_kwargs
                 ):
        """
        Args:
            model_path: Path to pretrained diffusion model
            problem: Problem instance defining objective and constraints
            temperature: lambda, softmax temperature of the k optimisation
        """
        super().__init__(model_path, problem, **base_kwargs)

        self.temperature = temperature

    def _name(self):
        return "Sequential"

    def guided_step(self, nets, naction, nparam, timestep, eta=0, use_constraints=False, verbose=True):
        """x_t -> x_{t-1} without guidance; at t = 0 optimise k for the final x.

        Args:
            nets: Noise prediction model network
            naction: Noisy action tensor (B, A)
            nparam: Noisy parameter tensor (B, P)
            timestep: Current diffusion timestep (t-1)

        Returns:
            naction_next, nparam_next
        """
        if timestep > 0:
            return self.step_normal(nets, naction, timestep), nparam
        else:
            final_action = self.step_normal(nets, naction, timestep)
            nparam = torch.clip(nparam, -1, 1)
            final_param  = self.sampling_based_optimization(final_action, nparam, use_constraints=use_constraints)
            return final_action, final_param
