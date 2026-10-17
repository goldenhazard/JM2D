"""JM2D ("Ours"): the um_dpcc JM2D sampler (legacy variant 'model_based-dpcc-c-tightened').

``u_steps`` selects the clean-sample estimate used inside the sampler (``K_value`` of the original code):
'full' (Table 2) = full denoising, implemented as K = n_timesteps; an integer u (Table 3) is passed unchanged.
Runs on the um_dpcc sampling loop (``DPCCDiffusion`` with ``use_jm2d=True``); exact semantics of ``u_steps`` in
``diffuser/models/dpcc_diffusion.py``.
"""
from diffuser.models.dpcc_diffusion import DPCCDiffusion
from diffuser.sampling import DPCCPolicy

from .base import Planner


def resolve_u_steps(u_steps, n_timesteps):
    """'full' -> n_timesteps (full denoising from every step); integers unchanged."""
    if u_steps in ("full", None):
        return int(n_timesteps)
    u = int(u_steps)
    if u < 0:
        raise ValueError(f"u_steps must be >= 0 or 'full', got {u_steps!r}")
    return u


class JM2DPlanner(Planner):
    name = "jm2d"
    policy_class = DPCCPolicy
    diffusion_class = DPCCDiffusion

    def __init__(self, num_mc_samples=100, temperature=0.05, u_steps="full", verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.num_mc_samples = num_mc_samples
        self.temperature = temperature
        self.u_steps = u_steps
        self.verbose = verbose
        self.u_steps_resolved = None

    @property
    def label(self):
        return self.name if self.u_steps in ("full", None) else f"{self.name}-u{int(self.u_steps)}"

    def settings(self):
        return dict(super().settings(), num_mc_samples=self.num_mc_samples, temperature=self.temperature,
                    u_steps=self.u_steps, K_value=self.u_steps_resolved)

    def prepare_diffusion(self, diffusion):
        diffusion.set_jm2d_sampler(num_mc_samples=self.num_mc_samples, temperature=self.temperature,
                                   verbose=self.verbose)
        self.u_steps_resolved = resolve_u_steps(self.u_steps, diffusion.n_timesteps)

    def policy_kwargs(self):
        return dict(use_jm2d=True, u_steps=self.u_steps_resolved)
