"""DPCC: projection of every denoising step onto the padded constraint set (legacy variant 'dpcc-c-tightened').

Code path of SafeRobomimic@origin/um_dpcc (a71ff97a, ``scripts/eval_dev.py``), the runner behind the stored Table 2
DPCC numbers: the um_dpcc sampling loop (``DPCCDiffusion``, ``use_jm2d=False``) projects the batch with
``Projector.project`` (scipy SLSQP) after every denoising step, the constraints include the dynamics rows, and the
executed plan is the one with the smallest summed projection cost.
"""
from diffuser.models.dpcc_diffusion import DPCCDiffusion
from diffuser.sampling import DPCCPolicy

from .base import Planner


class DPCCPlanner(Planner):
    name = "dpcc"
    policy_class = DPCCPolicy
    diffusion_class = DPCCDiffusion

    def policy_kwargs(self):
        return dict(use_jm2d=False, u_steps=0)
