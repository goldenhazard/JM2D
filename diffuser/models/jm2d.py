"""JM2D ("Ours" in Table 2 / Table 3): the JM2D mean-steering sampler.

Moved from SafeRobomimic@origin/um_dpcc (a71ff97a) ``diffuser/models/model_based_optimization.py``; algorithm body
unchanged, names changed: ``ModelBasedOptimization`` -> ``JM2DSampler``, ``N_sample_action`` -> ``num_mc_samples``,
``temp_mppi`` -> ``temperature``, ``_name() == "Yongxin"`` -> ``"JM2D"``, ``effective_sample_size(method="Yongxin")``
-> ``method="normalized"``.

It is called from the um_dpcc sampling loop (``diffuser/models/dpcc_diffusion.py``, ``DPCCDiffusion`` with
``use_jm2d=True``), which also provides the clean-sample estimate with ``u_steps`` denoising steps (see there).
"""
import numpy as np
import torch
import torch.nn as nn



class JM2DSampler(nn.Module):
    """JM2D mean steering for guided diffusion (was ``ModelBasedOptimization``).

    For each of the B noisy plans, N copies are denoised to clean samples, scored by the constraint violation
    of the model-based check (``projector.get_projection_costs``), and the noise estimate is the
    softmax(-1000 * violation / temperature)-weighted average of the Tweedie scores of the N samples.
    """
    def __init__(self,
            num_mc_samples=100,
            temperature=0.05,
            lam=1000,
            verbose=False,
        ):
        """Initialize the sampler.

        Args:
            num_mc_samples: Number of clean samples per plan (N; was N_sample_action)
            temperature: Temperature of the softmax weighting (was temp_mppi)
            lam: (unused; the penalty coefficient is ``lag_coeff`` = 1000 in ``optimize``)
            verbose: print ESS / valid ratio at every denoising step
        """

        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.temperature = temperature
        self.lam = lam
        self.constraint_mode = 'penalty'
        self.verbose = verbose
        self.log = {}

    def _name(self):
        return "JM2D"

    def optimize(
            self,
            model,
            naction,
            cond,
            timestep,
            projector,
            constraints,
            func_full_denoising,
            func_get_noise_from_xstart,
            sigma_t,
            inpainting_func=None,
        ):
        """
        Args
            naction: (B, H, A)
        """
        B, H, A = naction.shape
        N = self.num_mc_samples

        naction_repeated = naction.unsqueeze(1).repeat(1, N, 1, 1).reshape(-1, H, A)

        timestep_orig = timestep.clone()
        timestep = timestep.repeat(N)

        cond_orig = {key: cond[key].clone() for key in cond}

        for key in cond_orig:
            _, CD = cond_orig[key].shape
            cond_orig[key] = cond_orig[key].unsqueeze(1).repeat(1, N, 1).reshape(-1, CD)

        clean_action_samples = func_full_denoising(cond_orig, timestep, naction_repeated)

        tweedie_score = func_get_noise_from_xstart(clean_action_samples, timestep, naction_repeated)

        objectives, equality_constraints, inequality_constraints = projector.get_projection_costs(
            clean_action_samples,
            constraints
        )

        objectives = torch.Tensor(objectives).to(naction.device)
        inequality_constraints = torch.Tensor(inequality_constraints).to(naction.device)

        weights = torch.ones(B,N).to(naction.device)

        weights = torch.ones_like(weights)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        for constraints in [inequality_constraints]:
            constraints = constraints.reshape(B, N)
            lag_coeff = 1000
            weights = self.update_weights_from_constraints(
                weights,
                constraints,
                lag_coeff=lag_coeff,
                constraint_mode=self.constraint_mode
            )

        self.assert_weights_normalized(weights)

        tweedie_score = tweedie_score.reshape(B, N, H, A)
        tweedie_score = tweedie_score.reshape(B, N, H*A)
        noise_optimal = (tweedie_score * weights.unsqueeze(-1)).sum(dim=-2)
        noise_optimal = noise_optimal.reshape(B, H, A)

        # For logging purpose
        ess = self.effective_sample_size(weights)
        valid_ratio = (constraints < 1e-3).sum(dim=-1) / N

        if self.verbose:
            self.disp(f"ESS: {ess.mean():.2f} | Valid ratio: {valid_ratio.mean():.2f}")

        self.log = dict(
            timestep = timestep_orig,
            clean_samples = clean_action_samples.reshape(B, N, H, A),
            weights = weights,
            noise_optimal = noise_optimal,
        )

        return noise_optimal

    def disp(self, msg):
        print('\033[93m' + str(self._name()) + ': ' + msg + '\033[0m')

    def effective_sample_size(self, weights, method="normalized"):
        """ Effective sample size

        Args
            weights (torch.tensor) (B1, B2, ..., Bn, N)
            method: "Kish" (1 / sum w^2) or "normalized" (Kish ESS divided by N)

        Returns:
            N_eff (torch.tensor) (B1, B2, ..., Bn)
        """
        valid_mask = (weights > 0).any(dim=-1)

        N_eff_final = torch.zeros_like(weights[..., 0])
        weights_valid = weights[valid_mask]

        if method == "Kish":
            W2    = (weights_valid/weights_valid.sum(dim=-1, keepdim=True))**2
            N_eff = 1/W2.sum(dim=-1)

        if method == "normalized":
            W2        = (weights_valid/weights_valid.sum(dim=-1, keepdim=True))**2
            n_samples = weights_valid.shape[-1]
            N_eff     = 1/(n_samples * W2.sum(dim=-1))

        N_eff_final[valid_mask] = N_eff

        return N_eff_final

    def assert_weights_normalized(self, weights, atol=1e-4):
        """
        Assert that the weights are normalized

        Args:
            weights (torch.tensor): The weights to check (B, N)
        """
        assert weights.ndim == 2, "Weights must be a 2D tensor"

        weight_close_to_one  = (weights.sum(dim=-1) - 1).abs() < atol
        weight_close_to_zero = (weights.sum(dim=-1)).abs() < atol

        assert torch.logical_or(weight_close_to_one, weight_close_to_zero).all(), f"Weights are not normalized {weights.sum(dim=-1)}"

    def update_weights_from_constraints(self, weights, constraints, lag_coeff=100, constraint_mode='penalty'):
        """
        Update the weights from the constraints

        Args:
            weights (torch.tensor): The weights (B, N)
            constraints (torch.tensor): The constraints (B, N)
            lag_coeff: coefficient for penalty method
            constraint_mode: which constraint handling method to use (only 'penalty' is used in the paper)

        Returns:
            weights (torch.tensor): The updated weights
        """
        assert weights.shape == constraints.shape, "Weights and constraints must have the same shape"

        if constraint_mode == 'penalty':
            # numerical stability
            constraints = constraints - constraints.min(dim=1, keepdim=True)[0]

            # update weights
            constraint_weights = torch.nn.functional.softmax(-lag_coeff * constraints / self.temperature, dim=1)
            old_weights = weights.clone()
            weights *= constraint_weights
            weights = weights / weights.sum(dim=1, keepdim=True)

            if torch.isnan(weights).any():
                weights = old_weights.clone()

        else:
            raise ValueError(f"Invalid constraint mode: {constraint_mode}")

        return weights
