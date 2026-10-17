"""Gibbs baseline for joint x/k sampling (Fig. 2).

Moved from the original code; legacy config section ``gibbs_exact``.
``guided_step`` is unchanged except for attribute renames.
"""
import torch

from jm2d_toy.samplers.base import BaseJointSampler
from jm2d_toy.tasks.problem import TFGJointProblem


class GibbsSampler(BaseJointSampler):
    """Alternates x | k (full-denoise Monte-Carlo Tweedie guidance) and k | x (perturb-and-weight)."""
    def __init__(self,
                 model_path: str,
                 problem: TFGJointProblem,
                 num_mc_samples=128,
                 num_param_samples=16384,
                 temperature=0.1,
                 **base_kwargs
                 ):
        """
        Args:
            model_path: Path to pretrained diffusion model
            problem: Problem instance defining objective and constraints
            num_mc_samples: N, Monte-Carlo samples of x per x_t (x | k step)
            num_param_samples: N_K, samples of k (k | x step)
            temperature: lambda, softmax temperature of the importance weights
        """
        super().__init__(model_path, problem, **base_kwargs)

        self.num_mc_samples = num_mc_samples
        self.num_param_samples = num_param_samples
        self.temperature = temperature

    def _name(self):
        return "Gibbs"

    def guided_step(self, nets, naction, nparam, timestep, eta=0, use_constraints=False, verbose=True):
        """Perform one step of Gibbs alternating guided diffusion.

        Args:
            nets: Noise prediction model network
            naction: Noisy action tensor (B, A)
            nparam: Noisy parameter tensor (B, P)
            timestep: Current diffusion timestep (t-1)
            eta: DDIM eta parameter (if used)
            verbose: Whether to print debug info

        Returns:
            naction_next, nparam_next
        """
        # At timestep 0, no guidance available - just do normal step
        if timestep == 0:
            final_action = self.step_normal(nets, naction, timestep)
            final_param  = self.sampling_based_optimization(final_action, nparam, use_constraints=use_constraints)
            return final_action, final_param

        # Get dimensions and setup sampling
        B, A = naction.shape
        B, P = nparam.shape
        N = self.num_mc_samples
        N_K = self.num_param_samples
        sigma_t, prev_timestep = self.get_sigma_t(self.noise_scheduler, timestep, eta=eta, return_prev_timestep=True)

        """
        Optimize x given k
        """
        flat_samples = naction.clone().unsqueeze(1).repeat(1, N, 1).reshape(-1, A)

        # Simulate full trajectory for each sample
        inner_timesteps = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= timestep]
        inner_naction = flat_samples.clone()
        for inner_t in inner_timesteps:
            inner_naction = self.step_normal(nets=nets, naction=inner_naction, timestep=inner_t)

        action_samples = inner_naction.clone().reshape(B, N, A).reshape(-1, A)
        naction_samples = naction.clone().reshape(B, 1, A).repeat(1, N, 1).reshape(-1, A)

        tweedie_noise = self.get_noise(
            noisy_samples=naction_samples,
            original_samples=action_samples,
            timesteps=timestep
        )

        action_samples = action_samples.reshape(B, N, A)
        param_samples = nparam.clone().unsqueeze(1).repeat(1, N, 1)
        param_samples = torch.clip(param_samples, -1, 1)

        # Calculate importance weights
        objectives = self.problem.objective_function(action_samples.reshape(-1, A), param_samples.reshape(-1, P))

        objectives_orig = objectives.clone()
        objectives = objectives.reshape(B, N)
        objectives = objectives - objectives.min(dim=1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(-objectives / self.temperature, dim=1) # (B, N)

        if use_constraints:
            constraints = self.problem.constraint_function(action_samples.reshape(-1, A), param_samples.reshape(-1, P)) # (B*N)
            constraints = constraints.reshape(B, N)
            constraints_orig = constraints.clone()

            lag_coeff = 1000 * (1-sigma_t)
            weights = self.update_weights_from_constraints(weights, constraints, lag_coeff)

        self.assert_weights_normalized(weights)

        tweedie_noise = tweedie_noise.reshape(B, N, A)
        mask_to_update = (weights > 0).any(dim=1)
        noise_optimal = nets["policy"]["noise_pred_net"](sample = naction, timestep = timestep)
        noise_optimal[mask_to_update] = (tweedie_noise * weights.unsqueeze(-1)).sum(dim=1)[mask_to_update]

        naction_next = self.noise_scheduler.step(
            model_output=noise_optimal,
            timestep=timestep,
            sample=naction
        )

        optimized_action_sample = naction_next.pred_original_sample
        naction_next = naction_next.prev_sample

        """
        Optimize k given x
        """

        action_samples = optimized_action_sample.clone().unsqueeze(1).repeat(1, N_K, 1)

        cov_param = self.get_param_covariance(timestep)

        param_perturbations = torch.distributions.MultivariateNormal(
            loc=torch.zeros(P),
            covariance_matrix=cov_param
        ).sample((B, N_K)).to(nparam.device)

        param_samples = nparam.unsqueeze(1).repeat(1, N_K, 1) + param_perturbations
        param_samples = torch.clip(param_samples, -1, 1)

        objectives = self.problem.objective_function(action_samples.reshape(-1, A), param_samples.reshape(-1, P))

        objectives_orig = objectives.clone()
        objectives = objectives.reshape(B, N_K)
        objectives = objectives - objectives.min(dim=1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(-objectives / self.temperature, dim=1) # (B, N_K)

        if use_constraints:
            constraints = self.problem.constraint_function(action_samples.reshape(-1, A), param_samples.reshape(-1, P)) # (B*N_K)
            constraints = constraints.reshape(B, N_K)
            constraints_orig = constraints.clone()

            lag_coeff = 1000 * (1-sigma_t)
            weights = self.update_weights_from_constraints(weights, constraints, lag_coeff)

        self.assert_weights_normalized(weights)
        mask_to_update = (weights > 0).any(dim=1)
        optimal_param = nparam.clone()
        optimal_param[mask_to_update] = (param_samples * weights.unsqueeze(-1)).sum(dim=1)[mask_to_update]

        nparam_next = torch.clip(optimal_param, -1, 1)

        if verbose:
            # Get first batch element and convert to numpy for printing
            naction_next_repr = naction_next[0].detach().cpu().numpy().round(2)
            nparam_next_repr = nparam_next[0].detach().cpu().numpy().round(2)

            # Calculate effective sample size from importance weights
            ess = self.effective_sample_size(weights).mean()

            # Format debug message with current timestep, action, parameter, objective and ESS
            debug_msg = (
                f"(t={timestep}) | "
                f"X: {naction_next_repr} | "
                f"K: {nparam_next_repr} | "
                f"Mean obj: {objectives_orig.mean():.2f} | "
                f"ESS: {ess:.2f}"
            )
            if use_constraints:
                valid_ratio = 1 - ((constraints_orig > 0).sum(dim=-1)/N_K).mean()
                debug_msg += f" | Valid ratio: {valid_ratio:.2f}"
            self.disp(debug_msg)

        return naction_next, nparam_next
