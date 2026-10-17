"""JM2D samplers: ``JM2DSampler`` (constrained generation, Fig. 9) and ``JM2DJointSampler`` (joint x/k, Fig. 2).

Moved from the original code. ``guided_step`` bodies are unchanged
except for hyper-parameter attribute renames. Removed: an unused attribute, an interactive-debugger NaN trap,
commented-out code, and a duplicate of ``DenoiserMLP.get_noise``.
"""
import einops
import torch

from jm2d_toy.samplers.base import BaseJointSampler, BaseSampler
from jm2d_toy.tasks.problem import TFGJointProblem, TFGProblem


class JM2DSampler(BaseSampler):
    """JM2D for constrained generation.

    At every reverse step t, x_t is replicated N times and each copy is denoised to a clean sample by the
    full reverse chain. The Tweedie noise of each copy is weighted by softmax(-J/lambda) (with constraint
    rejection) and the weighted noise replaces epsilon_theta in the DDPM step.
    """
    def __init__(self,
                 model_path: str,
                 problem: TFGProblem,
                 num_mc_samples=512,
                 temperature=0.1,
                 **base_kwargs
                 ):
        """
        Args:
            model_path: Path to pretrained diffusion model
            problem: Problem instance defining objective and constraints
            num_mc_samples: N, Monte-Carlo samples per x_t
            temperature: lambda, softmax temperature of the importance weights
        """
        super().__init__(model_path, problem, **base_kwargs)

        self.num_mc_samples = num_mc_samples
        self.temperature = temperature

    def _name(self):
        return "JM2D"

    def guided_step(self, nets, naction, timestep, eta=0, use_constraints=False, verbose=True):
        """Perform one guided reverse step.

        Args:
            nets: Noise prediction model network
            naction: Noisy action tensor (B, A)
            timestep: Current diffusion timestep (t-1)
            eta: DDIM eta parameter (if used)
            verbose: Whether to print debug info

        Returns:
            naction_next: Updated noisy action
        """
        if timestep == 0:
            return self.step_normal(nets, naction, timestep)

        B, A = naction.shape
        N = self.num_mc_samples
        sigma_t = self.get_sigma_t(self.noise_scheduler, timestep, eta=eta)

        flat_samples = naction.clone().unsqueeze(1).repeat(1, N, 1).reshape(-1, A)

        # Simulate full trajectory for each sample
        inner_timesteps = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= timestep]
        inner_naction = flat_samples.clone()
        for inner_t in inner_timesteps:
            inner_naction = self.step_normal(nets=nets, naction=inner_naction, timestep=inner_t)

        clean_action_samples = inner_naction.clone().reshape(B, N, A).reshape(-1, A)

        naction_samples = naction.clone().reshape(B, 1, A).repeat(1, N, 1).reshape(-1, A)

        tweedie_score = self.get_noise(
            noisy_samples=naction_samples,
            original_samples=clean_action_samples,
            timesteps=timestep
        ) # (B*N, A)

        clean_action_samples = clean_action_samples.reshape(B, N, A)

        objectives = self.problem.objective_function(clean_action_samples)
        objectives_orig = objectives.clone()

        objectives = objectives - objectives.min(dim=1, keepdim=True)[0]
        weights = torch.nn.functional.softmax(-objectives / self.temperature, dim=-1)

        if use_constraints:
            constraints = self.problem.constraint_function(clean_action_samples) # (B, N)
            constraints_orig = constraints.clone()
            weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=1000 * (1-sigma_t))

        self.assert_weights_normalized(weights)

        tweedie_score = tweedie_score.reshape(B, N, A)

        mask_to_update = (weights > 0).any(dim=1)

        noise_optimal = nets["policy"]["noise_pred_net"](sample = naction, timestep = timestep)
        noise_optimal[mask_to_update] = (tweedie_score * weights.unsqueeze(-1)).sum(dim=-2)[mask_to_update]

        naction_next = self.noise_scheduler.step(
            model_output=noise_optimal,
            timestep=timestep,
            sample=naction
        ).prev_sample

        if verbose:
            naction_next_repr = naction_next[0].detach().cpu().numpy().round(2)
            debug_msg = (
                f"(t={timestep}) | "
                f"X: {naction_next_repr} | "
                f"Mean obj: {objectives_orig.mean():.2f} | "
                f"ESS: {self.effective_sample_size(weights).mean():.2f}"
            )
            if use_constraints:
                valid_ratio = 1 - ((constraints_orig > 0).sum(dim=-1)/(N)).mean()
                debug_msg += f" | Valid ratio: {valid_ratio:.2f}"
            self.disp(debug_msg)

        self.guide_log_cache['ess']                   = self.effective_sample_size(weights)
        self.guide_log_cache['objective_all']         = objectives_orig
        self.guide_log_cache['clean_samples']         = clean_action_samples
        self.guide_log_cache['weights']               = weights

        return naction_next


class JM2DJointSampler(BaseJointSampler):
    """JM2D for joint sampling of data x and parameter k (Alg. 2).

    The N clean samples of x (full reverse chain from x_t) are paired with N_K perturbed k samples;
    the N*N_K pairs are weighted by softmax(-J(x, k)/lambda) with constraint rejection, and both the noise
    of x and the value of k are updated with the same weights.
    """
    def __init__(self,
                 model_path: str,
                 problem: TFGJointProblem,
                 num_mc_samples=128,
                 num_param_samples=128,
                 temperature=0.1,
                 **base_kwargs
                 ):
        """
        Args:
            model_path: Path to pretrained diffusion model
            problem: Problem instance defining objective and constraints
            num_mc_samples: N, Monte-Carlo samples of x per x_t
            num_param_samples: N_K, samples of k per x sample
            temperature: lambda, softmax temperature of the importance weights
        """
        super().__init__(model_path, problem, **base_kwargs)

        self.num_mc_samples = num_mc_samples
        self.num_param_samples = num_param_samples
        self.temperature = temperature

    def _name(self):
        return "JM2DJoint"

    def guided_step(self, nets, naction, nparam, timestep, eta=0, use_constraints=False, verbose=True):
        """Perform one step of guided diffusion with joint optimization of actions and parameters.

        Args:
            nets: Noise prediction model network
            naction: Noisy action tensor (B, A)
            nparam: Noisy parameter tensor (B, P)
            timestep: Current diffusion timestep (t-1)
            eta: DDIM eta parameter (if used)
            use_constraints: Whether to apply constraints during optimization
            verbose: Whether to print debug info

        Returns:
            naction_next: Updated noisy action tensor (B, A)
            nparam_next: Updated parameter tensor (B, P)
        """
        if timestep == 0:
            final_action = self.step_normal(nets, naction, timestep)
            final_param  = self.sampling_based_optimization(final_action, nparam, use_constraints=use_constraints)
            return final_action, final_param

        # Get dimensions and setup sampling
        B, A = naction.shape
        B, P = nparam.shape
        N = self.num_mc_samples
        N_K = self.num_param_samples

        """
        Step 1. Monte-Carlo Sampling (clean_action_samples, tweedie_noise_samples)
        """
        flat_samples = naction.clone().unsqueeze(1).repeat(1, N, 1).reshape(-1, A)
        inner_timesteps = self.noise_scheduler.timesteps[self.noise_scheduler.timesteps <= timestep]
        inner_naction = flat_samples.clone()
        for inner_t in inner_timesteps:
            inner_naction = self.step_normal(nets=nets, naction=inner_naction, timestep=inner_t)
        clean_action_samples = inner_naction.clone() # (B*N, A)

        # Calculate Tweedie Noise
        naction_repeated = einops.repeat(naction.clone(), 'b a -> (b n) a', n=N) # (B*N, A)
        tweedie_noise_samples = self.get_noise(
            noisy_samples=naction_repeated,
            original_samples=clean_action_samples,
            timesteps=timestep
        ) # (B*N, A)

        tweedie_noise_samples = einops.rearrange(tweedie_noise_samples, '(b n) a -> b n a', b=B, n=N) # (B, N, A)
        tweedie_noise_samples = einops.repeat(tweedie_noise_samples, 'b n a -> b (n k) a', k=N_K) # (B, N*N_K, A)

        clean_action_samples = einops.rearrange(clean_action_samples, '(b n) a -> b n 1 a', b=B, n=N)
        clean_action_samples = einops.repeat(clean_action_samples, 'b n 1 a -> b n k a', k=N_K)
        clean_action_samples = einops.rearrange(clean_action_samples, 'b n k a -> b (n k) a') # (B, N*N_K, A)

        """
        Step 2. Monte-Carlo Sampling (param_samples, dparam_samples)
        """
        cov_param = self.get_param_covariance(timestep)
        param_perturbations = torch.distributions.MultivariateNormal(
            loc=torch.zeros(P),
            covariance_matrix=cov_param
        ).sample((B, N, N_K)).to(nparam.device)

        nparam_repeated = einops.repeat(nparam.clone(), 'b p -> b n k p', n=N, k=N_K) # (B, N, N_K, P)
        param_samples = nparam_repeated + param_perturbations # (B, N, N_K, P)
        param_samples = torch.clip(param_samples, -1, 1) # (B, N, N_K, P)
        param_samples = einops.rearrange(param_samples, 'b n k p -> b (n k) p') # (B, N*N_K, P)

        # Calculate objectives
        objectives = self.problem.objective_function(clean_action_samples, param_samples) # (B, N*N_K)
        objectives_orig = objectives.clone()

        # Calculate importance weights
        objectives = objectives - objectives.min(dim=-1, keepdim=True)[0]  # numerical stability
        weights = torch.nn.functional.softmax(-objectives / self.temperature, dim=-1) # (B, N*N_K)

        if use_constraints:
            constraints = self.problem.constraint_function(clean_action_samples.reshape(-1, A), param_samples.reshape(-1, P)) # (B*N*N_K)
            constraints = constraints.reshape(B, N*N_K)
            constraints_orig = constraints.clone()

            sigma_t   = self.get_sigma_t(self.noise_scheduler, timestep, eta=eta)
            lag_coeff = 1000 * (1-sigma_t)
            weights   = self.update_weights_from_constraints(weights, constraints, lag_coeff)

        self.assert_weights_normalized(weights)

        mask_to_update = (weights > 0).any(dim=1)

        noise_optimal                 = nets["policy"]["noise_pred_net"](sample = naction, timestep = timestep)
        noise_optimal[mask_to_update] = (tweedie_noise_samples * weights.unsqueeze(-1)).sum(dim=-2)[mask_to_update]

        param_optimal                 = nparam.clone()
        param_optimal[mask_to_update] = (param_samples * weights.unsqueeze(-1)).sum(dim=-2)[mask_to_update]

        # Update action using noise scheduler
        naction_next = self.noise_scheduler.step(
            model_output=noise_optimal,
            timestep=timestep,
            sample=naction
        ).prev_sample

        # Clip parameters to valid range
        nparam_next = torch.clip(param_optimal, -1, 1)

        return naction_next, nparam_next
