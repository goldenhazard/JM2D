"""Guided-sampling base classes.

Moved from legacy ``dev/policies/base_guided_policy.py`` (b649add0):
``GuidedDiffusionPolicy`` (L26) -> ``BaseSampler`` and ``GuidedDiffusionPolicyJoint`` (L325) -> ``BaseJointSampler``.
Sampling loops, weight updates and the MBD-style k optimisation are unchanged; hyper-parameter attributes
are renamed (``temperature``, ``num_opt_samples``). Removed: the unused
``use_ood_penalty`` / ``ood_regularization_coeff`` arguments. Console output is controlled by ``verbose``.
"""
import os
from abc import abstractmethod

import einops
import numpy as np
import torch
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from jm2d_toy.models.denoiser import DenoiserMLP
from jm2d_toy.tasks.problem import TFGJointProblem, TFGProblem


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return x


class BaseSampler(DenoiserMLP):
    """Guided DDPM sampler for x ~ p(x) subject to a problem's objective/constraints (Fig. 9)."""

    def __init__(self,
                 model_path: str,
                 problem: TFGProblem,
                 t_start_guidance = None,
                 constraint_mode = 'rejection',
                 device='cuda',
                 verbose=False):
        self.verbose = verbose

        # load model
        self.model_path = model_path
        self.load_model(model_path, device=device)

        # load problem
        self._setup_problem(problem)

        # common parameters
        if t_start_guidance is None:
            self.t_start_guidance = self.noise_scheduler.timesteps[0]
        else:
            self.t_start_guidance = self.to_tensor(t_start_guidance)
        self.constraint_mode          = constraint_mode

        # guide-relevant histories
        self.guide_log_cache = dict()

        self.num_opt_samples = 512

    def load_model(self, model_path: str, device: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        _model = torch.load(model_path, map_location=device)
        (hidden_dim, ob_dim) = _model["nets"]['policy.noise_pred_net.net1.0.weight'].shape

        super().__init__(ob_dim, hidden_dim, device=device)
        self.deserialize(_model)
        self.to(device = device)

    def _setup_problem(self, problem: TFGProblem):
        """
        Setup the problem: checks if the guidance problem is compatible with the data distribution
        """
        assert problem.input_dim == self.ob_dim, f"Problem input dim {problem.input_dim} does not match model output dim {self.ob_dim}"

        self.problem = problem

    def ood_penalty_term(self, nets, naction, timestep):
        """
        OOD penalty term (history logging only)

        Args:
            nets (dict): The networks
            naction (torch.tensor): x_t The action (B1, B2, ..., Bn, A)
            timestep (int): The timestep

        Returns:
            objective (torch.tensor): The objective (B1, B2, ..., Bn)

        Large score is bad: https://arxiv.org/abs/2010.13132
        """
        score_t = self.score_fn_t_x(nets, t = timestep, x = naction) # (B1, B2, ..., Bn, A)
        score_t_norm = torch.linalg.norm(score_t, dim=-1) # (B1, B2, ..., Bn)
        score_penalty_term = torch.sqrt(score_t_norm) # (B1, B2, ..., Bn)

        return score_penalty_term

    def save_default_histories(self, nets, naction, timestep, histories):
        """
        Save the default histories (histories that does not need information of guidance)

        Args
            nets (dict): The networks
            naction (torch.tensor): The action (B, A)
            timestep (int): The timestep (int)
            histories (dict): The histories of the samples

        Returns:
            histories (dict): The updated histories
        """
        B = naction.shape[0]

        _naction           = naction
        _timesteps         = timestep.repeat(B).cpu().numpy()
        _clean_sample      = self.estimate_clean_sample(nets, naction, timestep) # (B, A)
        _clean_sample_variance = self.estimate_clean_sample_variance_tweedie(nets, naction, timestep) # (B, A)
        _clean_sample_lazy = self.estimate_clean_sample_lazy(naction, timestep) # (B, A)
        _scores            = self.score_fn_t_x(nets, t = timestep, x = naction) # (B, A)
        _problem_objective = self.problem.objective_function(_clean_sample) # (B,)
        _ood_penalty       = self.ood_penalty_term(nets, naction, timestep) # (B,)

        histories['timesteps'].append(to_numpy(_timesteps))
        # samples
        histories['noisy_samples'].append(to_numpy(_naction))
        histories['clean_samples_lazy'].append(to_numpy(_clean_sample_lazy))
        histories['clean_samples'].append(to_numpy(_clean_sample))
        histories['clean_samples_variance'].append(to_numpy(_clean_sample_variance))
        # data-driven score
        histories['scores'].append(to_numpy(_scores))
        # objective
        histories['problem_objective'].append(to_numpy(_problem_objective))
        histories['ood_penalty'].append(to_numpy(_ood_penalty))

        return histories

    @torch.no_grad()
    def get_action(self, batch_size=1, use_constraints=False, return_histories=False):
        """
        Run the guided reverse diffusion.

        Args:
            batch_size (int): number of samples
            use_constraints (bool): use the problem constraints in the guidance weights
            return_histories (bool): also return per-step histories (consumes extra RNG; see README)

        Returns:
            naction (np.ndarray): (B, A) samples
            histories (dict): (B, T, ...) arrays, only if return_histories
        """
        num_inference_timesteps = self.algo_config['ddpm_config']['num_inference_timesteps']

        # select network
        nets = self.nets
        if self.ema is not None:
            self.ema.copy_to(parameters = self._shadow_nets.parameters())
            nets = self._shadow_nets

        # initialize action from Guassian noise
        naction = randn_tensor((batch_size, self.ob_dim), device=self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        # Store diffusion path
        histories = dict(
            noisy_samples = [], clean_samples = [], timesteps = [], scores = [],
            problem_objective = [], ood_penalty = [], clean_samples_lazy = [], clean_samples_variance = [],
            guide_log = dict())

        for k in tqdm(self.noise_scheduler.timesteps, disable=not self.verbose):
            if return_histories:
                # Get x_0|k related histories
                histories = self.save_default_histories(nets, naction, k, histories)

            if k <= self.t_start_guidance:
                naction = self.guided_step(
                    nets=nets,
                    naction=naction,
                    timestep=k,
                    eta=0,
                    use_constraints=use_constraints,
                    verbose=self.verbose
                )
            else:
                naction = self.step_normal(
                    nets=nets,
                    naction=naction,
                    timestep=k
                )

            # Store current values
            if return_histories:
                for key in self.guide_log_cache.keys():
                    if key not in histories['guide_log'].keys():
                        histories['guide_log'][key] = []
                    histories['guide_log'][key].append(to_numpy(self.guide_log_cache[key]))

        naction = naction.squeeze(0).cpu().numpy()

        if return_histories:
            # All histories have the format of (B, T, ...)
            for key in histories.keys():
                if key == 'guide_log':
                    for subkey in histories['guide_log'].keys():
                        raw_data = np.array(histories['guide_log'][subkey])
                        histories['guide_log'][subkey] = raw_data.transpose(1, 0, *range(2, raw_data.ndim))
                else:
                    # Convert list of arrays to single array and transpose to (T, B) -> (B, T)
                    raw_data = np.array(histories[key])
                    histories[key] = raw_data.transpose(1, 0, *range(2, raw_data.ndim))
            return naction, histories
        else:
            return naction

    def _name(self):
        """
        Name of the sampler (console output)
        """
        return type(self).__name__

    def disp(self, msg):
        if self.verbose:
            print('\033[93m' + str(self._name()) + ': ' + msg + '\033[0m')

    def to_tensor(self, x):
        return torch.tensor(x, device=self.device)

    def effective_sample_size(self, weights, method="normalized"):
        """ Effective sample size

        Args
            weights (torch.tensor) (B1, B2, ..., Bn, N)
            method: "Kish" (1 / sum w^2) or "normalized" (Kish ESS divided by N, in [1/N, 1])

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

    @abstractmethod
    def guided_step(self, nets, naction, timestep, eta=0, verbose=True, use_constraints=False):
        """
        Guided step

        Args:
            nets (dict): The networks
            naction (torch.tensor): The action (B, A)
            timestep (int): The timestep
            eta (float): The eta value
            verbose (bool): Whether to print verbose output
            use_constraints (bool): Whether to use constraints
        """
        pass

    def assert_weights_normalized(self, weights, atol=1e-4):
        """
        Assert that the weights are normalized

        Args:
            weights (torch.tensor): The weights to check (B, N)
        """
        assert weights.ndim == 2, "Weights must be a 2D tensor"

        weight_close_to_one  = (weights.sum(dim=-1) - 1).abs() < atol
        weight_close_to_zero = (weights.sum(dim=-1)).abs() < atol

        assert torch.logical_or(weight_close_to_one, weight_close_to_zero).all(), "Weights are not normalized"

    def update_weights_from_constraints(self, weights, constraints, lag_coeff = 100):
        """
        Update the weights from the constraints

        Args:
            weights (torch.tensor): The weights (B, N)
            constraints (torch.tensor): The constraints (B, N), > 0 means infeasible

        Returns:
            weights (torch.tensor): The updated weights
        """
        assert weights.shape == constraints.shape, "Weights and constraints must have the same shape"

        if self.constraint_mode == 'penalty':
            # numerical stability
            constraints = constraints - constraints.min(dim=1, keepdim=True)[0]

            # update weights
            weights *= torch.nn.functional.softmax(-lag_coeff * constraints / self.temperature, dim=1)
            weights = weights / weights.sum(dim=1, keepdim=True)

        elif self.constraint_mode == 'rejection':
            # reject samples that violate constraints

            weights[constraints > 0] = 0.0
            valid_mask = (weights > 0).any(dim=1)
            weights[valid_mask] /= weights[valid_mask].sum(dim=1, keepdim=True)

        else:
            raise ValueError(f"Invalid constraint mode: {self.constraint_mode}")

        return weights


class BaseJointSampler(BaseSampler):
    """Guided DDPM sampler for (x, k) where x ~ p(x) and k is a free parameter (Fig. 2)."""

    def __init__(self,
                 model_path: str,
                 problem: TFGJointProblem,
                 t_start_guidance = None,
                 param_schedule = 'linear',
                 param_schedule_timestep = 100,
                 constraint_mode = 'rejection',
                 device='cuda',
                 verbose=False):

        super().__init__(model_path, problem, t_start_guidance=t_start_guidance,
                         constraint_mode=constraint_mode, device=device, verbose=verbose)

        # param scheduler
        self.param_scheduler = DDPMScheduler(
            num_train_timesteps=param_schedule_timestep,
            beta_schedule=param_schedule,
            clip_sample=False,
            prediction_type="epsilon"
        )

    @torch.no_grad()
    def get_action(self, batch_size=1, use_constraints=False, return_histories=False):
        num_inference_timesteps = self.algo_config['ddpm_config']['num_inference_timesteps']

        # select network
        nets = self.nets
        if self.ema is not None:
            self.ema.copy_to(parameters = self._shadow_nets.parameters())
            nets = self._shadow_nets

        # initialize action from Guassian noise
        naction = randn_tensor((batch_size, self.ob_dim), device=self.device)
        nparam  = randn_tensor((batch_size, self.problem.param_dim), device = self.device)

        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        # Store diffusion path
        histories = dict(
            noisy_samples = [], clean_samples = [], clean_params = [],
            timesteps = [], scores = [],
            problem_objective = [], ood_penalty = [], noisy_params = [])

        for k in tqdm(self.noise_scheduler.timesteps, disable=not self.verbose):
            if k <= self.t_start_guidance:
                naction, nparam = self.guided_step(
                    nets=nets,
                    naction=naction,
                    nparam=nparam,
                    timestep=k,
                    eta=0,
                    use_constraints=use_constraints,
                    verbose=self.verbose
                )
            else:
                naction = self.step_normal(
                    nets=nets,
                    naction=naction,
                    timestep=k
                )

            # Store current values
            if return_histories:
                prev_timestep = self.noise_scheduler.previous_timestep(k)
                if prev_timestep < 0:
                    prev_timestep = 0
                clean_sample = self.estimate_clean_sample(nets, naction, prev_timestep)
                scores = self.score_fn_t_x(nets, t = prev_timestep, x = naction)
                problem_objective = self.problem.objective_function(clean_sample, nparam)
                ood_penalty = self.ood_penalty_term(nets, naction, prev_timestep)
                nparam_noisy = nparam @ self.to_tensor((self.get_param_covariance(prev_timestep) ** 0.5))

                histories['noisy_samples'].append(naction.squeeze(0).cpu().numpy())
                histories['timesteps'].append(k.cpu().numpy())
                histories['clean_samples'].append(clean_sample.squeeze(0).cpu().numpy())
                histories['clean_params'].append(nparam.squeeze(0).cpu().numpy())
                histories['noisy_params'].append(nparam_noisy.squeeze(0).cpu().numpy())
                histories['scores'].append(scores.squeeze(0).cpu().numpy())
                histories['problem_objective'].append(problem_objective.squeeze(0).cpu().numpy())
                histories['ood_penalty'].append(ood_penalty.squeeze(0).cpu().numpy())

                for key in self.guide_log_cache.keys():
                    if key not in histories.keys():
                        histories[key] = []
                    histories[key].append(self.guide_log_cache[key].squeeze(0).cpu().numpy())

        naction = naction.squeeze(0).cpu().numpy()
        nparam  = nparam.squeeze(0).cpu().numpy()

        if return_histories:
            # (T, B, ...) lists; the public API transposes them to (B, T, ...)
            return naction, nparam, histories
        else:
            return naction, nparam

    @abstractmethod
    def guided_step(self, nets, naction, nparam, timestep, eta=0, verbose=True):
        """
        Guided step
        """
        pass

    def get_param_covariance(self, data_timestep):
        """
        Return the covariance matrix of the parameters at a given timestep

        Args:
            data_timestep (int): The timestep of the data

        Returns:
            cov_param (torch.tensor): The covariance matrix of the parameters
        """
        data2param_scale = self.noise_scheduler.num_train_timesteps // self.param_scheduler.num_train_timesteps
        param_timestep = data_timestep // data2param_scale

        param_alphas_cumprod = self.param_scheduler.alphas_cumprod[param_timestep]
        cov_param = (1/param_alphas_cumprod - 1) * torch.eye(self.problem.param_dim)

        return cov_param

    def sampling_based_optimization(self, action, nparam, N_iter=10, use_constraints=False):
        """Perform sampling-based optimization of parameters using MBD formula

        Args:
            action: Action tensor of shape (B, A)
            nparam: Parameter tensor of shape (B, P)
            N_iter: Number of optimization iterations

        Returns:
            final_param_samples: Optimized parameter tensor of shape (B, P)
        """
        # Input validation
        assert action.device == nparam.device, "Action and parameter tensors must be on same device"
        assert action.is_cuda == nparam.is_cuda, "Both tensors must be on same compute device (CPU/CUDA)"

        # Get dimensions
        B, A = action.shape
        B, P = nparam.shape
        N_K = self.num_opt_samples

        # Repeat actions for parameter samples
        action_repeated = einops.repeat(action, 'b a -> b n a', n=N_K)

        # Create covariance schedule for parameter sampling
        cov_param_schedule = [self.get_param_covariance(50*iter) for iter in range(N_iter)]
        cov_param_schedule = cov_param_schedule[::-1]

        # Initialize tracking of best feasible parameters
        final_param_samples = nparam.clone()
        if use_constraints:
            final_objectives = torch.full((B,), float('inf'), device=nparam.device)
        else:
            final_objectives = self.problem.objective_function(action, nparam)
        inf_tensor = torch.tensor(float('inf'), device=nparam.device)

        # Iterative optimization
        for cov_param in cov_param_schedule:
            # Sample parameter perturbations
            dparam_samples = torch.distributions.MultivariateNormal(
                loc=torch.zeros(P),
                covariance_matrix=cov_param
            ).sample((B, N_K)).to(nparam.device)

            # Generate perturbed parameter samples
            param_samples = nparam.unsqueeze(1).repeat(1, N_K, 1) + dparam_samples
            param_samples = torch.clip(param_samples, -1, 1)

            # Evaluate objectives
            objectives = self.problem.objective_function(action_repeated, param_samples)
            objectives_orig = objectives.clone()

            # Calculate importance weights
            objectives = objectives - objectives.min(dim=-1, keepdim=True)[0]  # For numerical stability
            weights = torch.nn.functional.softmax(-objectives / self.temperature, dim=-1)

            if use_constraints:
                # Evaluate constraints
                constraints = self.problem.constraint_function(
                    action_repeated.reshape(-1, A),
                    param_samples.reshape(-1, P)
                ).reshape(B, N_K)

                # Update weights based on constraints
                # NOTE: sets the mode permanently on this instance (legacy behaviour); the public
                # Sampler.sample() builds a fresh instance per call, so it never leaks across calls.
                self.constraint_mode = "penalty"
                weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=1000)

            self.assert_weights_normalized(weights)

            # Update parameters using weighted average
            nparam = (param_samples * weights.unsqueeze(-1)).sum(dim=-2)
            nparam = torch.clip(nparam, -1, 1)

            # whether to update the parameters or not
            if use_constraints:
                feasible_mask = (constraints <= 0).any(dim=-1)
                objectives_valid = torch.where(constraints <= 0, objectives_orig, inf_tensor)
                best_obj, best_indices = objectives_valid.min(dim=-1)

                better_mask = best_obj < final_objectives
                update_mask = torch.logical_and(better_mask, feasible_mask)
            else:
                best_obj, best_indices = objectives_orig.min(dim = -1)
                update_mask = (best_obj < final_objectives)

            # Track best feasible parameters
            final_objectives[update_mask] = best_obj[update_mask]
            final_param_samples[update_mask] = param_samples[torch.arange(B), best_indices][update_mask]

            # Log progress
            if self.verbose:
                valid_ratio = (final_objectives < inf_tensor).sum() / B
                self.disp(f"K: {nparam[0].detach().cpu().numpy().round(2)}, "
                         f"obj: {objectives_orig.mean():.2f}, "
                         f"valid_ratio: {valid_ratio:.2f}")

        return final_param_samples
