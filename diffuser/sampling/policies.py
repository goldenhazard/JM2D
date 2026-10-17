from collections import namedtuple
import torch
import time
import einops
import numpy as np
import diffuser.utils as utils
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.arrays import to_device
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:
    """Receding-horizon planner around a GaussianDiffusion model (mpd-improved loop: SafeDiffuser, MPD).

    ``projector`` holds the constraints; ``invariance_type`` ('ReS' / 'TVS' / 'RoS') selects the SafeDiffuser
    invariance module; ``sampler_kwargs`` are passed to the sampling loop (MPD guide-step settings).
    """

    def __init__(self, model, normalizer, scheduler=None, preprocess_fns=[], test_ret=0, projector=None, 
                 trajectory_selection='random', invariance_type=None, **sampler_kwargs):
        self.model = model
        self.scheduler = scheduler,   # 'DDPM' or 'DDIM'
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.test_ret = test_ret
        self.invariance_type = invariance_type
        self.sampler_kwargs = sampler_kwargs

        # Inverse dynamics model
        if self.model.__class__.__name__ == 'GaussianInvDynDiffusion':
            self.inverse_dynamics = True
            self.inv_model = self.model.inv_model
            self.action_dim = 0
        else:
            self.inverse_dynamics = False

        # Projector
        self.projector = projector

        # Trajectory selection
        self.trajectory_selection = trajectory_selection        # 'random' or 'temporal_consistency' or 'minimum_projection_cost'

        # Previous observations
        self.prev_observations = None

        # Set invariance module
        self.model.set_invariance_module(self.invariance_type)

    def _sample(self, conditions, returns, projector, constraints, horizon):
        use_invariance = self.invariance_type is not None
        return self.model(
            conditions, 
            returns=returns, 
            projector=projector, 
            constraints=constraints, 
            horizon=horizon, 
            invariance=use_invariance,
            return_diffusion=True,
            **self.sampler_kwargs,
        )

    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False, return_diffusion=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')

        # Use GaussianDiffusion model with DDPM
        projector = self.projector if not disable_projection else None
        samples, infos = self._sample(conditions, returns, projector, constraints, horizon)

        trajectories = utils.to_np(samples)

        ## extract observations [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        if 'diffusion' in infos:
            diffusion_trajectories = utils.to_np(infos['diffusion'])         # Shape: batch_size x T x horizon x transition_dim     
            diffusion_observations = self.normalizer.unnormalize(diffusion_trajectories[:, :, :, self.action_dim:], 'observations')
            diffusion_actions = self.normalizer.unnormalize(diffusion_trajectories[:, :, :, :self.action_dim], 'actions')

        if 'diffusion_log' in infos:
            diffusion_logs = infos['diffusion_log']
            for key in infos['diffusion_log'].keys():
                diffusion_logs[key] = utils.to_np(infos['diffusion_log'][key])
            
            if 'clean_samples' in diffusion_logs.keys():
                diffusion_logs["clean_trajectory"] = self.normalizer.unnormalize(
                    infos['diffusion_log']['clean_samples'][..., self.action_dim:], 'observations'
                )
                diffusion_logs["clean_actions"] = self.normalizer.unnormalize(
                    infos['diffusion_log']['clean_samples'][..., :self.action_dim], 'actions'
                )
        
        # Sort according to similarity with previous observations
        if self.trajectory_selection == 'temporal_consistency' and not disable_projection and self.prev_observations is not None:   # Temporal consistency
            order = np.argsort(np.linalg.norm(observations[:,:-1,:] - self.prev_observations[:,1:,:], axis=(1,2)))
            which_trajectory = order[0]
            observations = observations[order]
        elif self.trajectory_selection == 'minimum_projection_cost' and not disable_projection:                                     # Minimum projection cost
            costs_total = np.zeros(batch_size)
            for timestep, cost in infos['projection_costs'].items():
                costs_total += cost
            which_trajectory = np.argmin(costs_total)
        else:                                                                                                                       # Random selection
            which_trajectory = 0
        self.prev_observations = np.repeat(np.expand_dims(observations[0], axis=0), batch_size, axis=0)

        ## Extract or calculate action
        if self.inverse_dynamics:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*samples.shape[-1])
            actions = self.inv_model(obs_comb)
            actions = utils.to_np(actions)
            actions = self.normalizer.unnormalize(actions, 'actions')
            action = actions[which_trajectory]
        else:
            ## extract action [ batch_size x horizon x action_dim ]
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[which_trajectory, 0]

        trajectories = Trajectories(actions, observations)

        if return_diffusion:
            diffusion_trajectories = Trajectories(diffusion_actions, diffusion_observations)

            return action, trajectories, diffusion_trajectories, diffusion_logs
        
        else:
            return action, trajectories

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions


class DPCCPolicy(Policy):
    """Policy for ``DPCCDiffusion`` (was ``Policy(model_based=..., K_value=...)`` on origin/um_dpcc).

    ``use_jm2d=False``: DPCC (per-step projection); ``use_jm2d=True``: JM2D, with ``u_steps`` denoising steps for the
    clean-sample estimate (``K_value`` in the original code); see ``diffuser/models/dpcc_diffusion.py``.
    """

    def __init__(self, model, normalizer, scheduler=None, preprocess_fns=[], test_ret=0, projector=None, 
                 trajectory_selection='random', use_jm2d=False, u_steps=0, **sampler_kwargs):
        super().__init__(model, normalizer, scheduler=scheduler, preprocess_fns=preprocess_fns, test_ret=test_ret,
                         projector=projector, trajectory_selection=trajectory_selection, **sampler_kwargs)
        self.use_jm2d = use_jm2d
        self.u_steps = u_steps

    def _sample(self, conditions, returns, projector, constraints, horizon):
        return self.model(
            conditions, 
            returns=returns, 
            projector=projector, 
            constraints=constraints, 
            horizon=horizon, 
            use_jm2d=self.use_jm2d,
            u_steps=self.u_steps,
            return_diffusion=True,
            **self.sampler_kwargs,
        )
