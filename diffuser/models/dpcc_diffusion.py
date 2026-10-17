"""Sampling loop of the DPCC code base as run by SafeRobomimic@origin/um_dpcc (a71ff97a) -- used by ``dpcc`` and ``jm2d``.

These are the um_dpcc ``GaussianDiffusion`` sampling methods (``diffuser/models/diffusion.py`` on that branch), moved
verbatim onto the ``GaussianDiffusion`` class of this repository (same constructor and buffers, so the same
checkpoint; built by ``diffuser.utils.load_diffusion(..., diffusion_class=DPCCDiffusion)``). Renamed:
``model_based`` -> ``use_jm2d``, ``K_value`` -> ``u_steps``, ``self.mbo`` -> ``self.jm2d``,
``p_mean_variance_model_based`` -> ``p_mean_variance_jm2d``.

Per denoising step (``p_sample_loop``):
  * ``use_jm2d=False`` ("Original code", planner ``dpcc``): plain DDPM step, then ``Projector.project`` (SLSQP) of the
    batch; ``costs[i]`` = the projection costs, used by ``trajectory_selection='minimum_projection_cost'``.
  * ``use_jm2d=True`` ("Our code", planner ``jm2d``): the noise estimate comes from ``JM2DSampler.optimize``;
    ``costs[i]`` = ``Projector.get_projection_costs`` (its cost term is 0, so the first plan is executed).

Clean-sample estimate of JM2D (``u_steps``, ``func_get_full_denoising``), exactly as in the original code:

    u_steps == 0     single-step Tweedie estimate  x0 = predict_start_from_noise(x_t, t, eps(x_t, t))
    u_steps >= t     full denoising from step t down to 0 (Table 2 "Ours": u_steps = n_timesteps)
    0 < u_steps < t  denoise with the plain DDPM loop from t until the loop index reaches
                     end = t - u_steps (the loop runs the steps t, t-1, ..., end, i.e. u_steps + 1 steps),
                     then a Tweedie estimate evaluated with timestep ``end``.
"""
import torch

from .diffusion import GaussianDiffusion
from .helpers import apply_conditioning
from .jm2d import JM2DSampler


class DPCCDiffusion(GaussianDiffusion):
    """GaussianDiffusion with the um_dpcc sampling loop (DPCC projection or JM2D). Same checkpoint as GaussianDiffusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jm2d = JM2DSampler()

    def set_jm2d_sampler(self, **sampler_kwargs):
        """Replace the JM2D sampler, e.g. ``set_jm2d_sampler(num_mc_samples=100, temperature=0.05)``."""
        self.jm2d = JM2DSampler(**sampler_kwargs)

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None, u_steps=0):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, None

    def p_mean_variance_jm2d(self, x, cond, t, returns=None, projector=None, constraints=None, u_steps=0):

        def func_get_full_denoising(cond, skip_t, curr_x, u_steps=u_steps, full_denoising=False):
            end_timestep = max(skip_t[0] - u_steps, 0) if not full_denoising else 0
            end_timestep = torch.ones_like(skip_t) * end_timestep
            if u_steps > 0:
                x_end, _ = self.conditional_sample(
                    cond=cond, current_sample=curr_x, skip_timestep=skip_t, end_timestep=end_timestep, return_diffusion=False, internal_loop=True
                )
                if end_timestep[0] > 0:
                    x_end = apply_conditioning(x_end, cond, self.action_dim, goal_dim=self.goal_dim)
                    noise = self.model(x_end, cond, end_timestep)
                    return self.predict_start_from_noise(x_end, t=end_timestep, noise=noise)
                else:
                    return x_end
            else:
                noise = self.model(curr_x, cond, skip_t)
                return self.predict_start_from_noise(curr_x, t=skip_t, noise=noise)

        info = None

        if projector is not None and t[0].item() > 0:
            # NOTE: Wonsuhk updated sigma_t to be correctly
            sigma_t = self.posterior_variance[t] ** 0.5
            epsilon = self.jm2d.optimize(
                self.model,
                x,
                cond,
                t,
                projector,
                constraints,
                func_get_full_denoising,
                self.predict_noise_from_start,
                sigma_t = sigma_t,
                inpainting_func=lambda x, c: apply_conditioning(x, c, self.action_dim, goal_dim=self.goal_dim),
            )
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, info

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None, use_jm2d=False, u_steps=0):
        info = None
        if use_jm2d:
            p_mean_var_fn = self.p_mean_variance_jm2d
        else:
            p_mean_var_fn = self.p_mean_variance
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, info = p_mean_var_fn(x=x, cond=cond, t=t, returns=returns, projector=projector, constraints=constraints, u_steps=u_steps)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, info

    @torch.no_grad()
    def p_sample_loop(self, shape, cond,
                      returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0,
                      skip_timestep=None, end_timestep=None, current_sample=None, use_jm2d=False, u_steps=0,
                      internal_loop=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if skip_timestep is not None and current_sample is not None:
            if self.goal_dim > 0:
                x[:,:,:-self.goal_dim] = current_sample
            else:
                x = current_sample

        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion:
            diffusion = [x]
            diffusion_log = []
        costs = {}

        all_infos = []

        # Denoising process
        last_timestep = -repeat_last if repeat_last > 0 and projector is not None else 0
        for i in reversed(range(last_timestep, self.n_timesteps)):
            t = i if i >= 0 else 0

            if skip_timestep is not None:
                if t > skip_timestep[0]:
                    continue

            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if projector is not None and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                x, info = self.p_sample(x, cond, timesteps, returns, projector=projector, constraints=constraints, use_jm2d=use_jm2d, u_steps=u_steps)
            else:
                x, info = self.p_sample(x, cond, timesteps, returns)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
            all_infos.append(info)

            projection_flag = torch.zeros(batch_size, device=device)
            if not use_jm2d:
                """
                Original code
                """
                if projector is not None and not projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
                    if self.goal_dim > 0:
                        x[:,:,:-self.goal_dim], projection_costs, projection_flag = projector.project(x[:,:,:-self.goal_dim], constraints, return_flag=True)
                        costs[i] = projection_costs
                    else:
                        x, projection_costs, projection_flag = projector.project(x, constraints, return_flag=True)
                        costs[i] = projection_costs
            else:
                """
                Our code
                """
                if projector is not None:
                    if self.goal_dim > 0:
                        projection_costs, _, _ = projector.get_projection_costs(x[:,:,:-self.goal_dim], constraints)
                        costs[i] = projection_costs
                    else:
                        projection_costs, _, _ = projector.get_projection_costs(x, constraints)
                        costs[i] = projection_costs

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

                log_iter = self.jm2d.log.copy()
                log_iter['projection_flag'] = projection_flag
                diffusion_log.append(log_iter)
                self.jm2d.log.clear()

            if end_timestep is not None:
                if t == end_timestep[0]:
                    break

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
            infos['diffusion_log'] = {}
            for key in diffusion_log[0].keys():
                infos['diffusion_log'][key] = torch.stack([log[key] for log in diffusion_log if key in log.keys()], dim=1)
        infos['projection_costs'] = costs
        infos['all_infos'] = all_infos

        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
