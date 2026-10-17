"""MLP noise-prediction network and the DDPM wrapper shared by all samplers.

Moved from legacy ``dev/policies/base_policy.py`` (b649add0):
``SinusoidalPosEmb`` (L22), ``ScoreNet`` (L36) and ``DiffusionPolicyUNet`` (L110, renamed ``DenoiserMLP``:
it is an MLP, not a UNet). Network, scheduler and helper bodies are unchanged. Removed: the unused DDIM
branches, the posterior helpers no kept sampler calls, and the side effect that created
``trained_models/dataset/{model,plots}`` in the working directory (``log_dir`` is now only used for training).
"""
import copy
import math
import os
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm


class SinusoidalPosEmb(nn.Module):
    # follows diffusion_policy (Chi et al., MIT License); see NOTICE
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScoreNet(nn.Module):
    """Noise prediction network epsilon(x_t, t). Layer creation order is part of the RNG protocol."""

    def __init__(self, ob_dim, hidden_dim):
        super(ScoreNet, self).__init__()

        input_dim = ob_dim
        output_dim = ob_dim
        self.hidden_dim = hidden_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.net1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.net4 = nn.Sequential(
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

        self.net5 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.net6 = nn.Sequential(
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, output_dim)
        )

    def forward(self, sample, timestep):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML (from diffusers, Apache-2.0; see NOTICE)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # Layered transformations
        o1 = self.net1(sample)
        o2 = self.net2(torch.cat([o1, global_feature], dim=-1))
        o3 = self.net3(torch.cat([o2], dim=-1))
        o4 = self.net4(torch.cat([o3, global_feature], dim=-1))
        o5 = self.net5(torch.cat([o2, o4], dim=-1))

        return self.net6(torch.cat([o5, global_feature], dim=-1))


class DenoiserMLP(nn.Module):
    """DDPM (1000 train / 100 inference steps, sigmoid betas, clipped x0, epsilon prediction, EMA 0.99)."""

    def __init__(self, ob_dim, hidden_dim, device='cuda', log_dir=None):
        super(DenoiserMLP, self).__init__()
        self.ob_dim = ob_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.algo_config = {
            "ddpm_config": {
                "num_train_timesteps": 1000,
                "num_inference_timesteps": 100,
                "beta_schedule": "sigmoid",
                "clip_sample": True,
                "prediction_type": "epsilon"
            },
            "ema": True,
            "ema_power": 0.99
        }

        self._create_networks()
        self._create_optimizers()
        self.log_dir = log_dir
        if log_dir is not None:
            self._create_dirs(log_dir)

    def _create_dirs(self, log_dir):
        self.log_dir = log_dir
        self.model_dir = f"{log_dir}/model"
        self.plot_dir = f"{log_dir}/plots"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _create_optimizers(self):
        """
        Creates optimizers and places them into @self.optimizers.
        """
        policy_optimizer = optim.Adam(
            self.nets['policy'].parameters(),
            lr=1e-4
        )

        self.optimizers = {
            'policy_optimizer': policy_optimizer
        }

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        noise_pred_net = ScoreNet(
            ob_dim=self.ob_dim,
            hidden_dim=self.hidden_dim
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().to(self.device)

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.algo_config["ddpm_config"]["num_train_timesteps"],
            beta_schedule=self.algo_config["ddpm_config"]["beta_schedule"],
            clip_sample=self.algo_config["ddpm_config"]["clip_sample"],
            prediction_type=self.algo_config["ddpm_config"]["prediction_type"]
        )

        self.nets = nets
        self._shadow_nets = copy.deepcopy(self.nets).eval()
        self._shadow_nets.requires_grad_(False)
        self.noise_scheduler = noise_scheduler

        ema = None
        if self.algo_config["ema"]:
            ema = EMAModel(nets.parameters(), power=self.algo_config["ema_power"])

        self.ema = ema

    def fit(self, dataset, num_train_steps=100000, plot_fn: Callable = None):
        """Train the denoiser (was ``DiffusionPolicyUNet.train``). Requires ``log_dir``."""
        assert self.log_dir is not None, "fit() needs log_dir to save checkpoints"
        total_steps = 0
        epoch = 0
        pbar = tqdm(total=num_train_steps, desc="Training")

        loss_log = {
            "val": [],
            "step": []
        }

        while total_steps < num_train_steps:
            for batch in dataset:
                loss = self.train_on_batch(batch)
                self.optimizers['policy_optimizer'].zero_grad()
                loss.backward()
                self.optimizers['policy_optimizer'].step()

                loss_log["val"].append(loss.item())
                loss_log["step"].append(total_steps)

                if total_steps % 10000 == 0:
                    suffix = f"step{total_steps // 10000}e4"
                    torch.save(self.serialize(), f'{self.model_dir}/model_{suffix}.pth')

                    action = self.get_action(batch_size=100).cpu().numpy()
                    samples = action

                    if plot_fn is not None:
                        plot_fn(samples, f'{self.plot_dir}/samples_{suffix}.png')

                    plt.plot(loss_log["step"], loss_log["val"])
                    plt.savefig(f'{self.plot_dir}/loss_{suffix}.png')
                    plt.close()

                total_steps += 1
                pbar.update(1)

            pbar.set_postfix({'Loss': loss.item(), 'Epoch': epoch})
            pbar.refresh()
            epoch += 1

        torch.save(self.serialize(), f'{self.model_dir}/model.pth')

    def train_on_batch(self, batch):
        batch = batch.to(self.device).float()
        B = batch.shape[0]

        # sample noise to add to actions
        noise = torch.randn(batch.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # forward diffusion process
        noisy_actions = self.noise_scheduler.add_noise(
            batch, noise, timesteps)

        # predict the noise residual
        noise_pred = self.nets['policy']['noise_pred_net'](
            noisy_actions, timesteps)

        loss = F.mse_loss(noise_pred, noise)

        if self.ema is not None:
            self.ema.step(self.nets.parameters())

        return loss

    @torch.no_grad()
    def get_action(self, batch_size=100):
        """Unguided DDPM sampling (used to monitor training)."""
        sample_dim = self.ob_dim
        B = batch_size

        num_inference_timesteps = self.algo_config['ddpm_config']['num_inference_timesteps']

        # select network
        nets = self.nets
        if self.ema is not None:
            self.ema.copy_to(parameters = self._shadow_nets.parameters())
            nets = self._shadow_nets

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, sample_dim), device=self.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction,
                timestep=k,
            )

            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize()
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.load_state_dict(model_dict["ema"])
        else:
            self.ema = None

    def step_normal(self, nets, naction, timestep):
        r"""
        Reverse diffusion step x_{t} -> x_{t-1} (DDPM, with noise for t > 0).

        Args:
            nets: (torch.nn.Module) noise prediction model epsilon(x_{k}, k)
            naction: (torch.tensor) x_{step}
            timestep: (int) current timestep

        Returns:
            naction: (torch.tensor) x_{step-1}
        """
        # Predict noise
        noise_pred = nets["policy"]["noise_pred_net"](sample = naction, timestep = timestep)

        # reverse diffusion step (remove noise) and add_noise to generate x_{timestep-1}
        naction    = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=naction,
        ).prev_sample

        return naction

    ###########################################
    # Helper Functions related to the Diffusion
    ###########################################
    def get_sigma_t(self, noise_scheduler, timestep, eta = 0, return_prev_timestep = False):
        """
        Get the sigma_t for the timestep for realizations
        """
        if isinstance(noise_scheduler, DDPMScheduler):
            # following https://arxiv.org/pdf/2006.11239 eq (7)
            assert noise_scheduler.variance_type in ["fixed_small"]
            std_dev_t = noise_scheduler._get_variance(timestep, predicted_variance=None) ** (0.5)

            if return_prev_timestep:
                prev_timestep = noise_scheduler.previous_timestep(timestep)
                return std_dev_t, prev_timestep

        else:
            raise NotImplementedError

        return std_dev_t

    def estimate_clean_sample_lazy(self, sample, timestep):
        """
        Estimate the clean sample from the noisy sample lazy
        x_0 = (x_t ) / sqrt(alphacumprod_t)
        """
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        clean_sample_lazy = sample / (alpha_prod_t ** 0.5)

        return clean_sample_lazy

    def estimate_clean_sample(self, nets, sample, timestep):
        """
        Estimate the clean sample from the noisy sample and the noise prediction
        Tweedie's formula: x_0 = (x_t - sqrt{1-alphacumprod_t} * epsilon(x_t, t)) / sqrt(alphacumprod_t)

        Args:
            nets: Noise prediction networks
            sample: Noisy sample (B1, B2, ..., Bn, A)
            timestep: Timestep (int)

        Returns:
            clean_sample: Clean sample (B1, B2, ..., Bn, A)
        """
        sample_shape = sample.shape
        sample = sample.reshape(-1, self.ob_dim)

        assert timestep >= 0, "timestep must be non-negative"

        # get noise prediction
        noise_pred = nets["policy"]["noise_pred_net"](sample = sample, timestep = timestep)

        # estimate clean sample
        clean_sample = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=sample
        ).pred_original_sample

        # reshape to original shape
        clean_sample = clean_sample.reshape(sample_shape)

        return clean_sample

    def estimate_clean_sample_variance_tweedie(self, nets, sample, timestep):
        """
        Estimate the variance of the Tweedie formula

        Args:
            nets: Noise prediction networks
            sample: Noisy sample (B1, B2, ..., Bn, A)
            timestep: Timestep (int)

        Returns:
            clean_sample_variance: Clean sample variance (B1, B2, ..., Bn, A)
        """
        assert timestep >= 0, "timestep must be non-negative"

        # get noise prediction
        with torch.enable_grad():
            sample = sample.detach().clone().requires_grad_(True)

            def clean_sample_fn(x):
                return self.estimate_clean_sample(nets, x, timestep)

            jacobian = torch.func.jacrev(clean_sample_fn)(sample) # (B, A, B, A)
            gradient = jacobian.diagonal(dim1=0, dim2=2) # (A, A, B)
            gradient = gradient.permute(2, 0, 1) # (B, A, A)

        Sigma_t = 1 - self.noise_scheduler.alphas_cumprod[timestep]
        clean_sample_variance = Sigma_t * gradient.permute(0, 2, 1)

        return clean_sample_variance

    def score_fn_t_x(self, nets, t, x):
        """
        Score function of the form:
        s(x, t) = -epsilon(x, t) / sqrt(1-alpha_t)

        Args:
            nets: Noise prediction networks
            t: Current timestep (int)
            x: Current state (B1, B2, ..., Bn, A)

        Returns:
            score: Score function (B1, B2, ..., Bn, A)
        """
        assert t >= 0, "timestep must be non-negative"

        x_shape = x.shape

        # get noise prediction
        x = x.reshape(-1, self.ob_dim)
        noise_pred = nets['policy']['noise_pred_net'](sample=x, timestep=t)

        # compute score
        score = - noise_pred / torch.sqrt(1 - self.noise_scheduler.alphas_cumprod[t])

        # reshape to original shape
        score = score.reshape(x_shape)

        return score

    def get_noise(
        self,
        noisy_samples: torch.Tensor,
        original_samples: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """Noise that maps ``original_samples`` (x_0) to ``noisy_samples`` (x_t) under q(x_t | x_0).

        Adapted from diffusers ``DDPMScheduler.add_noise`` (Apache-2.0, see NOTICE), solved for the noise.
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.noise_scheduler.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        noise = (noisy_samples - sqrt_alpha_prod * original_samples) / sqrt_one_minus_alpha_prod
        return noise
