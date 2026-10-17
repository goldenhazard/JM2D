"""Train the Diffuser planner on the D3IL Avoiding demonstrations (one model per seed).

    python scripts/train.py --seeds 0 1 2 3 4

Moved from SafeRobomimic@origin/mpd-improved ``scripts/train.py`` (body unchanged; the GaussianInvDynDiffusion branch,
which this repository does not contain, is removed). Settings: ``config/avoiding-d3il.py`` (horizon 8, 20 diffusion
steps, UNet1DTemporalCondModel, 1e5 steps). Run from the repository root; checkpoints are written to
``logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion/<seed>/`` (point ``--ckpt-dir`` of scripts/run.py
there). Training is not seeded beyond ``torch.manual_seed(seed)``, so a retrained model differs from the released one.
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import jm2d_d3il  # noqa: E402,F401  (sys.path for third_party/d3il)
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
cli = ap.parse_args()
sys.argv = [sys.argv[0]]  # diffuser.utils.Parser parses sys.argv

exp = 'avoiding-d3il'

seeds = cli.seeds

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

for seed in seeds:
    args = Parser().parse_args(experiment='diffusion', seed=seed)

    # args.seed = seed
    torch.manual_seed(args.seed)

    # Get dataset
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        # returns_scale=args.returns_scale,
        returns_scale=args.max_path_length,               # Because the reward is <= 1 in each timestep
        # returns_scale=args.n_diffusion_steps,
        discount=args.discount,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    goal_dim = dataset.goal_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#

    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
        returns_condition=args.returns_condition,
        dim=args.dim,
        condition_dropout=args.condition_dropout,
        device=args.device,
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        goal_dim=dataset.goal_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
        device=args.device,
    )

    # Create trainer
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset)

    trainer.train()
