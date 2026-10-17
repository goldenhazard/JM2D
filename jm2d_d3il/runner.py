"""Load a checkpoint and run D3IL-Avoiding episodes with a planner.

The episode loop and its metrics are moved from SafeRobomimic@origin/mpd-improved ``scripts/eval_paper.py``
(``main``, avoiding branch; identical to the loop of origin/um_dpcc ``scripts/eval_dev_Kfunc.py``). Removed: plotting,
the per-step trajectory / diffusion buffers (memory) and the position-tracking error (only printed for the unguided
planner). Per trial:

    torch.manual_seed(trial) -> env.reset() -> up to 200 steps of [count violations, plan, step the env]

Metrics per trial (keys of the original result JSON): ``n_success`` (goal reached), ``n_success_and_constraints``
(goal reached and no violation: Table 2 "SSucc"), ``n_steps``, ``n_violations`` (number of time steps that violate an
un-padded constraint: Table 2 "#Vio"), ``total_violations`` (summed violation depth), ``avg_time`` (s per planning step),
``collision_free_completed``.
"""
import os
import time

import numpy as np
import torch

import diffuser.utils as utils

TRIAL_KEYS = ("n_success", "n_success_and_constraints", "n_steps", "n_violations", "total_violations", "avg_time",
              "collision_free_completed")


def load_experiment(ckpt_dir, seed, diffusion_class=None, device='cuda', epoch='best'):
    """Checkpoint ``<ckpt_dir>/<seed>/state_<epoch>.pt`` + config pickles; returns diffuser's DiffusionExperiment.

    The dataset is rebuilt to obtain the normalizer, so the D3IL Avoiding data must be installed (README).
    """
    loadpath = os.path.join(ckpt_dir, str(seed))
    return utils.load_diffusion(loadpath, epoch=epoch, device=device, diffusion_class=diffusion_class)


def run_episodes(planner, task, env, trials, on_step=None, on_trial=None, max_episode_length=None, paths=None):
    """Run the given trial indices; returns {trial: {metric: value}}.

    ``on_step(trial, step, obs, action, samples)`` is called after every planning step (``obs`` = the observation
    the plan was conditioned on, ``samples`` = the planned batch), ``on_trial(trial, result)`` after every trial.
    If ``paths`` is a dict, ``paths[trial]`` receives the end-effector path of the trial (array (n_steps + 2, 2) of
    the observed (x, y): the start and the position after every environment step), set before ``on_trial`` is called.
    Recording only copies observations; it does not change the episode.
    """
    diffusion = planner.policy.model
    action_dim = diffusion.action_dim
    constraint_types = task.constraint_types
    obs_indices = task.observation_indices
    constraint_list_polytopic_not_tightened, obstacle_constraints = task.evaluation_constraints(diffusion)
    max_episode_length = max_episode_length or task.max_episode_length

    results = {}
    for iter in trials:
        n_success = 0
        n_success_and_constraints = 0
        n_steps = 0
        n_violations = 0
        total_violations = 0.0
        avg_time = 0.0
        collision_free_completed = 1

        torch.manual_seed(iter)

        # Reset environment
        obs = env.reset()
        action = env.robot_state()[:2]
        fixed_z = env.robot_state()[2:]
        obs = np.concatenate((action[:2], obs))
        if paths is not None:
            path = [obs[[obs_indices['x'], obs_indices['y']]]]

        disable_projection = False
        for step in range(max_episode_length):
            # Check if a safety constraint is violated
            violated_this_timestep = 0
            if 'halfspace' in constraint_types:
                for constraint in constraint_list_polytopic_not_tightened:
                    if constraint[0] == 'ineq':
                        c, d = constraint[1]
                        obs_to_check = obs[:-diffusion.goal_dim] if diffusion.goal_dim > 0 else obs
                        if obs_to_check @ c[action_dim:] >= d:
                            violated_this_timestep = 1
                            total_violations += obs_to_check @ c[action_dim:] - d
                            collision_free_completed = 0

            if 'obstacles' in constraint_types:
                for constraint in obstacle_constraints:
                    if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                        violated_this_timestep = 1
                        total_violations += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                        collision_free_completed = 0

            n_violations += violated_this_timestep

            # Calculate action
            obs_in = obs
            start = time.time()
            action, samples, diffusion_trajectories, diffusion_logs = planner(obs)
            avg_time += time.time() - start
            if on_step is not None:
                on_step(iter, step, obs_in, action, samples)

            # Step environment
            next_pos_des = action + obs[:2]
            obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
            success = info[1]
            obs = np.concatenate((next_pos_des[:2], obs))
            if paths is not None:
                path.append(obs[[obs_indices['x'], obs_indices['y']]])

            if success: n_success = 1
            if (terminated or step == max_episode_length - 1) and (not success): collision_free_completed = 0

            if success or terminated or step == max_episode_length - 1:
                n_steps = step
                avg_time /= step
                if success and collision_free_completed: n_success_and_constraints = 1
                break

        result = dict(n_success=float(n_success), n_success_and_constraints=float(n_success_and_constraints),
                      n_steps=float(n_steps), n_violations=float(n_violations),
                      total_violations=float(total_violations), avg_time=float(avg_time),
                      collision_free_completed=float(collision_free_completed))
        results[iter] = result
        if paths is not None:
            paths[iter] = np.stack(path).astype(np.float64)
        if on_trial is not None:
            on_trial(iter, result)
    return results
