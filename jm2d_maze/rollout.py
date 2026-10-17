"""Closed-loop PointMaze rollouts (Fig. 4 / Table 1 protocol).

``rollout`` and ``rollout_seeds`` are ``safediffusion_d4rl/pointmaze/utils.py:rollout`` / ``rollout_random_seed`` of
the research code (SafeRobomimic@robomimic 8b2371d2, :134-393). The control loop, the per-episode seeding and the
statistics are unchanged. I/O is now optional: no video unless ``render_mode`` is given, no per-seed
``stats.json``/``rollout_results.pkl`` files (``scripts/run.py`` writes one JSONL record per episode instead).

Statistics per episode (same keys as the stored paper runs):
    Success       reached the goal (|p - goal| <= 0.2) before ``horizon`` steps
    Horizon       number of control steps (dt = 0.01 s)
    Unsafe        number of steps in collision with a wall
    Intervention  number of replanning cycles (every n_head = 16 steps) at which the filter used the backup plan
    Plan          number of replanning cycles
    Stuck         no backup plan could be found more than 5 times
    Time          mean wall time of the replanning steps (s)
"""
import os
from time import time

import imageio
import numpy as np

from robomimic.algo.algo import RolloutPolicy

from safediffusion.envs.env_safety import SafetyEnv
from safediffusion.utils.deco_utils import time_profile
from safediffusion.utils.rand_utils import set_random_seed


def _is_stat_valid(stat):
    required_keys = ["Success", "Unsafe", "Intervention", "Horizon", "Plan"]

    if stat is None:
        return False

    is_valid = True
    for k in required_keys:
        if k not in stat.keys():
            is_valid = False

    return is_valid


def compute_aggregate_stats(stats):
    total_runs = len(stats)
    total_success = 0
    total_unsafe = 0
    total_intervention = 0
    total_horizon = 0
    total_safe_and_success = 0
    total_plan = 0

    for stat in stats.values():
        if _is_stat_valid(stat):
            total_success += int(stat['Success'])
            total_unsafe += stat['Unsafe']
            total_intervention += stat['Intervention']
            total_horizon += stat['Horizon']
            total_safe_and_success += (stat["Success"] and (stat["Unsafe"] == 0))
            total_plan += stat["Plan"]
        else:
            print("Given stat is not valid.")

    success_rate = total_success / total_runs
    unsafe_rate = total_unsafe / total_horizon if total_horizon > 0 else 0
    intervention_rate = total_intervention / total_plan if total_plan > 0 else 0
    success_and_safe_rate = total_safe_and_success / total_runs
    total_plan_length = total_plan

    return {
        'success_rate': success_rate,
        'unsafe_rate': unsafe_rate,
        'intervention_rate': intervention_rate,
        'total_runs': total_runs,
        'average_horizon': total_horizon / total_runs,
        'success_and_safe_rate': success_and_safe_rate,
        'total_plan_length': total_plan_length,
        'total_intervention': total_intervention,
    }


def rollout(policy,
            env,
            eval_env,
            horizon,
            x0,
            target_pos,
            render_mode = None,
            video_path = None,
            video_skip = 5):
    """
    Rollout the policy in the environment.

    Args:
        policy (Policy): policy to rollout
        env (Env): environment to rollout
        horizon (int): maximum rollout horizon
        x0 (np.ndarray): initial state (world frame)
        target_pos (np.ndarray): target position (world frame)
        render_mode (str): None (no rendering) or one of ["zonotope", "rgb_array", "janner"]
        video_path (str): mp4 path, required when ``render_mode`` is given

    Returns:
        stats (dict): see module docstring
        traj (dict): executed actions, states, interventions, safety flags, per-plan objective/inference time
    """
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy)

    obs  = env.reset() # dummyentry point
    obs  = env.set_state(pos = x0[:2], vel = np.array([0, 0]))
    goal = env.set_goal(pos = target_pos)

    policy.start_episode()

    video_writer = None
    if render_mode is not None:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    # logging variables
    num_intervention = 0
    num_unsafe = 0
    num_backupplan_unsafe = 0
    num_headplan_unsafe = 0
    num_stuck = 0
    num_plan  = 0
    is_stuck = False

    traj = dict(executed_actions = [],
                is_safe = [],
                states = [],
                interventions = [],
                inference_times = [],
                is_stuck=[],
                objective=[])

    for step_i in range(horizon):
        if step_i % 10 == 0:
            print("====================================================================")
            print(f"Step {step_i}:  State {np.round(obs['flat'][-1], 2)}, Goal {np.round(goal['flat'][:2], 2)}")
            print("====================================================================")

        t_plan = time()
        act, _ = policy(obs_dict=obs, goal_dict=goal)
        t_plan = time() - t_plan

        if policy.stuck:
            break

        next_obs, r, _, _ = env.step(act)
        success = env.is_success()["task"]
        is_safe = next_obs.pop("safe")
        is_intervened = policy.intervened

        # Render
        if video_writer is not None and step_i % video_skip == 0:
            nominal_plan_vis = None
            if policy.nominal_plan is not None:
                nominal_plan_vis = policy.nominal_plan.x_des + env.robotqpos_to_worldpos

            backup_plan_vis = None
            backup_FRS_vis  = None
            if policy._backup_plan is not None:
                backup_plan_vis = policy._backup_plan.x_des + env.robotqpos_to_worldpos
                backup_FRS_vis  = policy.backup_policy.FRS

            img = env.render(mode        = render_mode,
                             height      = 512,
                             width       = 512,
                             plan        = nominal_plan_vis,
                             backup_plan = backup_plan_vis,
                             intervened  = policy.intervened,
                             FRS         = backup_FRS_vis,
                             goal        = [env.goal_zonotope])

            video_writer.append_data(img)

        if policy.action_idx == 1:
            # Log per plan not step
            num_plan += 1
            if is_intervened:
                num_intervention += 1

            traj["objective"].append(policy.opt_info["objective"])
            traj["inference_times"].append(t_plan)

        # termination conditions
        if policy.stuck:
            print("Unable to compute the backup plan.")
            num_stuck += 1

            if num_stuck > 5:
                is_stuck = True
                break
            else:
                continue

        if success:
            print("Success!")
            break

        if not is_safe:
            num_unsafe += 1
            print("Safety violation!")

        if not is_safe:
            if is_intervened:
                num_backupplan_unsafe += 1

            else:
                num_headplan_unsafe += 1

        obs = next_obs

        # Log variables
        traj["executed_actions"].append(act)
        traj["states"].append(obs["flat"])
        traj["interventions"].append(is_intervened)
        traj["is_safe"].append(is_safe)
        traj["is_stuck"].append(is_stuck)

    if video_writer is not None:
        video_writer.close()

    inference_time = np.mean(traj["inference_times"])
    stats = dict(
                 Success      = success,
                 Horizon      = (step_i + 1),
                 Unsafe       = num_unsafe,
                 Intervention = num_intervention,
                 Plan         = num_plan,
                 Stuck        = is_stuck,
                 Time         = inference_time,
                 SafetyDetail = dict(HeadPlan   = num_headplan_unsafe,
                                     BackupPlan = num_backupplan_unsafe))

    return stats, traj


@time_profile
def rollout_seeds(policy,
                  env,
                  eval_env,
                  horizon,
                  seeds,
                  render_mode = None,
                  video_dir = None,
                  on_episode = None):
    """
    Simulate the policy in environment with the random seeds (legacy ``rollout_random_seed``).

    Episodes must run in order: the start/goal of env seed ``s`` come from the environment's own RNG, which is
    seeded once at construction and advanced by every reset, so seed ``s`` reproduces the stored runs only
    after seeds ``0 .. s-1`` (as in the paper runs, which always used seeds 0..99).

    Args:
        policy (Policy): policy to rollout
        env (Env): environment to rollout
        horizon (int): maximum rollout horizon
        seeds (list): list of random seeds
        render_mode (str): None or a render mode (writes ``<video_dir>/seed<k>.mp4``)
        on_episode (callable): called as ``on_episode(seed, stat, traj)`` after every episode

    Returns:
        stats (dict): per-seed stats (keys ``str(seed)``) plus the aggregate of ``compute_aggregate_stats``
    """
    stats = {}

    for i, seed in enumerate(seeds):
        set_random_seed(seed, deterministic_torch=False)

        obs  = env.reset()
        goal = env.get_goal()

        # translate it to the world frame
        x0         = obs["flat"][-1, :]
        x0[:2]    += env.robotqpos_to_worldpos
        target_pos = goal["flat"][:2] + env.robotqpos_to_worldpos

        video_path = None if render_mode is None else os.path.join(video_dir, f"seed{seed}.mp4")
        stat, traj  = rollout(policy      = policy,
                              env         = env,
                              eval_env    = eval_env,
                              horizon     = horizon,
                              x0          = x0,
                              target_pos  = target_pos,
                              render_mode = render_mode,
                              video_path  = video_path)

        print(f"Seed {i+1}/{len(seeds)}: {stat}")

        stats[str(seed)] = stat
        if on_episode is not None:
            on_episode(seed, stat, traj)

    summary_stats = compute_aggregate_stats(stats)
    stats.update(summary_stats)

    return stats
