"""Build planner + environments from a DP checkpoint and the experiment config.

``build`` is ``safediffusion_d4rl/pointmaze/utils.py:policy_and_env_from_checkpoint_and_config`` of the research code
(SafeRobomimic@robomimic 8b2371d2, :396-549) with the same call order (the order matters: building the networks
consumes the torch RNG and every ``SafeMazeEnv`` draws its gym seed from the numpy RNG). Changes are plumbing only:
the planner class and backup name are arguments, the padding is a number, gym spaces are seeded
(``jm2d_maze.seeding``), and the unused ``policy_kwargs`` / ``safety_kwargs`` options were removed.
"""
import json
from copy import deepcopy

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from jm2d_maze.backup import make_backup
from jm2d_maze.predictor import MazeEnvSimStatePredictor
from jm2d_maze.seeding import seed_gym_spaces
from jm2d_maze.tasks.maze import SafeMazeEnv, get_maze_spec, get_open_maze_spec


def build(planner_cls, ckpt_path, config, pad=0.0, backup="hq", seed_gym_spaces_=True, verbose=False,
          maze_layout=None):
    """Load the DP from ``ckpt_path``, build the padded maze and wrap the DP with ``planner_cls``.

    Args:
        planner_cls: a class from ``jm2d_maze.planners.REGISTRY``
        ckpt_path (str): robomimic DP checkpoint
        config: robomimic ``Config`` of the experiment (``jm2d_maze.config.load_config``)
        pad (float): test-time wall padding w >= 0
        backup (str): backup planner name (``hq``, ``xpyp``, ``xmym``)
        seed_gym_spaces_ (bool): seed gym's construction-time random action (``jm2d_maze.seeding``); False
            reproduces the non-deterministic research code
        maze_layout (str): maze layout instead of maze2d-large (``jm2d_maze.tasks.get_maze_spec``)

    Returns:
        planner, env_safe, eval_env_safe
    """
    if seed_gym_spaces_:
        seed_gym_spaces()
    env_kwargs = {"maze_spec": get_maze_spec(pad, maze_layout)}

    # load policy from `checkpoint` and update policy if different with provided `config``
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path = ckpt_path,
                                                         device    = device,
                                                         verbose   = verbose)

    config_policy = json.loads(ckpt_dict["config"])
    config_policy["algo"]["horizon"]["action_horizon"] = config.safety.filter.n_head
    config_policy["algo"]["ddpm"].update(config["algo"]["ddpm"])
    config_policy["algo"]["ddim"].update(config["algo"]["ddim"])
    ckpt_dict["config"] = json.dumps(config_policy, indent=4)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict = ckpt_dict,
                                                         device    = device,
                                                         verbose   = verbose)

    # maybe override the environment setting
    if env_kwargs is not None:
        for k, v in env_kwargs.items():
            ckpt_dict["env_metadata"]["env_kwargs"][k] = v

    # robomimic-style loading environment
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict,
                                           render           = False,
                                           render_offscreen = True,
                                           verbose          = verbose)

    eval_env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict,
                                           render           = False,
                                           render_offscreen = True,
                                           verbose          = verbose)

    # wrap the environment with the safety wrapper
    env_safe  = SafeMazeEnv(env, **config.safety)
    eval_env_safe = SafeMazeEnv(eval_env, **config.safety)
    dt_action = env_safe.unwrapped_env.sim.model.opt.timestep

    # wrap the policy that needs 1) state predictor and 2) backup_policy
    ckpt_dictCopy = deepcopy(ckpt_dict)

    maze_size_str = env.unwrapped.name.split("-")[1]
    ckpt_dictCopy["env_metadata"]["env_kwargs"]["maze_spec"] = get_open_maze_spec(maze_size_str)
    envCopy, _   = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dictCopy, render=False, render_offscreen=True, verbose=verbose)
    envCopy_safe = SafeMazeEnv(envCopy, **config.safety)
    predictor = MazeEnvSimStatePredictor(envCopy_safe)

    backup_policy = make_backup(backup, **config.safety)

    policy_wrapped = planner_cls(rollout_policy = policy,
                                 backup_policy  = backup_policy,
                                 dt_action      = dt_action,
                                 predictor      = predictor,
                                 **config.safety)

    return policy_wrapped, env_safe, eval_env_safe
