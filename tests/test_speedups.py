"""The release speed-ups (``jm2d_maze/backup/simple2d.py``, ``jm2d_maze/tasks/maze.py``) are bitwise exact.

(a) ``collision_check``: half-space normals computed once per obstacle instead of once per (waypoint, obstacle);
    compared with ``_collision_check_loop`` (the original code) on random head plans in the padded maze.
(b) ``_prepare_problem_data``: obstacle polytopes buffered by ``FO.Grest`` are memoised; compared with the original
    per-call computation, and ``Grest`` is checked to be the same for different start states / velocities.
(c) per-candidate observation copies share the (read-only) obstacle zonotopes instead of deep-copying them.

CPU only (MuJoCo + torch); no checkpoint needed. The end-to-end check is tests/test_equivalence.py.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="module")
def maze():
    import robomimic.utils.env_utils as EnvUtils
    from jm2d_maze.backup import make_backup
    from jm2d_maze.config import load_config
    from jm2d_maze.seeding import seed_gym_spaces, set_random_seed
    from jm2d_maze.tasks import SafeMazeEnv, get_maze_spec

    seed_gym_spaces()
    set_random_seed(0)
    config = load_config(planner="jm2d")
    env_meta = {"env_name": "maze2d-large-dense-v1", "type": 2,
                "env_kwargs": {"reset_target": True, "maze_spec": get_maze_spec(0.25)}}
    from robomimic.envs.wrappers import FrameStackWrapper
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    env = SafeMazeEnv(FrameStackWrapper(env, num_frames=2), **config.safety)   # as in the DP checkpoint config
    backup = make_backup("hq", **config.safety)
    return env, backup


def _random_plans(env, backup, n_states=12, n_plans=6, seed=0):
    """(obs_dict, head plan) pairs: random start states in the maze, random accelerations (some collide)."""
    from safediffusion.algo.plan import ReferenceTrajectory
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_states):
        obs = env.reset()
        obs["flat"] = obs["flat"][-1] if obs["flat"].ndim == 2 else obs["flat"]
        x0 = np.asarray(obs["flat"], dtype=np.float64)
        for _ in range(n_plans):
            T = 17
            acc = rng.normal(scale=rng.choice([2.0, 20.0, 60.0]), size=(T, 2))
            vel = x0[2:] + np.cumsum(acc, 0) * 0.01
            pos = x0[:2] + np.cumsum(vel, 0) * 0.01
            pos[0], vel[0] = x0[:2], x0[2:]
            plan = ReferenceTrajectory(torch.linspace(0, 0.16, T), pos, vel)
            out.append((obs, plan))
    return out


def test_collision_check_matches_loop(maze):
    env, backup = maze
    n_coll = 0
    cases = _random_plans(env, backup)
    for obs, plan in cases:
        robot = obs["zonotope"]["robot"][0].project([0, 1])
        obstacles = backup.get_obstacles_for_collision_check(obs)
        expected = backup._collision_check_loop(plan, robot, obstacles)
        assert backup.collision_check(plan, obs) == expected
        n_coll += int(expected)
    assert 0 < n_coll < len(cases), "test plans should include both safe and colliding ones"


def test_collision_offsets_bitwise(maze):
    """Every per-waypoint offset vector Pb equals the original zonotope polytope output."""
    from jm2d_maze.backup.simple2d import _zonotope_halfspaces_2d
    env, backup = maze
    for obs, plan in _random_plans(env, backup, n_states=3, n_plans=2, seed=1):
        robot = obs["zonotope"]["robot"][0].project([0, 1])
        centers = robot.center.unsqueeze(0).repeat(len(plan), 1)
        centers += plan.x_des
        centers -= plan.x_des[0]
        for obstacle in backup.get_obstacles_for_collision_check(obs):
            C, deltaD = _zonotope_halfspaces_2d(torch.vstack((obstacle.generators, robot.generators)))
            d = (obstacle.center - centers) @ C.T
            b_new = torch.cat((d + deltaD, -d + deltaD), dim=-1)
            for i in range(len(plan)):
                _, b_old = (obstacle - (robot + plan[i][1] - plan[0][1])).polytope(backup.combs)
                assert b_old.dtype == b_new.dtype and torch.equal(b_old, b_new[i])


def _same(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())


def test_obstacle_polytope_memo_matches_original(maze):
    import einops
    from safediffusion.armtdpy.reachability.conSet import batchZonotope
    env, backup = maze
    backup.update_weight(dict(goal=0.0, projection=0.0))
    grests = []
    for obs, plan in _random_plans(env, backup, n_states=4, n_plans=1, seed=2):
        ob, _ = backup.process_observation_and_goal_for_TO(obs_dict=dict(obs, flat=torch.tensor(obs["flat"])),
                                                           goal_dict=None)
        pdata = backup._prepare_problem_data(obs_dict=ob, goal_dict=None)
        FO = pdata["constraint"]["collision"]["FO"][0]
        grests.append(FO.Grest)
        A, b = pdata["constraint"]["collision"]["A"], pdata["constraint"]["collision"]["b"]
        n_interval = backup.FRS_info["n_reachable_set"]
        for i, obstacle in enumerate(backup.get_obstacles_for_collision_check(ob)):   # original computation
            obstacle = obstacle.project().deleteZerosGenerators()
            obs_Z = einops.repeat(obstacle.Z, 'm n -> repeat m n', repeat=n_interval)
            A_o, b_o = batchZonotope(torch.cat((obs_Z, FO.Grest), -2)).polytope(backup.combs)
            assert _same(A[0, i], A_o.cpu()) and _same(b[0, i], b_o.cpu())
    # the premise of the memo: Grest does not depend on the start state / velocity
    assert all(torch.equal(grests[0], g) for g in grests[1:])


def test_copy_obs_sharing_obstacles(maze):
    """(c) the per-candidate observation copy equals deepcopy except for the shared, read-only obstacle list."""
    from copy import deepcopy
    from jm2d_maze.tasks.maze import copy_obs_sharing_obstacles
    env, _ = maze
    obs = env.reset()
    ref, new = deepcopy(obs), copy_obs_sharing_obstacles(obs)
    assert new["zonotope"]["obstacle"] is obs["zonotope"]["obstacle"]
    assert new["zonotope"]["robot"] is not obs["zonotope"]["robot"]
    assert new["zonotope"]["robot"][0] is not obs["zonotope"]["robot"][0]
    assert sorted(new) == sorted(ref) and sorted(new["zonotope"]) == sorted(ref["zonotope"])
    for key in ref:
        if key != "zonotope":
            assert np.array_equal(np.asarray(new[key]), np.asarray(ref[key])), key
    for group in ref["zonotope"]:
        assert len(new["zonotope"][group]) == len(ref["zonotope"][group])
        for a, b in zip(new["zonotope"][group], ref["zonotope"][group]):
            assert torch.equal(a.Z, b.Z)
