"""D3IL-Avoiding task: scenario geometry, constraint lists and the environment.

``precompute_constraints`` is moved from SafeRobomimic@origin/mpd-improved ``scripts/eval_paper.py``
(identical in the constraint construction to the um_dpcc runner, which builds only the lists it uses).
The scenario geometry is in ``configs/scenarios.yaml``.
"""
import copy
import os
from copy import copy as shallow_copy

import numpy as np
import yaml

import diffuser.utils as utils

from ..paths import CONFIG_DIR

SCENARIOS = ("top_left", "top_right", "both", "cluttered")   # the paper scenarios; configs/scenarios.yaml may add more


def load_scenarios(path=None):
    with open(path or os.path.join(CONFIG_DIR, "scenarios.yaml")) as f:
        return yaml.safe_load(f)


def precompute_constraints(constraint_types, polytopic_constraints, obstacle_constraints, trajectory_dim, act_obs_indices,
                     bounds, enlarge_constraints, enlarge_constraints_ndgs, exp, action_dim=0):
    """
    Precompute constraints for different variants.
    Args:
        constraint_types:
        polytopic_constraints:
        obstacle_constraints:
        trajectory_dim:
        act_obs_indices:
        bounds:
        enlarge_constraints:
        enlarge_constraints_ndgs:
        exp:
        action_dim: 0 drops the 'deriv' (dynamics) rows for the Avoiding task (mpd-improved runner);
            the um_dpcc runner passed the diffusion action_dim (2), which adds them.

    Returns:

    """
    constraint_list = []
    constraint_list_tightened = []
    constraint_list_tightened_ndgs = []
    # Halfspace constraints
    constraint_list_polytopic_not_tightened = []
    if 'halfspace' in constraint_types:
        for constraint in polytopic_constraints:
            constraint_list.append(
                ('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
            constraint_list_tightened.append(('ineq',
                                              utils.formulate_halfspace_constraints(constraint, enlarge_constraints,
                                                                                    trajectory_dim, act_obs_indices)))
            constraint_list_tightened_ndgs.append(('ineq', utils.formulate_halfspace_constraints(constraint,
                                                                                                 enlarge_constraints_ndgs,
                                                                                                 trajectory_dim,
                                                                                                 act_obs_indices)))
            constraint_list_polytopic_not_tightened.append(
                ('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))

    # Bounds
    if 'bounds' in constraint_types:
        lower_bound, upper_bound = utils.formulate_bounds_constraints(constraint_types, bounds, trajectory_dim,
                                                                      act_obs_indices)
        constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])
        constraint_list_tightened.extend([['lb', lower_bound], ['ub', upper_bound]])

    # Obstacle constraints
    if 'obstacles' in constraint_types:
        for constr in obstacle_constraints:
            constraint_list.append(
                [constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]],
                 constr['center'], constr['radius']])
            constraint_list_tightened.append(
                [constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]],
                 constr['center'], constr['radius'] + enlarge_constraints])
            constraint_list_tightened_ndgs.append(
                [constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]],
                 constr['center'], constr['radius'] + enlarge_constraints_ndgs])

    # Dynamics constraints
    constraint_list_without_prior = shallow_copy(constraint_list)
    constraint_list_without_prior_tightened = shallow_copy(constraint_list_tightened)

    dynamics_constraints = []
    if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices,
                                                                                                   action_dim)
    for constraint in dynamics_constraints:
        constraint_list.append(constraint)
        constraint_list_tightened.append(constraint)
        constraint_list_tightened_ndgs.append(constraint)

    constraint_list_dict = {
        'constraint_list': constraint_list,
        'constraint_list_tightened': constraint_list_tightened,
        'constraint_list_tightened_ndgs': constraint_list_tightened_ndgs,
        'constraint_list_without_prior': constraint_list_without_prior,
        'constraint_list_without_prior_tightened': constraint_list_without_prior_tightened,
        'constraint_list_polytopic_not_tightened': constraint_list_polytopic_not_tightened,
    }

    return constraint_list_dict


class AvoidingTask:
    """One constraint scenario of D3IL-Avoiding (``top_left``, ``top_right``, ``both``, ``cluttered`` or any other entry
    of ``configs/scenarios.yaml``)."""

    def __init__(self, scenario, scenarios_path=None):
        cfg = load_scenarios(scenarios_path)
        if scenario not in cfg["scenarios"]:
            raise KeyError(f"unknown scenario {scenario!r}; choose from {sorted(cfg['scenarios'])}")
        common = cfg["common"]
        spec = cfg["scenarios"][scenario]
        self.name = scenario
        self.legacy_name = spec.get("legacy_name", scenario)   # scenario name of the original code (metadata only)
        self.exp = common["experiment"]
        self.robot_name = self.exp.split('-')[0]
        self.constraint_types = list(common["constraint_types"])
        self.init_end_eff_pos = list(common["init_end_eff_pos"])
        self.dt = common["dt"]
        self.observation_indices = dict(common["observation_indices"])
        self.action_indices = dict(common["action_indices"])
        self.bounds = copy.deepcopy(common["bounds"])
        self.ax_limits = common["ax_limits"]
        self.max_episode_length = common["max_episode_length"]
        self.polytopic_constraints = copy.deepcopy(spec["halfspaces"])
        self.obstacle_constraints = copy.deepcopy(spec["obstacles"])

    def index_maps(self, diffusion):
        """(trajectory_dim, action_dim, act_obs_indices) of a GaussianDiffusion over (action, observation)."""
        trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
        action_dim = diffusion.action_dim
        obs_indices_updated = {key: val + action_dim for key, val in self.observation_indices.items()}
        act_obs_indices = {**self.action_indices, **obs_indices_updated}
        return trajectory_dim, action_dim, act_obs_indices

    def constraint_lists(self, diffusion, padding, dynamics_constraints):
        """All constraint lists of the original runner for this scenario (see ``precompute_constraints``)."""
        trajectory_dim, action_dim, act_obs_indices = self.index_maps(diffusion)
        return precompute_constraints(self.constraint_types, self.polytopic_constraints, self.obstacle_constraints,
                                      trajectory_dim, act_obs_indices, self.bounds, padding, padding, self.exp,
                                      action_dim=action_dim if dynamics_constraints else 0)

    def planner_constraints(self, diffusion, padding, dynamics_constraints):
        """Constraint list handed to the projector: padded ('tightened') if padding > 0, else nominal."""
        lists = self.constraint_lists(diffusion, padding, dynamics_constraints)
        return lists['constraint_list_tightened'] if padding > 0 else lists['constraint_list']

    def evaluation_constraints(self, diffusion):
        """Un-padded halfspaces (formulated) and obstacles used to count violations."""
        lists = self.constraint_lists(diffusion, 0, False)
        return lists['constraint_list_polytopic_not_tightened'], self.obstacle_constraints

    def xy_constraints(self):
        """The constraints the runner counts violations of (``evaluation_constraints``), in the (x, y) plane of the end
        effector: halfspace rows (cx, cy, d), violated iff cx x + cy y >= d (``formulate_halfspace_constraints``
        without padding), and discs (cx, cy, r), violated iff the distance to (cx, cy) is < r. For plotting."""
        halfspaces = []
        if 'halfspace' in self.constraint_types:
            for constraint in self.polytopic_constraints:
                c, d = utils.formulate_halfspace_constraints(constraint, 0, 2, {'x': 0, 'y': 1})
                halfspaces.append((c[0], c[1], d))
        discs = []
        if 'obstacles' in self.constraint_types:
            discs = [(*o['center'], o['radius']) for o in self.obstacle_constraints]
        return np.array(halfspaces, dtype=np.float64).reshape(-1, 3), np.array(discs, dtype=np.float64).reshape(-1, 3)

    def make_env(self):
        from d3il.environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
        from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
        # D3IL numbers robots with a process-global counter, but the Avoiding termination check
        # (mj_helper.has_collision) looks up the geom 'rod:geom_rb0'. Every env after the first one in a process
        # would therefore never terminate on an obstacle collision. The original runs created one env per process
        # (robot 0); resetting the counter gives every env created here that robot id.
        MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
        env = ObstacleAvoidanceEnv()
        env.start(override_init_pos=self.init_end_eff_pos)
        return env


def make(scenario, scenarios_path=None):
    return AvoidingTask(scenario, scenarios_path)
