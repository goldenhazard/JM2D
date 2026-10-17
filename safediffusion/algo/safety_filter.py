"""
Receding-horizon safety filter (base class) and the DP / RAIL baselines.
"""
import abc
import time
from copy import deepcopy
from typing import Any
from collections import deque

import torch
import numpy as np
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import RolloutPolicy
from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.predictor import StatePredictor
from safediffusion.algo.planner_base import ParameterizedPlanner, FEASIBLE_FLAG
from safediffusion.utils.deco_utils import time_profile


class SafetyFilter(RolloutPolicy, abc.ABC):
    def __init__(self, 
                 rollout_policy: RolloutPolicy, 
                 backup_policy: ParameterizedPlanner,
                 predictor: StatePredictor,
                 dt_action: float,
                 dtype  = torch.float32,
                 device = torch.device("cpu"),
                **config):
        """Receding-horizon safety filter class

        Args:
            rollout_policy (RolloutPolicy): The rollout policy
            backup_policy (ParameterizedPlanner): The backup policy
            predictor (StatePredictor): The state prediction model
            dt_action (float): The delta timestep for the action
            dtype (_type_, optional): _description_. Defaults to torch.float32.
            device (_type_, optional): _description_. Defaults to torch.device("cpu").
        
        TODO
            1. Implement delayed RAIL safety filter (merge with `feat/delay` branch)
        """
        assert isinstance(rollout_policy, RolloutPolicy)
        assert isinstance(backup_policy, ParameterizedPlanner)
        assert isinstance(predictor, StatePredictor)

        self.rollout_policy = rollout_policy
        self.backup_policy  = backup_policy
        self.predictor      = predictor
        self.dt_action      = dt_action                     # action time step
        self.dtype          = dtype
        self.device         = device
        
        # performance policy-related
        self.rollout_policy_obs_keys  = self.rollout_policy.policy.global_config.all_obs_keys
        self.rollout_policy_goal_keys = [key for modality in self.rollout_policy.policy.obs_config.modalities.goal.values()
                                         for key in modality]
        
        # backup policy-related
        self.backup_policy_weight_keys = list(self.backup_policy.weight_dict.keys())
        
        # safety filter parameter
        self.n_head            = config["filter"]["n_head"]    # length of the head actions (each unit corresponds to action applied to the environment)
        self.max_init_attempts = config["filter"]["max_init_attempts"] # maximum number of attempts to initialize the backup plan
        self.verbose           = config["filter"]["verbose"]
        
        self.ddim_eta          = config["policy"]["ddim_eta"]

        # internal data structure & data
        self.nominal_actions = deque(maxlen=self.n_head)     # nominal actions
        self.nominal_plan    = None
        self._backup_plan    = None                          # backup plan (reference trajectory)
        self.backup_plan_to_execute = None                   # current backup plan to use
        
        self._action_idx     = 0                             # action index
        self.intervened      = False                         # whether the policy has intervened
        self.stuck           = False                         # robot is stuck - no feasible backup plan could be found

        self.delay_observation = config["test_time_perturb"]["delay_observation"]["enabled"]

    def update_policy_obs_queue(self, obs_dict):
        """
        Update the observation queue of the rollout policy
        """
        # Preprocess the observation dictionary compatible to the rollout policy
        obs_dict_copy = {k: obs_dict[k] for k in self.rollout_policy_obs_keys}   # (an unused deepcopy was removed)
        obs_dict_copy = self.rollout_policy._prepare_observation(obs_dict_copy)
        
        To = self.rollout_policy.policy.algo_config.horizon.observation_horizon
        
        for k in obs_dict_copy.keys():
            if len(obs_dict_copy[k].shape) == 3 and obs_dict[k].shape[-2] == To:
                # If the observation is already stacked, use the last one
                obs_dict_copy[k] = obs_dict_copy[k][:, -1, :]
            n_repeats = max(To - len(self.rollout_policy.policy.obs_queue), 1)
            self.rollout_policy.policy.obs_queue.extend([obs_dict_copy] * n_repeats)

    def __call__(self, obs_dict, goal_dict=None, **kwargs):
        """
        Main entry point of our receding-horizon safety filter.
              
        Args:
            obs_dict   (dict): observation dictionary
            goal_dict (dict): goal_dict dictionary
        
        Returns:
            ac (np.ndarray): action to take
            controller_to_use (str): controller to use
        """
        # If the robot is stopped, restart the policy
        if self.has_no_backup_plan():
            self.restart_and_setup_initial_backup_plan(obs_dict, goal_dict)
            self.update_action_idx(0)
            if self.stuck:
                return None, None
        
        self.update_policy_obs_queue(obs_dict)
            
        # Start generating next plan and the backup plan
        if self.action_idx % self.n_head == 0:
            # initialize action count
            self.update_action_idx(0)
            
            # compute next plan and backup plan
            plan, actions, head_plan_safety, backup_plan, opt_info = self.sample_next_plan_actions_verify_head_plan_and_sample_backup_plan(obs_dict, goal_dict)
            
            is_safe = (head_plan_safety) and (backup_plan is not None)
            
            # 1. If verified, set new backup plan
            if is_safe:
                self.clear_backup_plan()
                self.set_backup_plan(backup_plan)
                
            # 2. If not verified, use the backup plan
            else:
                horizon = self.n_head * self.dt_action
                backup_plan = self.pop_backup_plan(horizon)
                self.backup_plan_to_execute = backup_plan
            
            self.nominal_actions = actions
            self.nominal_plan    = plan
            self.intervened      = not is_safe
            self.opt_info        = opt_info
            
            k_backup             = self._backup_plan.get_trajectory_parameter().cpu().numpy()
            msg = f'Using Diffusion: {not self.intervened} | ' + \
                    f'Head Plan Safety: {head_plan_safety}  | ' + \
                    f'{len(self._backup_plan)} backup actions left: (k={np.round(k_backup, 2)})'
            self.disp(msg)
        
        ac, controller_to_use = self.get_action(obs_dict=obs_dict)
        self.update_action_idx(self.action_idx+1)
        
        return ac, controller_to_use
    
    def get_action(self, obs_dict):
        """
        Given the current observation and internal reference trajectory, get the 
        actions that can be executed to the environment

        Args
            obs_dict : observation dictionary
        
        Returns
            ac : action
            controller_to_use : controller to use
        """
        if self.intervened:
            # use backup policy
            action = self.backup_policy.get_action_from_plan_at_t(
                t_des     = (self.action_idx+1) * self.dt_action,
                plan      = self.backup_plan_to_execute,
                obs_dict  = obs_dict
            )
            controller_to_use = self.backup_controller_name
            
        else:
            # use nominal policy
            action = self.nominal_actions[self.action_idx]
            controller_to_use = self.rollout_controller_name
        
        return action, controller_to_use
    
    @property
    def action_idx(self):
        """ Getter for action idx
        """
        return self._action_idx
    
    def update_action_idx(self, new_value):
        """ Special function to update action_idx."""
        if not isinstance(new_value, int) or new_value < 0:
            raise ValueError("action_idx must be a non-negative integer.")
        self._action_idx = new_value

    @action_idx.setter
    def action_idx(self, value):
        raise AttributeError("Direct modification of action_idx is not allowed. Use update_action_idx()")
    
    @property
    @abc.abstractmethod
    def rollout_controller_name(self):
        """
        Return the controller name of the rollout policy
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def backup_controller_name(self):
        """
        Return the controller name of the rollout policy
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def sample_next_plan_actions_verify_head_plan_and_sample_backup_plan(self, obs_dict, goal_dict=None):
        """
        Sample the next plan and corresponding backup plan.
        
        Design of main sampling strategy

        Args
            obs_dict   : observation dictionary
            goal_dict : goal_dict dictionary
        
        Returns
            plan             : (ReferenceTrajectory)
            actions          : (T_p, D_a) np.ndarray
            head_plan_safety : (bool) safety of the head plan
            backup_plan      : (ReferenceTrajectory)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def safety_critical_state_keys(self):
        """
        Return the keys of the safety-critical states
        """
        raise NotImplementedError
    
    def check():
        """
        Check the compatibility of the performance policy, backup policy, observation, goal_dict dictionary

        TODO: PLEASE IMPLEMENT THIS FUNCTION
        """
        pass

    def process_opt_info(self, opt_info, solver_name):
        """
        Process the optimization information
        """
        if solver_name == "ipopt":
            if opt_info is not None:
                ret = {
                    "objective": opt_info["obj_val"],
                }
            else:
                ret = {
                    "objective": 1e4,
                }
        elif solver_name == "sampling":
            pass
        else:
            raise ValueError(f"Unknown solver name: {solver_name}")
        return ret
    
    @abc.abstractmethod
    def is_robot_moving(obs_dict):
        """
        Check whether the robot is moving or not
        
        Args
            obs_dict : observation dictionary
        
        Returns
            out : (bool) whether the robot is moving or not
        """
        raise NotImplementedError
    
    def restart_and_setup_initial_backup_plan(self, obs_dict, goal_dict=None):
        """
        Restart the policy from current observation - the robot has stopped from fail-safe maneuver. 
        
        Action
            - set action index to zero
            - set initial backup plan
        """
        self.disp("Robot has stopped. Restarting the policy from the current observation.")
        # assert not self.is_robot_moving(obs_dict), "Robot is not stopped. Please check the velocity."
        
        # find the backup plan extensively
        backup_plan     = self.extensive_backup_plan_search(obs_dict, goal_dict=goal_dict)
        
        # set up the backup plan if found
        if backup_plan is None:
            self.stuck = True
        else:
            self.set_backup_plan(backup_plan)
            self.stuck = False
        
    @time_profile
    def extensive_backup_plan_search(self, obs_dict, goal_dict=None):
        """
        Search the backup plan with extensive computational resources. This strategy is not feasible to run in real-time.

        Args
            obs_dict   : (dict) observation dictionary
            goal_dict : (dict) goal_dict dictionary
            
        Return
            out  : (ReferenceTrajectory) backup plan, None if the backup plan is not found
        """
        self.disp("Extensive backup plan search is running.")
        
        out = None
        
        ob_init, goal_init = self.process_dicts_for_backup_policy(obs_dict=obs_dict, goal_dict=goal_dict, plan=None)

        # set backup policy objective function null, trajectory optimization finds the feasible solution only
        weight_dict = {k: 0.0 for k in self.backup_policy_weight_keys}
        self.backup_policy.update_weight(weight_dict)
            
        # solves trajectory optimization multiple times to find the feasible solution
        for _ in range(self.max_init_attempts):
            backup_plan, info = self.backup_policy(obs_dict = ob_init, goal_dict = goal_init, random_initialization = True)
            if info["status"] in FEASIBLE_FLAG:
                out = backup_plan
                break
        
        # NOTE: Additional strategy is defined by inheriting this function
        
        return out

            
    @time_profile
    def monitor_and_compute_backup_plan(self, plan, obs_dict, goal_dict=None):
        """
        Monitor the safety of the head plan and compute the backup plan if necessary

        Args
            plan : (ReferenceTrajectory) The reference plan to monitor and compute the backup plan
        
        Returns
            info : dict containing the safety information
        """
        assert isinstance(plan, ReferenceTrajectory) or plan is None

        if plan is None:
            info = dict(head_plan = False, backup_plan = None)
            return info
        
        info = dict()
        # Checking 1st criteria
        head_plan = plan[:self.n_head+1]
        is_head_plan_safe = self.check_head_plan_safety(head_plan, obs_dict)
        info["head_plan"] = is_head_plan_safe

        # Checking 2nd criteria
        backup_plan = None
        opt_info = None
        if is_head_plan_safe:
            tail_plan = plan[self.n_head:]
            tail_plan.set_start_time(0)
            backup_plan, opt_info = self.compute_backup_plan(tail_plan, obs_dict, goal_dict)

        info["backup_plan"] = backup_plan
        info["opt_info"] = opt_info
        
        return info
    
    @time_profile
    def compute_backup_plan(self, plan, obs_dict, goal_dict = None):
        """ 
        Return the plan and optimization information by solving the backup trajectory optimization (TO).
        
        Args:
            plan: (ReferenceTrajectory): The reference plan to compute the backup plan
                - If `plan` is None, it computes the backup plan attached at the current state, parsed from `obs_dict`.
                - If `plan` is not None, it computes the backup plan attached at the initial state of the `plan`.
            obs_dict   (dict): observation dictionary
            goal_dict (dict): goal_dict dictionary
        
        Returns:
            backup_plan (ReferenceTrajectory): The backup plan
            info (dict): information dictionary for TO

        TODO: This function is very confusing right now. Change it later (e.g., decouple plan = None mode)
              Each function in this function is not right-away to notice.
        """
        ob_backup, goal_backup = self.process_dicts_for_backup_policy(obs_dict=obs_dict, goal_dict=goal_dict, plan=plan)

        self.update_backup_policy_weight()
        backup_plan, info = self.backup_policy(obs_dict=ob_backup, goal_dict=goal_backup)

        if info["status"] in FEASIBLE_FLAG:
            return backup_plan, info
        else:
            return None, info

    def disp(self, msg):
        print(f"\033[33m[{self.__class__.__name__}]:      {msg}\033[0m")

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Abstract class to override ------------------------------  #
    # ------------------------------------------------------------------------------------ #
    @abc.abstractmethod
    def obs_dict_to_state(self, obs_dict):
        """
        Convert the observation dictionary to the state representation

        Args
            obs_dict : observation dictionary
        
        Returns
            state : state representation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_backup_policy_weight(self):
        """
        Update the weight dictionary of the backup policy according to the internal status.
        This function reflects the strategy of the objective of the backup planner.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def process_dicts_for_backup_policy(self, obs_dict, goal_dict=None, plan=None):
        """
        Given the plan, observation dictionary, and goal_dict dictionary from the environment (Safety Wrapper)
        preprocess the data to be compatible with the backup policy

        Args
            plan : ReferenceTrajectory object, plan from the nominal policy
            obs_dict   : observation dictionary
            goal_dict : goal_dict dictionary
        
        Returns
            obs_dict : observation dictionary
            goal_dict: goal_dict dictionary

        NOTE: This function should be written based on 1) backup policy's input keys and 2) ReferenceTrajectory object
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_plan_from_nominal_policy(self, obs_dict, goal_dict=None):
        """
        Get the plan from the nominal policy

        Args
            obs_dict   : (B, 1) Observation array
            goal_dict : (B, 1) Goal array
        
        Returns
            plan   : (B, 1) ReferenceTrajectory array
            actions: (B, T_p, D_a) np.ndarray
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def check_head_plan_safety(self, plan, obs_dict):
        """
        Check the safety of the head plan

        Args
            plan : (B, 1) ReferenceTrajectory array
        
        Returns
            is_safe : (B, 1) np.ndarray
        """
        assert isinstance(plan, ReferenceTrajectory)
        assert isinstance(obs_dict, dict)
        assert len(plan) == self.n_head + 1, f"Head Plan should be of length n_head + 1 not {len(plan)}"

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Backup Plan Management ----------------------------------  #
    # ------------------------------------------------------------------------------------ #
    def has_no_backup_plan(self):
        """
        Check whether the backup plan is empty
        """
        return self._backup_plan is None or len(self._backup_plan) <= 1
    
    def clear_backup_plan(self):
        """
        Clear the backup plan
        """
        self._backup_plan = None
    
    def set_backup_plan(self, plan):
        """
        Set the backup plan
        """
        self._backup_plan = plan
    
    def pop_backup_plan(self, horizon):
        """
        Pop the backup plan amount of horizon
        """        
        # retrieve the plan
        plan = self._backup_plan[self._backup_plan.t_des <= horizon]
        
        # update the remaining backup plan
        self._backup_plan = self._backup_plan[self._backup_plan.t_des >= horizon]
        if len(self._backup_plan) > 0:
            self._backup_plan.set_start_time(0)

        return plan
    
    def get_plan_backup_trajparam(self):
        """
        Get the trajectory parameter of the backup plan
        """
        return self._backup_plan.get_trajectory_parameter()

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Compatibility with RolloutPolicy ------------------------- #
    # ------------------------------------------------------------------------------------ #
    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        # initialize the rollout policy
        self.rollout_policy.start_episode()

        # initialize the backup policy
        self.backup_policy.start_episode()

        # initialize the safety filter internal variables
        self.clear_backup_plan()
        self.backup_plan_to_execute = None
        self.nominal_actions        = None
        self.nominal_plan           = None
        self.intervened             = False
        self.stuck                  = False

    def _prepare_observation(self, obs_dict):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            obs_dict (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        return self.rollout_policy._prepare_observation(obs_dict)

    def __repr__(self):
        """Pretty print network description"""
        return self.rollout_policy.__repr__()
    
    def maybe_to_device_perf(self, tensor):
        perf_device = self.rollout_policy.policy.device
        if tensor.device != perf_device:
            tensor = TensorUtils.to_device(tensor, device = perf_device)
        
        return tensor
    
    def maybe_normalize_actions_perf(self, actions):
        """ If the rollout policy has action normalization stats, normalize the actions
        
        
        Args:
            actions (np.ndarray): actions to normalize
        
        Returns:
            actions (np.ndarray): normalized actions
        """
        ac_norm_stats = self.rollout_policy.action_normalization_stats
        
        if ac_norm_stats is not None:
            assert len(ac_norm_stats.keys()) == 1, "Only one action key is supported for now"
            
            ac_key        = list(ac_norm_stats.keys())[0]
            # ensure obs_normalization_stats are torch Tensors on proper device
            ac_norm_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(ac_norm_stats), self.device))
            # limit normalization to obs keys being used, in case environment includes extra keys
            actions_dict = {ac_key: actions}
            actions_dict = ObsUtils.normalize_dict(actions_dict, normalization_stats=ac_norm_stats)
            actions      = actions_dict[ac_key]
        
        return actions
    
    def maybe_unnormalize_actions_perf(self, actions):
        """ If the rollout policy has action normalization stats, unnormalize the actions
        
        Args:
            actions (np.ndarray): actions to unnormalize
        
        Returns:
            actions (np.ndarray): unnormalized actions
        """
        if self.rollout_policy.action_normalization_stats is not None:
            action_keys = self.rollout_policy.policy.global_config.train.action_keys
            action_shapes = {k:self.rollout_policy.action_normalization_stats[k]["offset"].shape[1:] 
                            for k in self.rollout_policy.action_normalization_stats}
            ac_dict = AcUtils.vector_to_action_dict(actions, action_shapes=action_shapes, action_keys=action_keys)
            ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.rollout_policy.action_normalization_stats)
            action_config = self.rollout_policy.policy.global_config.train.action_config
            for key, value in ac_dict.items():
                this_format = action_config[key].get('format', None)
                if this_format == 'rot_6d':
                    rot_6d = torch.from_numpy(value).unsqueeze(0)
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                    ac_dict[key] = rot
            actions = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
        
        return actions

class DPPolicy(SafetyFilter):
    """DP baseline: execute the diffusion-policy plan without filtering (was ``IdentityRolloutPolicy``)."""
    def sample_next_plan_actions_verify_head_plan_and_sample_backup_plan(self, obs_dict, goal_dict=None):
        """
        Sample the next plan and corresponding backup plan.

        Args
            obs_dict   : observation dictionary
            goal_dict : goal_dict dictionary
        
        Returns
            plan             : ReferenceTrajectory
            actions          : (T_p, D_a) np.ndarray
            head_plan_safety : (bool) safety of the head plan
            backup_plan      : ReferenceTrajectory
        """
        (plan, actions)  = self.get_plan_from_nominal_policy(obs_dict, goal_dict)
        head_plan_safety = True
        backup_plan      = ReferenceTrajectory(t_des = self.backup_policy.t_des,
                                               x_des = torch.zeros(self.backup_policy.t_des.shape[0], self.backup_policy.n_state))
        backup_plan.stamp_trajectory_parameter(torch.zeros(self.backup_policy.n_param,))

        opt_info = {
            "objective": 0.0,
        }
        
        return plan, actions, head_plan_safety, backup_plan, opt_info
    
class RAILPolicy(SafetyFilter):
    """
    RAIL baseline (was ``ExplicitSafetyFilter``): sample one DP plan, verify its head plan and solve the backup
    trajectory optimisation (IPOPT) at the end of the head plan; fall back to the previous backup plan otherwise.
    """
    def __init__(self, 
                 rollout_policy, 
                 backup_policy,
                 predictor,
                 dt_action,
                 dtype  = torch.float32,
                 device = torch.device("cpu"),
                **config):
        
        super().__init__(rollout_policy, backup_policy, predictor, dt_action, dtype, device, **config)
    
    @time_profile
    def sample_next_plan_actions_verify_head_plan_and_sample_backup_plan(self, obs_dict, goal_dict=None):
        """
        Sample the next plan and corresponding backup plan.

        Args
            obs_dict   : observation dictionary
            goal_dict : goal_dict dictionary
        
        Returns
            plan             : ReferenceTrajectory
            actions          : (T_p, D_a) np.ndarray
            head_plan_safety : (bool) safety of the head plan
            backup_plan      : ReferenceTrajectory
        """
        (plan, actions)  = self.get_plan_from_nominal_policy(obs_dict, goal_dict)
        info             = self.monitor_and_compute_backup_plan(plan, obs_dict, goal_dict)
        head_plan_safety = info["head_plan"]
        backup_plan      = info["backup_plan"]
        opt_info         = info["opt_info"]
        opt_info         = self.process_opt_info(opt_info, "ipopt")
        
        return plan, actions, head_plan_safety, backup_plan, opt_info
