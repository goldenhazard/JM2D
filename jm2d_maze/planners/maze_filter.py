"""PointMaze specialisation of the receding-horizon safety filter.

``SafetyFilterMaze`` was ``safediffusion_d4rl/algo/safety_filter_maze.py:SafetyFilterMaze`` (SafeRobomimic@robomimic,
8b2371d2); unchanged. The planners (``dp``, ``rail``, ``jm2d``, ``gibbs``) combine it with a generic filter.
"""
from copy import deepcopy

import torch
import numpy as np
import einops

from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.safety_filter import SafetyFilter
from safediffusion.utils.deco_utils import time_profile
from jm2d_maze.tasks.maze import copy_obs_sharing_obstacles

class SafetyFilterMaze(SafetyFilter):
    @property
    def safety_critical_state_keys(self):
        return ["flat"]
    
    @property
    def rollout_controller_name(self):
        return "Torque"
    
    @property
    def backup_controller_name(self):
        return "Torque"
    
    def obs_dict_to_state(self, obs_dict):
        return obs_dict["flat"][-1]
    
    def is_robot_moving(self, obs_dict):
        vel = self.obs_dict_to_state(obs_dict)[2:]
        return np.linalg.norm(vel) > 1e-1
    
    def update_backup_policy_weight(self):
        """
        Update the weight of the backup policy
        """
        weight_dict = dict(goal = 0.0, projection = 1.0)
        self.backup_policy.update_weight(weight_dict)
    
    @time_profile
    def get_plan_from_nominal_policy(self, obs_dict, goal_dict=None, policy_env=None, num_samples=1):
        """
        Coupling diffusion policy with guiding of the safety filter

        Returns:
            plan
            actions
            info: dict containing head_plan and backup_plan
        
        NOTE: Technically, the backup plan is computed along with this procedure.
        However, we compute again since we are going to replace the trajopt with some
        other neural network later.
        """
        # Preprocess the observation dictionary compatible to the rollout policy
        obs_perf = {k: obs_dict[k] for k in self.rollout_policy_obs_keys}
        obs_perf = self.rollout_policy._prepare_observation(obs_perf)
        if goal_dict is not None:
            goal_perf = {k: goal_dict[k] for k in self.rollout_policy_obs_keys}
            goal_perf = self.rollout_policy._prepare_observation(goal_perf)

        # Compute the multiple trajectories with the policy
        actions_candidates = self.rollout_policy.policy.get_action_predictions(obs_perf, 
                                                                               goal_perf, 
                                                                               num_trajectories=num_samples, 
                                                                               eta=self.ddim_eta,
                                                                               overwrite_obs_queue=(not self.delay_observation))
        # actions_candidates = self.rollout_policy.policy.get_action_predictions(obs_perf, goal_perf, policy_env=policy_env, backup_policy=self.backup_policy, num_trajectories=num_samples)
        actions_candidates = np.stack([actions.cpu().numpy() for actions in actions_candidates], 0) # (B, Tp, Da)
        
        # State prediction model
        init_states = einops.repeat(obs_dict["flat"][-1, :], 'd -> repeat d', repeat=num_samples) # (B, Dx)
        plans_candidates = self.predictor(init_states, actions_candidates)
        
        # TODO: Ranking the plans
        if num_samples > 1:
            best_plan, best_plan_idx, ranking_info = self.rank_plans(plans_candidates, obs_dict, goal_dict)
            best_actions = actions_candidates[best_plan_idx]
            self.ranking_info = ranking_info
        else:
            best_plan = plans_candidates[0]
            best_actions = actions_candidates[0]

        self.action_predictions = best_actions
        
        return best_plan, best_actions
        
    # def postprocess_to_actions(self, reference_traj, obs_dict):
    #     """
    #     Given the reference trajectory and observation dictionary, 
    #     postprocess to actions for the robomimic environment.
    #     TODO: Implement the Map: backup_policy.state_dim -> rollout_policy.action_dim
    #     """
    #     Kp = 30
    #     Kd = 0.5
        
    #     state = self.obs_dict_to_state(obs_dict)
    #     x  = state[:2]
    #     dx = state[2:]

    #     state = np.concatenate([x, dx])

    #     actions = []
    #     states  = [state]
    #     # TODO: change to track the next waypoint
    #     for i in range(len(reference_traj)-1):
    #         x_des = reference_traj[i+1][1]
    #         action = Kp*(x_des - state[:2]) - Kd*state[2:]
    #         action = np.expand_dims(action, 0)
            
    #         plan = self.predictor(np.expand_dims(state, 0), 
    #                               np.expand_dims(action, 0))[0]
            
    #         next_state = np.concatenate([plan.x_des[-1], plan.dx_des[-1]])
    #         state = next_state
            
    #         actions.append(action[0])
    #         states.append(state)

    #     actions = np.stack(actions, 0)
        
    #     states  = np.stack(states, 0)
    #     error = states[:, :2] - reference_traj.x_des.cpu().numpy()
    #     error = np.linalg.norm(error, axis=1)

    #     # self.disp(f"Backup Plan Tracking Error: {error.max()}")

    #     return actions
    
    # @time_profile
    def check_head_plan_safety(self, plan, obs_dict):
        """
        TODO: Implement the safety verifier, maybe this better suit as a method of the Planner class
        """
        super().check_head_plan_safety(plan = plan, obs_dict = obs_dict)
        collision = self.backup_policy.collision_check(plan, obs_dict)
        return not collision
    
    # @time_profile
    def process_dicts_for_backup_policy(self, obs_dict, goal_dict, plan=None):
        """
        Given the plan, observation dictionary, and goal_dict dictionary from the environment,
        preprocess the data to be compatible with the backup policy

        Args
            obs_dict   : observation dictionary
            goal_dict : goal dictionary
            plan : ReferenceTrajectory object (optional)
        
        Returns
            obs_dict : observation dictionary
            goal_dict: goal dictionary

        # TODO: Since the safety filter is the wrapper of the backup policy, it would be good if we can parse the
        #       available objective function keys from the backup policy
        """
        assert plan is None or isinstance(plan, ReferenceTrajectory)

        obs_dict  = copy_obs_sharing_obstacles(obs_dict)   # was deepcopy(obs_dict); obstacles are read-only
        goal_dict = deepcopy(goal_dict)

        initialize_mode = (plan is None)

        if initialize_mode:
            # When there is no nominal plan available, compute the backup plan starting at current state
            for key in self.safety_critical_state_keys:
                if obs_dict[key].ndim == 2:
                    obs_dict[key] = obs_dict[key][-1, :]

        else:
            # When nominal plan is available, compute the backup plan at the initial state of the nominal plan
            
            # update observation
            # The obs_dict["flat"] and obs_dict["zonotope"]["robot"] is updated accordingly
            if obs_dict["flat"].ndim == 2:
                current_qposvel = obs_dict["flat"][-1, :]
            else:
                current_qposvel = obs_dict["flat"]
            
            backup_qposvel = torch.hstack([plan[0][1], plan[0][2]]) 
            center_shift = torch.hstack([(backup_qposvel[:2] - current_qposvel[:2]), torch.tensor(0)])

            obs_dict["flat"] = backup_qposvel 
            obs_dict["zonotope"]["robot"][0] += center_shift

            # update goal_dict dictionary to provide info. about the objective function
            goal_dict["reference_trajectory"] = plan

        return obs_dict, goal_dict

    @time_profile
    def rank_plans(self, plans, obs_dict, goal_dict):
        OBJ_VAL_INIT = 1e4

        best_obj_val = OBJ_VAL_INIT

        ranking_dict = dict()
        for idx, plan in enumerate(plans):
            tail_plan         = plan[self.n_head:]
            backup_plan, info = self.compute_backup_plan(tail_plan, obs_dict, goal_dict)

            is_safe = info["status"] == 0
            obj_val = info["obj_val"]

            if is_safe and obj_val < best_obj_val:
                best_obj_val = obj_val
                best_plan = plan
                best_plan_idx = idx
            
            # log info into ranking_dict
            # metrics
            ranking_dict[idx] = dict()
            ranking_dict[idx]["plan"]        = plan
            ranking_dict[idx]["safe"]        = is_safe
            ranking_dict[idx]["obj_val"]     = obj_val

            # optional
            ranking_dict[idx]["backup_plan"] = backup_plan
            ranking_dict[idx]["backup_FRS"]  = self.backup_policy.FRS

        if best_obj_val == OBJ_VAL_INIT:
            # This plan would not be used anyway
            best_plan = plans[0]
            best_plan_idx = 0
        
        self.pretty_print_ranking(ranking_dict)

            # NOTE: maybe change the mode of the backup plan to target mode?
        
        return best_plan, best_plan_idx, ranking_dict
    
    def pretty_print_ranking(self, ranking_dict):
        print("====================================================================")
        print("Ranking of the Plans")
        for idx in range(len(ranking_dict)):
            print(f'{idx}: Plan {ranking_dict[idx]["plan"].x_des[-1]} | ObjVal {ranking_dict[idx]["obj_val"]} | Safe {ranking_dict[idx]["safe"]}')
        print("====================================================================")
