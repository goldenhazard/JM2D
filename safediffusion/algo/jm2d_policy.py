"""JM2D and Gibbs samplers for receding-horizon safety filtering (arXiv 2509.08775, Sec. 5.1).

``JM2DPolicy`` was ``SafeDiffusionPolicy`` in the research code (SafeRobomimic@robomimic = origin/rail_release,
8b2371d2, ``safediffusion/algo/safe_diffusion_policy.py``). Two guidance modes are kept (``align.mode``):

* ``jm2d``  (legacy ``score_importance_sampling``): ``step_score_importance_sampling``, the JM2D update.
* ``gibbs`` (legacy ``gibbs_sampling``): ``multistep_gibbs_sampling``, ported from SafeRobomimic@paper/gibbs
  (be46370b, ``safe_diffusion_policy.py:1599``); the Gibbs baseline of Fig. 4.

Algorithm bodies are unchanged; the research code's other (experimental, not in the paper) guidance modes and the
unused ``serial``/``parallel`` score styles were removed.
"""
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Dict, Protocol, Any
import multiprocessing as mp
import time

import torch
import numpy as np
import einops
from tqdm import tqdm

from robomimic.algo import DiffusionPolicyUNet
import robomimic.utils.tensor_utils as TensorUtils

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from safediffusion.algo.planner_base import ReferenceTrajectory
from safediffusion.algo.safety_filter import SafetyFilter
from safediffusion.algo.planner_base import FEASIBLE_FLAG
from safediffusion.utils.deco_utils import time_profile

# guidance modes; legacy names are accepted for old configs
GUIDANCE_MODES = {"jm2d": "jm2d", "score_importance_sampling": "jm2d", "gibbs": "gibbs", "gibbs_sampling": "gibbs"}


#######################
# Protocols
#######################

class ScoreFunction(Protocol):
    """Protocol defining the interface for score functions"""
    def __call__(self,
                action_samples: Union[np.ndarray, torch.Tensor],
                Ks_samples: Union[np.ndarray, torch.Tensor],
                obs_dict: Dict[str, Any],
                goal_dict: Dict[str, Any],
                ) -> Tuple[np.ndarray, Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Dict]:
        """
        Score the action samples and backup parameters.
        
        Args:
            action_samples: Action samples of shape (N, T, D_A)
            Ks_samples: Backup parameters of shape (N, N_K, D_K)
            obs_dict: Observation dictionary
            goal_dict: Goal dictionary
            return_plans: Whether to return plans
            
        Returns:
            Tuple containing:
                headplan_scores: Binary scores of shape (N,)
                Obj: Objective values of shape (N, N_K)
                Cons: Constraint values of shape (N, N_K)
        """
        ...

class JM2DPolicy(SafetyFilter):
    """
    Safety filter that samples the DP actions and the backup parameter jointly (JM2D, or the Gibbs baseline).
    """    
    def __init__(self, 
                 rollout_policy, 
                 backup_policy, 
                 predictor, 
                 dt_action, 
                 param_schedule = 'linear',
                 param_schedule_timestep = 100,
                 dtype=torch.float32, 
                 device=torch.device("cpu"),
                 **config):
        
        super().__init__(rollout_policy, backup_policy, predictor, dt_action, dtype, device, **config)

        self.N_steps_mppi          = config["align"]["N_steps_mppi"]
        self.timestep_for_guidance = config["align"]["timestep_for_guidance"] # apply guidance at this time
        self.N_sample_action       = config["align"]["N_sample_action"]
        self.N_sample_backup       = config["align"]["N_sample_backup"]
        self.opt_K_again           = config["align"]["opt_K_again"]
        self.score_comp_style      = config["align"]["score_comp_style"]
        self.mode                  = GUIDANCE_MODES[config["align"]["mode"]]
        self.temp_mppi             = config["align"]["temp_mppi"]
        self.N_sample_trajopt      = config["align"]["N_sample_trajopt"]

        assert not ((self.ddim_eta == 0) and (self.mode == "jm2d")), "JM2D is at no use with deterministic sampler"
                    
        # Set up score function
        self.model_score_fn: ScoreFunction
        if self.score_comp_style == "fast":
            self.model_score_fn = self.fast_score_batch
        else:
            raise ValueError(f"Unknown score computation style: {self.score_comp_style}")
        
        # Set-up for parallel evaluation (pool of MuJoCo predictors, one task per DP candidate)
        if self.score_comp_style in ["fast"]:
            num_workers = config["align"]["num_workers"] or mp.cpu_count()
            self.num_processes = min(num_workers, self.N_sample_action)
            self.pool = mp.Pool(processes=self.num_processes, 
                            initializer=self._init_worker, 
                            initargs=(self,))
            
        # Set-up param scheduler
        self.param_scheduler = DDPMScheduler(
            num_train_timesteps=param_schedule_timestep,
            beta_schedule=param_schedule,
            clip_sample=False,
            prediction_type="epsilon"
        )

        # NOTE: eta should be reflected here, check if noise scheduler responds to eta
            
        # data structure for diffusion visualization
        self.render_diffusion   = config["render"]["diffusion"]["enabled"]
        self.diffusion_vis_data = {}

        self.param_lb = torch.tensor(self.backup_policy.k_lb, dtype=self.dtype, device=torch.device("cuda"))
        self.param_ub = torch.tensor(self.backup_policy.k_ub, dtype=self.dtype, device=torch.device("cuda"))

    def __del__(self):
        """Cleanup resources when object is destroyed"""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    # -------------------------------------
    # Parallel evaluation of plan
    # -------------------------------------
    @staticmethod
    def _init_worker(instance):
        """Initialize each worker process with a copy of the instance"""
        global _instance
        _instance = instance
    
    
    @staticmethod
    def _rollout_and_headplan_wrapper(args):
        global _instance
        assert isinstance(_instance, JM2DPolicy), "instance error"
        
        actions, obs_dict = args

        x0 = _instance.obs_dict_to_state(obs_dict)

        plan = _instance.predictor(x0[None, :], actions[None, :])[0]
        
        Ta                = _instance.n_head
        head_plan         = plan[:Ta+1]
        headplan_score    = _instance.check_head_plan_safety(head_plan, obs_dict)# 1 if safe 0 if unsafe
        
        return plan, headplan_score
    
    @time_profile
    def parallel_rollout_and_evaluate_headplan(self, actions, obs_dict):
        args = [(action, obs_dict) for action in actions]

        results = self.pool.map(self._rollout_and_headplan_wrapper, args)

        plans           = [result[0] for result in results]
        headplan_scores = np.array([result[1] for result in results])

        return plans, headplan_scores
    
    # @time_profile
    def fast_score_batch(self, actions, Ks_samples, obs_dict, goal_dict):
        """ Parallelly rollout sim and prepare data and batch-evaluate the objective and constraint

        Args:
            actions: (np.array) the K samples of (B, Tp-To, D_A)
            Ks  : (np.array) the K samples of (B, N_K, D_K)
            obs_dict: (dict) the observation dictionary
            goal_dict: (dict) the goal dictionary

        
        Returns:
            headplan_score : (bool) true if safe
            Objs           : (np.array) objective function of (N_K, 1)
            Cons           : (np.array) maximum constraint violation of (N_K, 1)
        """
        BP = actions.shape[0]
        B  = Ks_samples.shape[1]
        Tp = self.rollout_policy.policy.algo_config.horizon.prediction_horizon
        To = self.rollout_policy.policy.algo_config.horizon.observation_horizon
        
        assert actions.shape[1] == Tp-To+1, "This assume that the given actions start from To-1"

        plans, headplan_scores = self.parallel_rollout_and_evaluate_headplan(actions, obs_dict)
        obs_dict["collision_center"] = self.backup_policy.obs_dict_to_collision_center(obs_dict)

        # start 
        Ta  = self.n_head
        
        # (2) tail-plan score
        import time
        t_start = time.time()
        problem_data_all = []
        for i, (plan, Ks) in enumerate(zip(plans, Ks_samples)):
            if headplan_scores[i]:
                # batching code
                Ta = self.rollout_policy.policy.algo_config.horizon.action_horizon
                
                tail_plan = plan[Ta:]
                tail_plan.set_start_time(0)
                
                ob_backup_iter, goal_backup_iter = self.process_dicts_for_backup_policy(obs_dict   = obs_dict, 
                                                                                        goal_dict  = goal_dict, 
                                                                                        plan       = tail_plan)
                
                self.update_backup_policy_weight()
                
                ob_TO_iter, goal_TO_iter = self.backup_policy.process_observation_and_goal_for_TO(
                                                                        obs_dict  = ob_backup_iter,
                                                                        goal_dict = goal_backup_iter)
                
                # for batch evaluation, the collision_center is fixed to keep the number of constraints equal
                problem_data_iter = self.backup_policy._prepare_problem_data(obs_dict  = ob_TO_iter,
                                                                            goal_dict = goal_TO_iter)
                problem_data_all.append(problem_data_iter)
        if self.verbose:
            print(f"Time for processing: {time.time() - t_start:.2f} seconds")
        # end
        mask = headplan_scores

        Obj_ret = torch.ones(BP, B) * 1e6
        Con_ret = torch.ones(BP, B) * 1e6
        
        t_start = time.time()
        if mask.any():
            self.update_backup_policy_weight()
            Objs  = self.backup_policy._compute_objective_batch(Ks_samples[mask], problem_data_all, use_gpu = True)
            Conss = self.backup_policy._compute_constraints_batch(Ks_samples[mask], problem_data_all, use_gpu = True)
            
            Cons  = Conss.max(dim=-1)[0]
            # Cons  = torch.relu(Conss).sum(dim=-1)

            Obj_ret[mask]  = Objs
            Con_ret[mask]  = Cons

            # NOTE: Only for Debugging
            # obj_deltas = []
            # cons_deltas = []
            # for i in range(BP):
            #     for j in range(B):
            #         obj_delta =  (Objs[i, j].numpy() - self.backup_policy._compute_objective(Ks_samples[mask][i, j], problem_data_all[i])[0])
            #         cons_delta =  (Conss[i, j].numpy() - self.backup_policy._compute_constraints(Ks_samples[mask][i, j], problem_data_all[i])[0])
            #         obj_deltas.append(obj_delta)
            #         cons_deltas.append(cons_delta)
            
        if self.verbose:
            print(f"Batch Eval time: {time.time() - t_start:.2f} seconds")

        extraout = dict(plans = plans)

        return headplan_scores, Obj_ret, Con_ret, extraout

    # @time_profile
    def sample_next_plan_actions_verify_head_plan_and_sample_backup_plan(self, obs_dict, goal_dict=None):
        """
        Sample the next plan and corresponding backup plan.
        
        Design of main sampling strategy

        Args
            obs_dict   : observation dictionary
            goal_dict : goal dictionary
        
        Returns
            plan             : (ReferenceTrajectory)
            actions          : (T_p, D_a) np.ndarray
            head_plan_safety : (bool) safety of the head plan
            backup_plan      : (ReferenceTrajectory)
        """
        actions, plan, opt_var, opt_info = self.sample_actions_plan_and_optvar(obs_dict=obs_dict, 
                                                                               goal_dict=goal_dict, 
                                                                               replace_obs_queue = (not self.delay_observation))
        
        # verify head plan safety
        head_plan = plan[:self.n_head+1]
        head_plan_safety = self.check_head_plan_safety(head_plan, obs_dict)

        # verify backup plan existence
        backup_plan = None
        if opt_info["status"] in FEASIBLE_FLAG:
            tail_plan   = plan[self.n_head:]
            obs_tail, _ = self.process_dicts_for_backup_policy(obs_dict, goal_dict, tail_plan)
            
            k           = self.backup_policy.unnormalize_k(opt_var)
            backup_plan = self.backup_policy.make_plan(obs_tail, k)
            
            # for visualization purpose
            self.backup_policy.FRS = self.backup_policy.get_FRS_from_obs_and_optvar(obs_tail, k)

        # remove the status key
        del opt_info["status"]
            
        return plan, actions, head_plan_safety, backup_plan, opt_info
    
    # @time_profile
    def sample_actions_plan_and_optvar(self, obs_dict, goal_dict, replace_obs_queue=False):
        """
        Sample the actions, plan and optvar
        
        Args:
            obs_dict (dict): the observation dictionary
            goal_dict (dict): the goal dictionary
            
        Returns:
            actions (np.ndarray): the actions
            plan (ReferenceTrajectory): the plan
            opt_var (np.ndarray): the opt var
            opt_info (dict): the opt info
        """
        assert replace_obs_queue, "We disable the delayed observation for now."

        if self.render_diffusion:
            self.diffusion_vis_data.clear()

        diff_policy = self.rollout_policy.policy
        assert isinstance(diff_policy, DiffusionPolicyUNet)
        
        if self.verbose:
            print(f"Running Guidance with mode: {self.mode} and K optimization: {self.opt_K_again}")
        
        # obs_dict: key: [1, To, Do]
        To = diff_policy.algo_config.horizon.observation_horizon
        Ta = diff_policy.algo_config.horizon.action_horizon
        Tp = diff_policy.algo_config.horizon.prediction_horizon
        dim_optvar = len(self.backup_policy.opt_dim)
        
        start   = To - 1

        obs_model  = deepcopy(obs_dict)
        goal_model = deepcopy(goal_dict)

        # Get the observation tensor
        obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(diff_policy.obs_queue))
        obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k, v in obs_dict_list.items())
        obs_dict_tensor = dict((k, v.squeeze(-1)) for k, v in obs_dict_tensor.items())

        # Process the goal dictionary: process key and repeat
        if goal_dict is not None:
            goal_dict = {k: goal_dict[k] for k in self.rollout_policy_goal_keys}
            goal_dict = self.rollout_policy._prepare_observation(goal_dict)
            goal_dict = {k: goal_dict[k].unsqueeze(1) for k in goal_dict}
            goal_dict = {k: v.repeat((1, To) + (1,) * (v.dim() - 2)) for k, v in goal_dict.items()}
            
        # Get the observation / goal tensor
        inputs = {
            'obs': obs_dict_tensor,  # (B, To, D)
            'goal': goal_dict        # (B, To, D)
        }

        for k in diff_policy.obs_shapes:
            # inputs['obs'][k] should be [B, T, D1, D2, ..., Dn] for inputs
            assert inputs['obs'][k].ndim - 2 == len(diff_policy.obs_shapes[k])
        
        for k in diff_policy.goal_shapes:
            assert inputs['goal'][k].ndim - 2 == len(diff_policy.goal_shapes[k])
        
        obs_features = TensorUtils.time_distributed(inputs, diff_policy.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)
            
        # Start loading diffusion model
        action_dim = diff_policy.ac_dim
        if diff_policy.algo_config.ddpm.enabled is True:
            num_inference_timesteps = diff_policy.algo_config.ddpm.num_inference_timesteps
        elif diff_policy.algo_config.ddim.enabled is True:
            num_inference_timesteps = diff_policy.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        nets = diff_policy.nets
        if diff_policy.ema is not None:
            diff_policy.ema.copy_to(parameters=diff_policy._shadow_nets.parameters())
            nets = diff_policy._shadow_nets
            
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=diff_policy.device)
        naction = noisy_action

        # init scheduler
        diff_policy.noise_scheduler.set_timesteps(num_inference_timesteps)

        # nparam = torch.randn((dim_optvar), device = diff_policy.device)
        nparam = randn_tensor((B, dim_optvar), device = diff_policy.device)
        # t0     = diff_policy.noise_scheduler.timesteps[0]
        # alpha0 = diff_policy.noise_scheduler.alphas_cumprod[t0]
        # K      = torch.clip(nparam/(alpha0 ** 0.5), -1, 1)
        
        # information to estimate score function from the "model"
        for _, timestep in tqdm(enumerate(diff_policy.noise_scheduler.timesteps)):
            if timestep <= self.timestep_for_guidance:
                if self.mode == "jm2d":
                    naction, nparam = self.step_score_importance_sampling(
                        naction = naction, nparam = nparam, nets = nets, timestep = timestep, obs_cond = obs_cond,
                        obs_model = obs_model, goal_model = goal_model, eta = self.ddim_eta)
                
                elif self.mode == "gibbs":
                    naction, nparam = self.multistep_gibbs_sampling(
                        naction = naction, nparam = nparam, nets = nets, timestep = timestep, obs_cond = obs_cond,
                        obs_model = obs_model, goal_model = goal_model, eta = self.ddim_eta)
                
                else:
                    raise NotImplementedError
            
            else:
                naction = self.step_normal(
                    nets= nets, 
                    naction= naction, 
                    timestep= timestep, 
                    obs_cond= obs_cond
                )

        # Final actions and verification
        actions_final = naction[:, start:]
        actions_final = self.maybe_unnormalize_actions_perf(actions_final.cpu().numpy())
        param_final = nparam
        
        # Make to the plan
        # Anchor
        x0             = self.obs_dict_to_state(obs_model)[None, :]
        plan_final     = self.predictor(x0, actions_final)[0]
        headplan_final = plan_final[:Ta+1]
        actions_final  = actions_final[0]
        param_final    = param_final[0].to(self.device)
        
        head_plan_safety = self.check_head_plan_safety(headplan_final, obs_model)
        if head_plan_safety:
            if self.opt_K_again:
                param_final, info = self.run_sampling_based_backup_trajopt(
                    plan = plan_final, 
                    obs_dict = obs_model, 
                    goal_dict = goal_model, 
                    K_init = param_final
                )
            
            Obj, Cons = self.eval_obj_and_cons_of_plan(plan      = plan_final, 
                                                        Ks        = param_final.unsqueeze(0), 
                                                        obs_dict  = obs_model, 
                                                        goal_dict = goal_model)
            info = dict()
            if Cons.max() < 0:
                info["status"] = FEASIBLE_FLAG[0]
            else:
                info["status"] = -1
            info["objective"] = Obj.squeeze(0)

        else:
            info = dict()
            info["status"] = -1
            info["objective"] = 1e4
            
        if self.render_diffusion:
            self.diffusion_vis_data[-1] = dict(
                actions     = actions_final,
                plan        = plan_final,
                K           = param_final,
                obs_model   = obs_model,
                goal_model  = goal_model,
                obs_cond    = obs_cond,
                feasible    = info["status"]
            )
    
        return actions_final, plan_final, param_final, info

    # @time_profile
    def run_sampling_based_backup_trajopt(self, plan, obs_dict, goal_dict, K_init=None, verbose = True):
        r""" Compute the backup trajectory parameter using sampling based optimization

        Args:
            plan (ReferenceTrajectory): plan length of t=0..Tp
            obs_dict (dict) observation dictionary at t=0
            goal_dict (dict) goal dictionary at t=Tp
        
        Returns:
            optvar (np.ndarray): optimization variable \in [-1, 1], t=Ta
            
        NOTE: Currently runs mppi without covariance update. Look at samping-based literature for covariance update
        TODO: This can be definitely improved...
        """
        # parse frequently used static variables
        dim_optvar   = len(self.backup_policy.opt_dim)
        N_samples    = self.N_sample_trajopt
        N_mppi_steps = self.N_steps_mppi
        
        mean_x = torch.zeros(dim_optvar)
        cov_x  = torch.eye(dim_optvar)

        if K_init is not None:
            mean_x = torch.tensor(K_init)
        
        for mppi_step in range(N_mppi_steps):
            # Step 1. sample from current distribution
            xs_sampled = torch.distributions.MultivariateNormal(mean_x,
                                                                covariance_matrix = cov_x,
            ).sample((N_samples,))
            xs_sampled = torch.clip(xs_sampled, self.param_lb.cpu(), self.param_ub.cpu())

            # Step 2. evaluate the weight for each sample
            Objs, Conss = self.eval_obj_and_cons_of_plan_batch(plan      = plan, 
                                                               Ks        = xs_sampled, 
                                                               obs_dict  = obs_dict, 
                                                               goal_dict = goal_dict)
            Conss = Conss.max(dim=-1)[0]

            # Objs_Z       = (Objs - Objs.mean())/Objs.std()
            Objs_Z       = (Objs - Objs.mean())
            weights_all  = torch.ones((N_samples))/(N_samples) # (N_k,)
            weights_all *= np.exp(-Objs_Z/self.temp_mppi)
            weights_all *= (Conss < 0)
            
            # Step 3. If valid sample exists, update the mean (and covariance)
            if torch.abs(weights_all).sum() > 1e-5:
                # mean update strategy
                weights_all /= weights_all.sum()

                # mean update strategy
                if mppi_step < N_mppi_steps - 1:
                    new_mean = torch.sum(weights_all[:,None] * xs_sampled, axis=0)
                else:
                    new_mean = xs_sampled[torch.argmax(weights_all)]
                
                # covariance update strategy
                # NS = self.rollout_policy.policy.noise_scheduler
                # new_sigma = 1/NS.alphas_cumprod[int((1-(mppi_step+1)/self.N_steps_mppi) * NS.num_inference_steps)] - 1
                # new_cov   = new_sigma * torch.eye(dim_optvar)
                
                mean_x = new_mean
                # cov_x  = new_cov
            
            if verbose:
                if (weights_all != 0).sum() > 0:
                    N_eff = self.effective_sample_size(weights_all, "normalized")
                else:
                    N_eff = 0
                self.disp(f"Objective {Objs.mean()}, Feasible Samples {(Conss<0).sum()}, N_eff (normalized): {N_eff}")
        
        opt_is_feasible = torch.abs(weights_all.sum()) > 1e-5
        
        info = dict()
        if opt_is_feasible:
            info["status"] = FEASIBLE_FLAG[0]
        else:
            info["status"] = -1

        return mean_x, info
    
    # @time_profile
    def eval_obj_and_cons_of_plan(self, Ks, plan, obs_dict, goal_dict):
        """
        Evaluate the objective and constraints of the backup policy given params, plan, obs_dict, goal_dict
        
        Args:
            Ks: (torch.tensor) trajectory parameters, shape -> (B, dim_optvar)
            plan: (ReferenceTrajectory) a single plan, the backup trajectory is evaluated at plan[Ta]
            obs_dict: (dict) observation at t
            goal_dict: (dict) goal_dict at t
        
        Returns:
            Objs: (np.ndarray) objective values, shape -> (B, 1)
            Conss: (np.ndarray) constraint values, shape -> (B, n_constraints)
        
        """
        Ta = self.rollout_policy.policy.algo_config.horizon.action_horizon
        
        tail_plan = plan[Ta:]
        tail_plan.set_start_time(0)
        
        ob_backup_iter, goal_backup_iter = self.process_dicts_for_backup_policy(obs_dict   = obs_dict, 
                                                                                goal_dict  = goal_dict, 
                                                                                plan       = tail_plan)
        
        self.update_backup_policy_weight()
        
        ob_TO_iter, goal_TO_iter = self.backup_policy.process_observation_and_goal_for_TO(
                                                                obs_dict  = ob_backup_iter,
                                                                goal_dict = goal_backup_iter)
        
        problem_data_iter = self.backup_policy._prepare_problem_data(obs_dict  = ob_TO_iter,
                                                                     goal_dict = goal_TO_iter)
        
        # Evaluate the objective and constraints
        Objs  = np.array([self.backup_policy._compute_objective(optvar, problem_data_iter)[0] for optvar in Ks])
        Conss = np.array([self.backup_policy._compute_constraints(optvar, problem_data_iter)[0] for optvar in Ks])

        return Objs, Conss
    
    # @time_profile
    def eval_obj_and_cons_of_plan_batch(self, Ks, plan, obs_dict, goal_dict):
        """
        Evaluate the objective and constraints of the backup policy given params, plan, obs_dict, goal_dict
        
        Args:
            Ks: (torch.tensor) trajectory parameters, shape -> (B, dim_optvar)
            plan: (ReferenceTrajectory) a single plan, the backup trajectory is evaluated at plan[Ta]
            obs_dict: (dict) observation at t
            goal_dict: (dict) goal_dict at t
        
        Returns:
            Objs: (np.ndarray) objective values, shape -> (B, 1)
            Conss: (np.ndarray) constraint values, shape -> (B, n_constraints)
        """
        obs_dict["collision_center"] = self.backup_policy.obs_dict_to_collision_center(obs_dict)
        
        Ta = self.rollout_policy.policy.algo_config.horizon.action_horizon
        
        tail_plan = plan[Ta:]
        tail_plan.set_start_time(0)
        
        ob_backup_iter, goal_backup_iter = self.process_dicts_for_backup_policy(obs_dict   = obs_dict, 
                                                                                goal_dict  = goal_dict, 
                                                                                plan       = tail_plan)
        
        self.update_backup_policy_weight()
        
        ob_TO_iter, goal_TO_iter = self.backup_policy.process_observation_and_goal_for_TO(
                                                                obs_dict  = ob_backup_iter,
                                                                goal_dict = goal_backup_iter)
        
        problem_data = self.backup_policy._prepare_problem_data(obs_dict  = ob_TO_iter,
                                                                goal_dict = goal_TO_iter)
        
        # Evaluate the objective and constraints
        self.update_backup_policy_weight()
        Objs  = self.backup_policy._compute_objective_batch(Ks.unsqueeze(0), [problem_data], use_gpu = True)
        Conss = self.backup_policy._compute_constraints_batch(Ks.unsqueeze(0), [problem_data], use_gpu = True)
        
        Objs = Objs[0]
        Conss = Conss[0]

        return Objs, Conss
    def get_noise(self, noisy_samples, original_samples, timesteps):
        """
        Calculate Tweedie Noise

        Args:
            noisy_samples: (torch.tensor) $x_{t}$: noisy action tensor , shape -> (B, Tp, A)
            original_samples: (torch.tensor) $x_{t-1}$: original action tensor, shape -> (B, Tp, A)
            timesteps: (int), diffusion denoising timestep t, 0 if data

        Returns:
            noise: (torch.tensor) noise tensor, shape -> (B, Tp, A)
        """
        noise_scheduler = self.rollout_policy.policy.noise_scheduler
        alphas_cumprod = noise_scheduler.alphas_cumprod.clone().to(dtype=original_samples.dtype, device=original_samples.device)
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

    def effective_sample_size(self, weights, method="Kish"):
        """ Effective sample size
        
        Args
            weights (torch.tensor) (B1, B2, ..., Bn)
        """
        if method == "Kish":
            W2    = (weights/weights.sum())**2
            N_eff = 1/W2.sum()

        if method == "normalized":
            # Kish ESS divided by the number of samples (in [1/n, 1])
            W2        = (weights/weights.sum())**2
            n_samples = weights.numel()
            N_eff     = 1/(n_samples * W2.sum())
        
        return N_eff
    
    def assert_weights_normalized(self, weights, atol=1e-4):
        """
        Assert that the weights are normalized

        Args:
            weights (torch.tensor): The weights to check (B, N)
        """
        assert weights.ndim == 2, "Weights must be a 2D tensor"
        
        weight_close_to_one  = (weights.sum(dim=-1) - 1).abs() < atol
        weight_close_to_zero = (weights.sum(dim=-1)).abs() < atol
        
        assert torch.logical_or(weight_close_to_one, weight_close_to_zero).all(), "Weights are not normalized"

    def get_param_covariance(self, data_timestep):
        """
        Return the covariance matrix of the parameters at a given timestep

        Args:
            data_timestep (int): The timestep of the data

        Returns:
            cov_param (torch.tensor): The covariance matrix of the parameters
        """
        data2param_scale = self.rollout_policy.policy.noise_scheduler.num_train_timesteps // self.param_scheduler.num_train_timesteps
        param_timestep = data_timestep // data2param_scale

        param_alphas_cumprod = self.param_scheduler.alphas_cumprod[param_timestep]
        cov_param = (1/param_alphas_cumprod - 1) * torch.eye(len(self.backup_policy.opt_dim))

        return cov_param
    
    def update_weights_from_constraints(self, weights, constraints, lag_coeff = 100, constraint_mode = 'rejection'):
        """
        Update the weights from the constraints

        Args:
            timestep (int): The timestep
            weights (torch.tensor): The weights (B, N)
            constraints (torch.tensor): The constraints (B, N), Cons < 0 means feasible

        Returns:
            weights (torch.tensor): The updated weights
        """
        assert weights.shape == constraints.shape, "Weights and constraints must have the same shape"

        if constraint_mode == 'lagrangian':
            # numerical stability
            constraints = torch.relu(constraints)
            constraints = constraints - constraints.min(dim=1, keepdim=True)[0]

            # update weights
            constraint_weights = torch.nn.functional.softmax(-lag_coeff * constraints / self.temp_mppi, dim=1)
            old_weights = weights.clone()
            weights *= constraint_weights
            weights = weights / weights.sum(dim=1, keepdim=True)

            if torch.isnan(weights).any():
                weights = old_weights.clone()

        elif constraint_mode == 'augmented_lagrangian':
            raise NotImplementedError("Augmented Lagrangian is not tested yet")
            # Track optimization iterations
            self.constraint_iter += 1
            
            # Augmented Lagrangian term: λc + (ρ/2)c²
            # Only apply to constraint violations (c > 0)
            constraints_violation = torch.relu(constraints)
            aug_lag_term = self.current_lambda * constraints_violation + (self.rho/2) * constraints_violation**2
            
            # For numerical stability
            aug_lag_term = aug_lag_term - aug_lag_term.min(dim=1, keepdim=True)[0]
            
            # Update weights using softmax
            constraint_weights = torch.nn.functional.softmax(-aug_lag_term / self.temp_mppi, dim=1)
            
            # Update trajectory weights
            old_weights = weights.clone()
            weights *= constraint_weights
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            if torch.isnan(weights).any():
                weights = old_weights.clone()

            # Compute mean constraint violation (weighted)
            mean_constraint = constraints.mean(dim=1, keepdim=True)[0]
            # mean_constraint = (constraints * weights).sum(dim=1, keepdim=True)
            mean_violation = (constraints_violation * weights).sum(dim=1, keepdim=True)
            max_violation = constraints_violation.max(dim=1, keepdim=True)[0]
            
            # Update lambda multipliers (only for batches with violations)
            has_violation = (max_violation > self.max_constraint_violation).float()
            self.current_lambda -= self.lambda_lr * mean_constraint
            # self.current_lambda += self.lambda_lr * mean_violation * has_violation
            self.current_lambda = torch.relu(self.current_lambda)  # Keep non-negative

            if self.verbose:
                self.disp(f"Lambda: {self.current_lambda.mean().item():.2f}, min constraint: {mean_constraint.mean().item()}")
            
            # Increase rho periodically if constraints still violated
            if self.constraint_iter % 2 == 0 and max_violation.mean() > self.max_constraint_violation:
                self.rho *= self.rho_growth_factor
                if self.verbose:
                    self.disp(f"Increasing rho to {self.rho:.4f}, mean violation: {mean_violation.mean().item():.4e}")
            
        elif constraint_mode == 'rejection':
            # reject samples that violate constraints
            weights[constraints > 0] = 0.0
            valid_mask = (weights > 0).any(dim=1)
            weights[valid_mask] /= weights[valid_mask].sum(dim=1, keepdim=True)
        
        else:
            raise ValueError(f"Invalid constraint mode: {constraint_mode}")
        
        return weights
    
    def step_normal(self, nets, naction, timestep, obs_cond):
        r"""
        Reverse and forward diffusion step normally.

        Args:
            nets: (torch.nn.Module) noise prediction model epsilon(x_{k}, k, obs_cond)
            naction: (torch.tensor) x_{step}
            step: (int) x_{-1} is final, N if full gaussian noise
            obs_cond: (torch.tensor) global conditioning

        Returns:
            naction: (torch.tensor) x_{step-1} or x_{step-k} (for DDIM)
        """
        diff_policy = self.rollout_policy.policy
        
        # Predict noise
        noise_pred = nets["policy"]["noise_pred_net"](sample = naction, timestep = timestep, global_cond = obs_cond)

        # reverse diffusion step (remove noise) and add_noise to generate x_{timestep-1} 
        naction    = diff_policy.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=naction,
            eta = self.ddim_eta
        ).prev_sample

        return naction
    
    @time_profile
    def step_score_importance_sampling(self, nets, naction, nparam, timestep, obs_cond, obs_model, goal_model, eta=0, verbose=True):
        """
        JM2D denoising step (named score importance sampling, SIS, in the research code): joint importance
        weighting of N clean-action estimates and N x N_K backup-parameter samples.

        Args:
            nets: (torch.nn.Module) noise prediction model epsilon(x_{k}, k, obs_cond)
            naction: (torch.tensor) x_{step}
            nparam: (torch.tensor) x_{-1}
            timestep: (int) x_{-1} is final, N if full gaussian noise
            obs_cond: (torch.tensor) global conditioning
            obs_model: (dict) observation model
            goal_model: (dict) goal model
            eta: (float) DDIM eta
            verbose: (bool) whether to print verbose output

        Returns:
            naction_next: (torch.tensor) x_{step-1}
            nparam: (torch.tensor) x_{-1}
        """
        if timestep == 0:
            naction_next = self.step_normal(nets, naction, timestep, obs_cond)
            return naction_next, nparam

        # Get dimensions
        B, Tp, A = naction.shape
        _, P = nparam.shape
        N = self.N_sample_action  
        N_K = self.N_sample_backup
        To = self.rollout_policy.policy.algo_config.horizon.observation_horizon
        
        assert B == 1, "Batch size must be 1 for SIS"
        noise_scheduler = self.rollout_policy.policy.noise_scheduler

        """
        Step 1. Monte-Carlo Sampling
        """
        # More efficient sampling using expand instead of clone + repeat + reshape
        flat_samples = naction.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, Tp, A) # (B*N, Tp, A)
        obs_cond_batch = obs_cond.expand(N, -1) # (N, obs_dim)

        # Full-step denoising
        inner_naction = flat_samples
        inner_timesteps = noise_scheduler.timesteps[noise_scheduler.timesteps <= timestep]
        
        for inner_t in inner_timesteps:
            inner_naction = self.step_normal(nets, inner_naction, inner_t, obs_cond_batch)
        clean_action_samples = inner_naction  # No need for clone, (B*N, Tp, A)

        # Calculate Tweedie Noise
        naction_repeated = naction.expand(N, -1, -1)
        tweedie_noise_samples = self.get_noise(
            noisy_samples=naction_repeated,
            original_samples=clean_action_samples,
            timesteps=timestep
        )
        tweedie_noise_samples = einops.rearrange(tweedie_noise_samples, '(b n) tp a -> b n tp a', b=B, n=N)
        clean_action_samples = einops.rearrange(clean_action_samples, '(b n) tp a -> b n tp a', b=B, n=N)

        """
        Step 2. Parameter Sampling
        """
        cov_param = self.get_param_covariance(timestep)
        param_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(P, device=nparam.device),
            covariance_matrix=cov_param.to(nparam.device)
        )
        param_perturbations = param_dist.sample((B, N, N_K))
        
        # Use expand instead of repeat for nparam
        param_samples = nparam.unsqueeze(1).unsqueeze(2).expand(-1, N, N_K, -1)
        param_samples = param_samples + param_perturbations
        param_samples = torch.clip(param_samples, self.param_lb, self.param_ub)

        """
        Step 3. Evaluate interaction potential
        """
        # Avoid unnecessary copies for evaluation
        clean_actions_eval = clean_action_samples.squeeze(0)
        clean_actions_eval = self.maybe_unnormalize_actions_perf(clean_actions_eval.cpu().numpy())
        clean_actions_eval = clean_actions_eval[:, To-1:]
        param_samples_eval = param_samples.squeeze(0).cpu()

        headplan_scores, objectives, constraints, extraout = self.model_score_fn(
            clean_actions_eval, param_samples_eval, obs_model, goal_model
        )

        # Rest of the function remains largely the same, just removing unnecessary .clone() calls
        headplan_scores = headplan_scores[None, :]  # Add batch dim without clone
        objectives = objectives.unsqueeze(0)  
        constraints = constraints.unsqueeze(0)

        headplan_scores_orig = headplan_scores.copy()
        objectives_orig = objectives.clone()
        constraints_orig = constraints.clone()

        """
        Step 4. Compute Importance Weight
        """

        # Weight computation
        objectives = einops.rearrange(objectives, 'b n nk -> b (n nk)', n=N, nk=N_K)
        constraints = einops.rearrange(constraints, 'b n nk -> b (n nk)', n=N, nk=N_K)
        headplan_scores = torch.from_numpy(-headplan_scores.astype(np.int32)).to(naction.device) + 0.5
        headplan_scores = einops.repeat(headplan_scores, 'b n -> b (n nk)', n=N, nk=N_K)

        # Calculate weights
        weights = torch.zeros_like(objectives)  # Initialize directly
        objectives = objectives - objectives.min(dim=-1, keepdim=True)[0]
        weights = torch.nn.functional.softmax(-objectives/self.temp_mppi, dim=-1)

        # Update weights based on constraints
        weights = self.update_weights_from_constraints(weights, headplan_scores, constraint_mode='rejection')
        max_timestep = noise_scheduler.timesteps[0]
        lag_coeff = 10 * (1-timestep/max_timestep)
        # weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='rejection')
        weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='rejection')
        
        self.assert_weights_normalized(weights)
        """
        Step 5. Compute Optimal Action and Param
        """
        mask_to_update = (weights > 0).any(dim=-1)
        weights = weights.to(naction.device)
        noise_optimal = nets["policy"]["noise_pred_net"](
            sample=naction,
            timestep=timestep,
            global_cond=obs_cond
        )
        tweedie_noise_samples = einops.repeat(tweedie_noise_samples, 'b n tp a -> b (n nk) tp a', nk=N_K)

        if mask_to_update.any():
            noise_optimal[mask_to_update] = (tweedie_noise_samples * weights[..., None, None]).sum(dim=1)[mask_to_update]
            param_samples = einops.rearrange(param_samples, 'b n k p -> b (n k) p')
            nparam[mask_to_update] = (param_samples * weights[..., None]).sum(dim=1)[mask_to_update]

        naction_next = noise_scheduler.step(
            model_output=noise_optimal,
            timestep=timestep,
            sample=naction,
            eta=eta
        ).prev_sample

        # Logging
        if verbose:
            obj_mean = objectives_orig.mean()
            valid_head_ratio = (headplan_scores_orig).sum()/N
            valid_back_ratio = (constraints_orig<0).sum()/(N*N_K)
            ess = self.effective_sample_size(weights, method="normalized")
            self.disp(f"SIS (t={timestep}) Obj = {obj_mean:.2f} | Head = {valid_head_ratio:.2f} | Back = {valid_back_ratio:.2f} | ESS = {ess:.2f}")

        return naction_next, nparam
    
    @time_profile
    def multistep_gibbs_sampling(self, nets, naction, nparam, timestep, obs_cond, obs_model, goal_model, eta=0, verbose=True):
        """
        Gibbs baseline (Fig. 4): alternate an importance-weighted action update with the parameter held fixed
        and a parameter update for the new action. Ported from SafeRobomimic@paper/gibbs (be46370b).

        Args:
            nets: (torch.nn.Module) noise prediction model epsilon(x_{k}, k, obs_cond)
            naction: (torch.tensor) x_{step}
            nparam: (torch.tensor) x_{-1}
            timestep: (int) x_{-1} is final, N if full gaussian noise
            obs_cond: (torch.tensor) global conditioning
            obs_model: (dict) observation model
            goal_model: (dict) goal model
            eta: (float) DDIM eta
            verbose: (bool) whether to print verbose output

        Returns:
            naction_next: (torch.tensor) x_{step-1}
            nparam: (torch.tensor) x_{-1}
        """
        if timestep == 0:
            naction_next = self.step_normal(nets, naction, timestep, obs_cond)
            return naction_next, nparam

        # Get dimensions
        B, Tp, A = naction.shape
        _, P = nparam.shape
        N = self.N_sample_action  
        N_K = self.N_sample_backup
        To = self.rollout_policy.policy.algo_config.horizon.observation_horizon
        
        assert B == 1, "Batch size must be 1 for SIS"
        noise_scheduler = self.rollout_policy.policy.noise_scheduler

        """
        Step 1. Monte-Carlo Sampling
        """
        # More efficient sampling using expand instead of clone + repeat + reshape
        flat_samples = naction.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, Tp, A) # (B*N, Tp, A)
        obs_cond_batch = obs_cond.expand(N, -1) # (N, obs_dim)

        # Full-step denoising
        inner_naction = flat_samples
        inner_timesteps = noise_scheduler.timesteps[noise_scheduler.timesteps <= timestep]
        
        for inner_t in inner_timesteps:
            inner_naction = self.step_normal(nets, inner_naction, inner_t, obs_cond_batch)
        clean_action_samples = inner_naction  # No need for clone, (B*N, Tp, A)

        # Calculate Tweedie Noise
        naction_repeated = naction.expand(N, -1, -1)
        tweedie_noise_samples = self.get_noise(
            noisy_samples=naction_repeated,
            original_samples=clean_action_samples,
            timesteps=timestep
        )
        tweedie_noise_samples = einops.rearrange(tweedie_noise_samples, '(b n) tp a -> b n tp a', b=B, n=N)
        clean_action_samples = einops.rearrange(clean_action_samples, '(b n) tp a -> b n tp a', b=B, n=N)

        """
        Step 2. Parameter Sampling
        """

        param_samples = nparam.unsqueeze(1).unsqueeze(2).expand(-1, N, N_K, -1)
        param_samples = torch.clip(param_samples, self.param_lb, self.param_ub)  # paper/gibbs: (-1, 1); same for hq

        """
        Step 3. Evaluate interaction potential
        """
        # Avoid unnecessary copies for evaluation
        clean_actions_eval = clean_action_samples.squeeze(0)
        clean_actions_eval = self.maybe_unnormalize_actions_perf(clean_actions_eval.cpu().numpy())
        clean_actions_eval = clean_actions_eval[:, To-1:]
        param_samples_eval = param_samples.squeeze(0).cpu()

        headplan_scores, objectives, constraints, extraout = self.model_score_fn(
            clean_actions_eval, param_samples_eval, obs_model, goal_model
        )

        # Rest of the function remains largely the same, just removing unnecessary .clone() calls
        headplan_scores = headplan_scores[None, :]  # Add batch dim without clone
        objectives = objectives.unsqueeze(0)  
        constraints = constraints.unsqueeze(0)

        headplan_scores_orig = headplan_scores.copy()
        objectives_orig = objectives.clone()
        constraints_orig = constraints.clone()

        """
        Step 4. Compute Importance Weight
        """

        # Weight computation
        objectives = einops.rearrange(objectives, 'b n nk -> b (n nk)', n=N, nk=N_K)
        constraints = einops.rearrange(constraints, 'b n nk -> b (n nk)', n=N, nk=N_K)
        headplan_scores = torch.from_numpy(-headplan_scores.astype(np.int32)).to(naction.device) + 0.5
        headplan_scores = einops.repeat(headplan_scores, 'b n -> b (n nk)', n=N, nk=N_K)

        # Calculate weights
        weights = torch.zeros_like(objectives)  # Initialize directly
        objectives = objectives - objectives.min(dim=-1, keepdim=True)[0]
        weights = torch.nn.functional.softmax(-objectives/self.temp_mppi, dim=-1)

        # Update weights based on constraints
        weights = self.update_weights_from_constraints(weights, headplan_scores, constraint_mode='rejection')
        max_timestep = noise_scheduler.timesteps[0]
        lag_coeff = 1000 * (1-timestep/max_timestep)
        weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='rejection')
        # weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='lagrangian')
        
        self.assert_weights_normalized(weights)
        """
        Step 5. Compute Optimal Action and Param
        """
        mask_to_update = (weights > 0).any(dim=-1)
        weights = weights.to(naction.device)
        noise_optimal = nets["policy"]["noise_pred_net"](
            sample=naction,
            timestep=timestep,
            global_cond=obs_cond
        )
        tweedie_noise_samples = einops.repeat(tweedie_noise_samples, 'b n tp a -> b (n nk) tp a', nk=N_K)

        if mask_to_update.any():
            noise_optimal[mask_to_update] = (tweedie_noise_samples * weights[..., None, None]).sum(dim=1)[mask_to_update]

        naction_next = noise_scheduler.step(
            model_output=noise_optimal,
            timestep=timestep,
            sample=naction,
            eta=eta
        ).prev_sample

        """
        Step 6. Keep the best naction and optimize the param
        """

        prev_timestep = timestep - noise_scheduler.config.num_train_timesteps // noise_scheduler.num_inference_steps
        # More efficient sampling using expand instead of clone + repeat + reshape
        flat_samples = naction_next.reshape(B, Tp, A) # (B*N, Tp, A)
        obs_cond_batch = obs_cond # (N, obs_dim)

        # Full-step denoising
        inner_naction = flat_samples

        if prev_timestep > 0:
            inner_timesteps = noise_scheduler.timesteps[noise_scheduler.timesteps <= prev_timestep]
            
            for inner_t in inner_timesteps:
                inner_naction = self.step_normal(nets, inner_naction, inner_t, obs_cond_batch)
        clean_action_samples = inner_naction  # No need for clone, (B*N, Tp, A)

        # Calculate Tweedie Noise
        clean_action_samples = clean_action_samples
        naction_repeated = naction_next
        tweedie_noise_samples = self.get_noise(
            noisy_samples=naction_repeated,
            original_samples=clean_action_samples,
            timesteps=prev_timestep
        )

        clean_action_samples = clean_action_samples.expand(N, -1, -1)
        tweedie_noise_samples = tweedie_noise_samples.expand(N, -1, -1)
        tweedie_noise_samples = einops.rearrange(tweedie_noise_samples, '(b n) tp a -> b n tp a', b=B, n=N)
        clean_action_samples = einops.rearrange(clean_action_samples, '(b n) tp a -> b n tp a', b=B, n=N)

        """
        Step 7. Parameter Sampling
        """
        cov_param = self.get_param_covariance(timestep)
        param_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(P, device=nparam.device),
            covariance_matrix=cov_param.to(nparam.device)
        )
        param_perturbations = param_dist.sample((B, N, N_K))
        
        # Use expand instead of repeat for nparam
        param_samples = nparam.unsqueeze(1).unsqueeze(2).expand(-1, N, N_K, -1)
        param_samples = param_samples + param_perturbations
        param_samples = torch.clip(param_samples, self.param_lb, self.param_ub)  # paper/gibbs: (-1, 1); same for hq

        """
        Step 8. Evaluate interaction potential
        """
        # Avoid unnecessary copies for evaluation
        clean_actions_eval = clean_action_samples.squeeze(0)
        clean_actions_eval = self.maybe_unnormalize_actions_perf(clean_actions_eval.cpu().numpy())
        clean_actions_eval = clean_actions_eval[:, To-1:]
        param_samples_eval = param_samples.squeeze(0).cpu()

        headplan_scores, objectives, constraints, extraout = self.model_score_fn(
            clean_actions_eval, param_samples_eval, obs_model, goal_model
        )

        # Rest of the function remains largely the same, just removing unnecessary .clone() calls
        headplan_scores = headplan_scores[None, :]  # Add batch dim without clone
        objectives = objectives.unsqueeze(0)  
        constraints = constraints.unsqueeze(0)

        headplan_scores_orig = headplan_scores.copy()
        objectives_orig = objectives.clone()
        constraints_orig = constraints.clone()

        """
        Step 9. Compute Importance Weight
        """

        # Weight computation
        objectives = einops.rearrange(objectives, 'b n nk -> b (n nk)', n=N, nk=N_K)
        constraints = einops.rearrange(constraints, 'b n nk -> b (n nk)', n=N, nk=N_K)
        headplan_scores = torch.from_numpy(-headplan_scores.astype(np.int32)).to(naction.device) + 0.5
        headplan_scores = einops.repeat(headplan_scores, 'b n -> b (n nk)', n=N, nk=N_K)

        # Calculate weights
        weights = torch.zeros_like(objectives)  # Initialize directly
        objectives = objectives - objectives.min(dim=-1, keepdim=True)[0]
        weights = torch.nn.functional.softmax(-objectives/self.temp_mppi, dim=-1)

        # Update weights based on constraints
        weights = self.update_weights_from_constraints(weights, headplan_scores, constraint_mode='rejection')
        max_timestep = noise_scheduler.timesteps[0]
        lag_coeff = 1000 * (1-timestep/max_timestep)
        weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='rejection')
        # weights = self.update_weights_from_constraints(weights, constraints, lag_coeff=lag_coeff, constraint_mode='lagrangian')
        
        self.assert_weights_normalized(weights)
        """
        Step 10. Compute Optimal Param
        """
        mask_to_update = (weights > 0).any(dim=-1)
        weights = weights.to(naction.device)

        if mask_to_update.any():
            param_samples = einops.rearrange(param_samples, 'b n k p -> b (n k) p')
            nparam[mask_to_update] = (param_samples * weights[..., None]).sum(dim=1)[mask_to_update]

        # Logging
        if verbose:
            obj_mean = objectives_orig.mean()
            valid_head_ratio = (headplan_scores_orig).sum()/N
            valid_back_ratio = (constraints_orig<0).sum()/(N*N_K)
            ess = self.effective_sample_size(weights, method="normalized")
            self.disp(f"SIS (t={timestep}) Obj = {obj_mean:.2f} | Head = {valid_head_ratio:.2f} | Back = {valid_back_ratio:.2f} | ESS = {ess:.2f}")

        return naction_next, nparam
