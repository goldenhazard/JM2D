"""Backup planners for PointMaze (was ``safediffusion_d4rl/algo/planner_maze.py``, SafeRobomimic@robomimic 8b2371d2).

``Simple2DPlanner`` is the "HQ" backup of the paper (accelerate for 0.5 s with parameter k_a, then brake to a stop
at t_f = 1 s; zonotope forward-reachable sets from ``jm2d_maze/reachability``). ``Simple2DPlannerUnderAct``
restricts the normalised parameter box to [k_lb, k_ub] (Table 1: x+y+ = [0, 1]^2, x-y- = [-1, 0]^2).
"""
import os
from copy import deepcopy

import numpy as np
import torch
import einops

from scipy.io import loadmat

import matplotlib.pyplot as plt

from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner, FEASIBLE_FLAG
from safediffusion.algo.helper import traj_uniform_acc

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope, batchZonotope, polynomial_zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope, polyZonotope, batchPolyZonotope
import safediffusion.utils.reachability_utils as ReachUtils
from jm2d_maze.tasks.maze import copy_obs_sharing_obstacles

FRS_TENSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "reachability", "frs_tensor_saved", "frs_tensor_mat.mat")

def _tensor_key(x):
    """Exact, hashable key of a tensor's dtype, shape and bytes."""
    x = x.detach()
    return (str(x.dtype), tuple(x.shape), x.cpu().contiguous().numpy().tobytes())


def _zonotope_halfspaces_2d(generators):
    """Half-space normals C and offsets deltaD of a 2-D zonotope with these generators.

    Same operations as ``safediffusion.armtdpy.reachability.conSet.zonotope.polytope`` (dim == 2), whose output is
    PA = [C; -C], Pb = [C @ c + deltaD, -C @ c + deltaD] for the center c; only the center-free part is computed.
    """
    G = torch.clone(generators)
    h = torch.linalg.vector_norm(G,dim=1)
    h_sort, indicies = torch.sort(h,descending=True)
    h_zero = h_sort < 1e-6
    if torch.any(h_zero):
        first_reduce_idx = torch.nonzero(h_zero)[0,0]
        Gunred = G[indicies[:first_reduce_idx]]
        G = Gunred
    C = torch.hstack((-G[:,1:2],G[:,0:1]))
    C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
    index = torch.sum(torch.isnan(C),dim=1) == 0
    C = C[index]
    deltaD = torch.sum(abs(C@generators.T),dim=1)
    return C, deltaD


class Simple2DPlanner(ParameterizedPlanner):
    def __init__(self, **kwargs):
        """
        Tests out the ARMTD-style planner
        """
        # TODO: Make this as an abstract function
        state_dict = {"p_x": 0 , "p_y": 1}
        param_dict = {"k_vx": 0, "k_vy": 1, "k_ax": 2, "k_ay": 3}
        
        # timing
        dt = 0.01
        t_f = 1.0
        
        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # Piecewise trajectory design
        self.time_pieces = [0.5]
        self.offset = self.to_tensor([1, 1])
        assert min(self.time_pieces) > 0
        assert max(self.time_pieces) < t_f

        # loading reachable sets
        self.FRS_tensor_path = FRS_TENSOR_PATH
        self.FRS_tensor, self.FRS_info = self.preload_frs(dtype = self.dtype, device=self.device)
        self.combs = self.generate_combinations_upto(200)

        # useful var
        self.state_key = "flat"

        # trajectory optimization
        self.opt_dim     = [2, 3]
        self.weight_dict = dict(goal = 0.0, projection = 0.0)

        # control conservatism
        PAD_RADIUS         = 0.025
        self.robot_padding = ReachUtils.get_zonotope_from_sphere_geom(pos  = np.zeros(3,), 
                                                                      rot  = np.eye(3), 
                                                                      size = [PAD_RADIUS])
        self.robot_padding = self.robot_padding.to(dtype = self.dtype, device = self.device)
        
        self.use_reachable_scene_filter = kwargs["trajopt"]["use_reachable_scene_filter"]
        self.reachable_radius           = kwargs["trajopt"]["reachable_scene_radius"]
        if self.use_reachable_scene_filter:
            self.disp(f"[CAUTION] Using reachable scene filter with radius: {self.reachable_radius}")

    # ------------------------------------------------------------------------------------------- #
    # --------------------- Define the Planning Model ------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    def get_pstate_from_obs(self, obs_dict):
        """
        Define a map from the observation dictionary to the planning state

        For Maze2D, the planning state is the position of the robot
        """
        p = obs_dict[self.state_key][:2]
        
        return p
    
    def get_pparam_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Define a map from the observation dictionary and the optimized trajectory parameter to the trajectory parameter

        For Maze2D, the trajectory parameter is the initial velocity and acceleration of the robot
        """
        v0 = obs_dict[self.state_key][2:]
        param = torch.hstack([v0, k_opt])
        
        return param
    
    def model(self, t, p0, param, return_gradient=False):
        """
        Get x(t; x0, k), given the initial state x0, trajectory parameter k.

        Args:
            t :    (T,),   time
            p0:    (..., 2), initial planning state, (p_x, p_y)
            param: (..., 4), initial planning param, (k_vx, k_vy, k_ax, k_ay)
        
        Returns:
            x:  (..., T, 2), desired trajectory
            dx: (..., T, 2), desired velocity

        NOTE: This model should be consistent with the model defined at MATLAB CORA
        """
        assert t.max() <= self.t_f
        t = t.to(device = p0.device, dtype = p0.dtype)

        if p0.ndim == 1:
            p0 = p0[None, :]
        if param.ndim == 1: 
            param = param[None, :]

        batch_shape = p0.shape[:-1]
        N = t.shape[0]

        x  = torch.zeros((*batch_shape, N, self.n_state), dtype=p0.dtype, device=p0.device)
        dx = torch.zeros((*batch_shape, N, self.n_state), dtype=p0.dtype, device=p0.device)

        # parse parameter
        k_v = param[..., [self.param_dict["k_vx"], self.param_dict["k_vy"]]]
        k_a = param[..., [self.param_dict["k_ax"], self.param_dict["k_ay"]]]

        t1 = self.time_pieces[0]

        mask_first = t <= t1
        mask_second = t > t1

        x1, dx1 = traj_uniform_acc(t[mask_first], p0, k_v, k_a)

        p1 = p0 + k_v*t1 + 0.5*k_a*t1**2
        v1 = k_v + k_a*t1
        a1 = -v1/(self.t_f-t1)
        x2, dx2 = traj_uniform_acc(t[mask_second]-t1,  p1, v1, a1)

        x[..., mask_first, :] = x1
        x[..., mask_second, :] = x2
        dx[..., mask_first, :] = dx1
        dx[..., mask_second, :] = dx2

        if return_gradient:
            grad_to_optvar = torch.zeros((*batch_shape, N, self.n_state, len(self.opt_dim)), dtype = p0.dtype, device = p0.device)
            grad_first_piece  = 0.5*(t[mask_first])**2
            grad_second_piece = 0.5*t1**2 + t1*(t[mask_second]-t1) + 0.5 * (-t1/(self.t_f-t1)) * (t[mask_second]-t1)**2
            grad_to_optvar[..., mask_first, :, :] = grad_first_piece[:, None, None]*torch.eye(self.n_state, device=p0.device)
            grad_to_optvar[..., mask_second, :, :] = grad_second_piece[:, None, None]*torch.eye(self.n_state, device=p0.device)

            return x, dx, grad_to_optvar
        
        else:
            return x, dx
        
    def get_action_from_plan_at_t(self, t_des, plan, obs_dict):
        """
        Plan to the action representation

        The action configuration is the normalized delta joint angle.

        Args:
            t_des (float): the query time to retrieve the plan, desired joint angle
            plan (ReferenceTrajectory): the plan to track
            obs_dict (dict): this dictionary includes the current state information
        
        Returns:
            action (float): the normalized delta joint angle
        """
        # Kp = 30
        # Kd = 0.5
        
        Kp = 36
        Kd = 12

        x = obs_dict["flat"][-1, :2]
        dx = obs_dict["flat"][-1, 2:]

        plan_idx = int(t_des/self.dt_plan)
        
        x_des = plan.x_des[plan_idx].cpu().numpy()
        dx_des = plan.dx_des[plan_idx].cpu().numpy()
        action = Kp*(x_des - x) - Kd*(dx-dx_des)
        
        pos_err = np.abs(x_des - x)
        
        # self.disp(f"pos_err: {pos_err}")
        
        return action
        
    # ------------------------------------------------------------------------------------------- #
    # --------------------------------- Trajectory Optimization --------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # @time_profile
    def _prepare_problem_data(self, obs_dict, goal_dict, random_initialization = False):
        """
        TODO: Setup the useful variables for the trajectory optimization
        """
        # prepare the problem data
        problem_data = {}

        # prepare the meta-data
        problem_data["meta"] = {}
        problem_data["constraint"] = {}
        problem_data["objective"]  = {}
        
        problem_data["meta"]["n_optvar"]      = len(self.opt_dim)
        problem_data["meta"]["n_constraint"]  = 0
        problem_data["meta"]["state"]         = self.to_tensor(obs_dict[self.state_key])

        if random_initialization:
            problem_data["meta"]["optvar_0"]  = self.to_tensor((torch.rand(len(self.opt_dim))-0.5)*2) # initial point
        else:
            problem_data["meta"]["optvar_0"]  = self.to_tensor(torch.zeros(len(self.opt_dim)))

        # prepare the data for constraints
        
        # --------------------------------------------------------------------------------- #
        # ------------------(C-1) Collision Constraints ----------------------------------- #
        # --------------------------------------------------------------------------------- #
        obstacles = self.get_obstacles_for_collision_check(obs_dict)
        n_interval = self.FRS_info["n_reachable_set"]

        # (C-1-1) Get the forward occupancy of the robot given the current position and velocity
        # NOTE: position is already included in robot zonotope)
        # NOTE: test this function using test_forward_occupancy(self, FO, pos_0, vel_0, acc_0)
        num_robot_zonotope    = len(obs_dict["zonotope"]["robot"])
        num_obstacle_zonotope = len(obstacles)
        
        FO = self.forward_occupancy(robot_zonotope=obs_dict["zonotope"]["robot"], vel=obs_dict[self.state_key][2:])
        A  = np.zeros((num_robot_zonotope, num_obstacle_zonotope), dtype=object)
        b  = np.zeros((num_robot_zonotope, num_obstacle_zonotope), dtype=object)

        # (C-1-2) Buffer the obstacle zonotope with the non-optvar generators
        # NOTE: deleteZerosGenerators() is used but not sure if it actually works
        
        grest_key = _tensor_key(FO.Grest)
        for (i, obstacle) in enumerate(obstacles):
            obstacle = obstacle.project().deleteZerosGenerators()
            A_o, b_o = self._buffered_obstacle_polytope(obstacle.Z, FO.Grest, n_interval, grest_key)
            A[0, i]  = A_o
            b[0, i]  = b_o

        problem_data["constraint"]["collision"]       = {}
        problem_data["constraint"]["collision"]["A"]  = A
        problem_data["constraint"]["collision"]["b"]  = b
        problem_data["constraint"]["collision"]["FO"] = [FO]

        problem_data["meta"]["n_constraint"] += num_robot_zonotope*num_obstacle_zonotope*n_interval
        # --------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------- #
        # ------------------(C-2) Velocity Constraints ------------------------------------ #
        # --------------------------------------------------------------------------------- #
        v_dim = [self.param_dict["k_vx"], self.param_dict["k_vy"]]
        problem_data["constraint"]["velocity"]          = {}
        problem_data["constraint"]["velocity"]["v_max"] = self.FRS_info["delta_k"][v_dim]
        problem_data["constraint"]["velocity"]["v_min"] = -self.FRS_info["delta_k"][v_dim]
            
        problem_data["meta"]["n_constraint"] += 2 * len(v_dim)


        # ---------------- Objectives are optional               ------------------------- #
        # -------------------------------------------------------------------------------- #
        # ------------------(J-1) Objective ---------------------------------------------- #
        #--------------------------------------------------------------------------------- #
        if "goal" in self.weight_dict.keys() and self.weight_dict["goal"] > 0:
            problem_data["objective"]["goal"] = {}
            problem_data["objective"]["goal"]["target"] = self.to_tensor(goal_dict[self.state_key][0:2])

        # -------------------------------------------------------------------------------- #
        # ------------------(J-2) Objective ---------------------------------------------- #
        #--------------------------------------------------------------------------------- #
        if "projection" in self.weight_dict.keys() and self.weight_dict["projection"]:
            problem_data["objective"]["projection"] = {}
            problem_data["objective"]["projection"]["reference_trajectory"] = goal_dict["reference_trajectory"]

        return problem_data
    
    def _buffered_obstacle_polytope(self, obstacle_Z, Grest, n_interval, grest_key=None):
        """H-representation (A, b) of the obstacle buffered by the k-independent FO generators ``Grest``.

        Speed-up (release, exact): (A, b) only depend on the obstacle and on ``Grest``, which is the same for every
        DP candidate, denoising step and replanning cycle (it holds the FRS error generators and the robot
        zonotope; the candidate enters through the sliced center / dependent generators). The original code
        recomputed it N x I times per replan; it is now memoised on the exact bytes of its inputs, so a cache hit
        returns the tensors the original computation produces for these inputs (tests/test_speedups.py).
        """
        key = (n_interval, _tensor_key(obstacle_Z), grest_key or _tensor_key(Grest))
        cache = self.__dict__.setdefault("_obstacle_polytope_cache", {})
        if key not in cache:
            obs_Z    = einops.repeat(obstacle_Z, 'm n -> repeat m n', repeat=n_interval)
            A_o, b_o = batchZonotope(torch.cat((obs_Z, Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
            if len(cache) >= 4096:
                cache.clear()
            cache[key] = (A_o.cpu(), b_o.cpu())
        return cache[key]

    # @time_profile
    def _compute_constraints_batch(self, k, problem_data, use_gpu = False):
        """
        Get C(k; problem_data) <= 0

        Args
            k            : (np.array) size of (Bp, B, opt_dima)
            problem_data : (List <dict>) size of (Bp)

        Returns:
            g (torch.tensor) (N, B, n_constraint)

        NOTE: the k is the normalized one
        TODO: this evaluation should be done on GPU
        """
        # Input Argument Checking
        assert k.shape[-1] == len(self.opt_dim) and abs(k).max() <= 1, "k should be normalized to [-1, 1]"
        assert isinstance(problem_data, list) and isinstance(problem_data[0], dict), "problem_data should be list of dict"
        assert len(problem_data) == k.shape[0], "The first dimension of k and length of problem_data should be same."
        
        meta_data = [p["meta"] for p in problem_data]
        
        BP = len(problem_data)
        B  = k.shape[1]
        M  = meta_data[0]["n_constraint"]
        T  = self.FRS_info["n_reachable_set"]
        D  = len(self.opt_dim)

        if use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        Cons = torch.zeros(BP, B, M, dtype = self.dtype).to(device = device)
        # Jac  = torch.zeros(BP, B, M, D, dtype = self.dtype).to(device = device)

        beta = einops.repeat(k, 'BP B D -> BP B repeat D', repeat = T).to(device = device)  # (BP, B, T, D)
        k    = self.unnormalize_k(k).to(device = device)

        # ------------------(C-1) Collision Constraints --------------------- #
        collision_data = [p["constraint"]["collision"] for p in problem_data]

        # A = torch.zeros()
        # for j in range(n_robot):
        #     for o in range(n_obstacle):
        #         A[j][o] = p

        # collision constraints
        # A  = collision_data["A"]                                        # (Bp, T, M, n_workspace)
        # b  = collision_data["b"]                                        # (T, M)
        # FO = collision_data["FO"]

        (n_robot, n_obstacle) = collision_data[0]['A'].shape

        for j in range(n_robot):
            FO_j = [p["FO"][j].to(device=device) for p in collision_data]
            FO_j = [FO.repeat(B) for FO in FO_j]
            FO_j = batchPolyZonotope(Z          = torch.concat([FO.Z.unsqueeze(0) for FO in FO_j]), 
                                     n_dep_gens = FO_j[0].n_dep_gens,
                                     expMat     = FO_j[0].expMat,
                                     id         = FO_j[0].id, 
                                     compress   = 0)                           # (BP, B, T, NG, NW)
            
            c_k  = FO_j.center_slice_all_dep(beta).unsqueeze(-1)               # (BP, B, T, NW, 1)
            
            for o in range(n_obstacle):
                # constraint
                A_jo = torch.concat([p['A'][j][o].unsqueeze(0) for p in collision_data]).to(device=device) # (Bp, T, M, n_workspace)
                b_jo = torch.concat([p['b'][j][o].unsqueeze(0) for p in collision_data]).to(device=device) # (Bp, T, M,)
                
                A_batch   = einops.repeat(A_jo, 'Bp T M N -> Bp repeat T M N', repeat=B) # (Bp, B, T, M, NW)
                b_batch   = einops.repeat(b_jo, 'Bp T M -> Bp repeat T M', repeat=B)
                h_obs     = (A_batch @ c_k).squeeze(-1) - b_batch                        # (Bp, B, T, M)
                h_obs     = h_obs.nan_to_num(-torch.inf)
                cons, ind = torch.max(h_obs, -1)                                         # (Bp, B, T)
                Cons[:, :, (j+n_robot*o)*T:(j+n_robot*o+1)*T] = -cons

                # gradient of the constraint
                # A_max     = A[j][o].gather(-2, ind.reshape(T, 1, 1).repeat(1, 1, ndim))
                # grad_cons = (A_max@grad_c_k).squeeze(-2)
                # Jac[(j+n_robot*o)*T: (j+n_robot*o+1)*T] = -grad_cons
        
        # ------------------(C-2) Velocity Constraints --------------------- #
        # x0 + v0*t <= v_ub, x0 + v0*t >= v_lb
        data = [p["constraint"]["velocity"] for p in problem_data]
        x0   = torch.concat([p["state"].unsqueeze(0) for p in meta_data])              # (Bp, N_state)
        v0   = self.to_tensor(x0[:, 2:]).to(device = device)
        v0   = einops.repeat(v0, "Bp N -> Bp repeat N", repeat = B)     # (Bp, B, 2)
        
        v_max = data[0]["v_max"].to(device=device)
        v_min = data[0]["v_min"].to(device=device)
        t_pk  = self.time_pieces[0]

        vpk  = v0 + t_pk * k                            # (Bp, B, 2)
        cons = torch.concat((vpk - v_max, v_min - vpk), dim = -1) # (Bp, B, 4)
        Cons[:, :, (n_robot*n_obstacle)*T:] = cons

        Cons = Cons.cpu()

        # jac_vpk_to_k    = t_pk * torch.eye(2)
        # jac_k_to_beta   = torch.diag(self.FRS_info["delta_k"][self.opt_dim])
        # jac_vpk_to_beta = jac_vpk_to_k @ jac_k_to_beta
        # jac  = torch.vstack((jac_vpk_to_beta, -jac_vpk_to_beta))
        # Jac[(n_robot*n_obstacle)*T:]  = jac

        return Cons
    
    # @time_profile
    def _compute_constraints(self, k, problem_data):
        """
        Get C(k; problem_data) <= 0

        Args
            k: (n_opt_var, ), trajectory parameter, numpy array

        NOTE: the k is the normalized one
        """
        assert k.shape[0] == len(self.opt_dim) and k.ndim == 1

        meta_data = problem_data["meta"]
        M  = meta_data["n_constraint"]
        n  = meta_data["n_optvar"]
        x0 = meta_data["state"]
        T  = self.FRS_info["n_reachable_set"]
        

        # ------------------(C-0) Initial Constraints --------------------- #
        Cons = torch.zeros(M, dtype=self.dtype)
        Jac  = torch.zeros(M, n, dtype=self.dtype)


        # k_orig           = k.clone()
        k1               = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]
        k2               = self.unnormalize_k(k)
        assert torch.allclose(k1, k2), "The transformation is not correct"
        k    = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]
        beta = (k-self.FRS_info["c_k"][self.opt_dim])/self.FRS_info["delta_k"][self.opt_dim]
        beta = einops.repeat(beta, 'n -> repeat n', repeat=T)

        # ------------------(C-1) Collision Constraints --------------------- #
        collision_data = problem_data["constraint"]["collision"]

        # collision constraints
        A  = collision_data["A"]                                        # (200, n_constraint, n_workspace)
        b  = collision_data["b"]                                        # (200, n_constraint, 1)
        FO = collision_data["FO"]
        ndim = FO[0].dimension                                          # dimension of the workspace

        (n_robot, n_obstacle) = A.shape
        for j in range(n_robot):
            c_k      = FO[j].center_slice_all_dep(beta).unsqueeze(-1)   # (200, n_workspace, 1)
            grad_c_k = FO[j].grad_center_slice_all_dep(beta)            # (200, n_workspace, n_optvar)
            for o in range(n_obstacle):
                # max(Ax - b) <= 0
                h_obs     = (A[j][o]@c_k).squeeze(-1) - b[j][o]         # (200, n_constraint)
                h_obs     = h_obs.nan_to_num(-torch.inf)                # (we need to handle nan values to be meaningless)
                cons, ind = torch.max(h_obs, -1)                        # (200, 1)
                
                # gradient of the constraint
                A_max     = A[j][o].gather(-2, ind.reshape(T, 1, 1).repeat(1, 1, ndim))
                grad_cons = (A_max@grad_c_k).squeeze(-2)

                Cons[(j+n_robot*o)*T:(j+n_robot*o+1)*T] = -cons
                Jac[(j+n_robot*o)*T: (j+n_robot*o+1)*T] = -grad_cons

        # ------------------(C-2) Velocity Constraints --------------------- #
        # x0 + v0*t <= v_ub, x0 + v0*t >= v_lb
        data = problem_data["constraint"]["velocity"]

        v0    = self.to_tensor(x0[2:])
        v_max = data["v_max"]
        v_min = data["v_min"]
        t_pk  = self.time_pieces[0]

        vpk             = v0 + t_pk * k
        jac_vpk_to_k    = t_pk * torch.eye(2)
        jac_k_to_beta   = torch.diag(self.FRS_info["delta_k"][self.opt_dim])
        jac_vpk_to_beta = jac_vpk_to_k @ jac_k_to_beta

        cons = torch.hstack((vpk - v_max, v_min - vpk))
        jac  = torch.vstack((jac_vpk_to_beta, -jac_vpk_to_beta))

        Cons[(n_robot*n_obstacle)*T:] = cons
        Jac[(n_robot*n_obstacle)*T:]  = jac
        
        Cons = Cons.cpu().numpy()
        Jac  = Jac.cpu().numpy()
        
        return (Cons, Jac)
    
    # @time_profile
    def process_observation_and_goal_for_TO(self, obs_dict, goal_dict=None):
        """
        Process the observation and goal for the trajectory optimization
        """

        obs_dict = copy_obs_sharing_obstacles(obs_dict)   # was deepcopy(obs_dict); obstacles are read-only
        obs_dict[self.state_key] = self.to_tensor(obs_dict[self.state_key])
        obs_dict[self.state_key][:2] += self.offset

        obs_dict["zonotope"]["robot"][0] = obs_dict["zonotope"]["robot"][0] + self.robot_padding

        if goal_dict is not None:
            goal_dict = deepcopy(goal_dict)
            goal_dict[self.state_key] = self.to_tensor(goal_dict[self.state_key])
            goal_dict[self.state_key][:2] += self.offset

            if "reference_trajectory" in goal_dict.keys():
                goal_dict["reference_trajectory"].x_des += self.offset

        return obs_dict, goal_dict
    
    def postprocess_plan(self, plan):
        """
        The reference trajectory model is defined in the workspace.
        This function shifts the plan to the robot joint space
        """
        plan.x_des = plan.x_des - self.offset

        return plan
    
    # @time_profile
    def _compute_objective_batch(self, k, problem_data, use_gpu = False):
        """
        Get J(k; problem_data)

        Args:
            k : normalized trajectory parameter, numpy array, this corresponds to the indeterminate of the generators
        Returns:
            Obj (torch.tensor) objective function value
        """
        # Input Argument Checking
        assert k.shape[-1] == len(self.opt_dim) and abs(k).max() <= 1, "k should be normalized to [-1, 1]"
        assert isinstance(problem_data, list) and isinstance(problem_data[0], dict), "problem_data should be list of dict"
        assert len(problem_data) == k.shape[0], "The first dimension of k and length of problem_data should be same."

        meta_data = [p["meta"] for p in problem_data]
        
        BP = len(problem_data)
        B  = k.shape[1]
        D  = len(self.opt_dim)

        if use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        k    = self.unnormalize_k(k).to(device = device)

        # parse the initial condition of the backup plan
        states     = torch.concat([d["state"].unsqueeze(0) for d in meta_data]).to(device=device)
        states     = einops.repeat(states, "BP D -> BP repeat D", repeat = B)
        pos0       = states[:, :, :2]
        v0         = states[:, :, 2:] # (BP, B, 2)

        # (1) Projection Objective
        # The objective tries to minimize the distance between the desired trajectory and the backup plan

        cost = {}
        grad_cost = {}

        # (1-1) parse the required data
        if "projection" in self.weight_dict.keys() and self.weight_dict["projection"] > 0:
            desired_plan_orig = [p["objective"]["projection"]["reference_trajectory"] for p in problem_data]
            t_des = desired_plan_orig[0].t_des
            mask = t_des <= self.t_f
            pos_des = torch.concat([p[mask].x_des.unsqueeze(0) for p in desired_plan_orig]) # (B, T, D)
            pos_des = einops.repeat(pos_des, "BP T D -> BP repeat T D", repeat=B).to(device=device)

            # (1-2) compute the cost and the gradient
            traj_param = torch.concat([v0, k], dim=-1)
            pos, _, gradpos_to_k   = self.model(t_des, 
                                                pos0, 
                                                traj_param, 
                                                return_gradient=True) # (BP, B, T, D)

            pos_err                = pos - pos_des

            cost_proj              = torch.sum(torch.mean(pos_err**2, dim=-2), dim=-1)                     

            cost["projection"] = cost_proj.cpu().numpy()

        # (2) Goal objective
        # The objective tries to minimize the distance between the final position and the goal
        if "goal" in self.weight_dict.keys() and self.weight_dict["goal"] > 0:
            goal = torch.concat([d["objective"]["goal"]["target"].unsqueeze(0) for d in problem_data])
            goal = einops.repeat(goal, "BP D -> BP repeat D", repeat=B).to(device=device) # (BP, B, 2)

            traj_param = torch.concat([v0, k], dim=-1)
            pos, _, gradpos_to_k   = self.model(torch.tensor([0, self.t_f]), 
                                                pos0, 
                                                traj_param, 
                                                return_gradient=True) # (BP, B, T, 2)
            
            pos_last               = pos[:, :, -1, :] # (BP, B, 2)

            goal_err               = pos_last - goal
            cost_goal              = torch.sum(goal_err**2, dim=-1)

            cost["goal"] = cost_goal.cpu().numpy()

        # weighting
        total_cost = torch.zeros(BP, B)
        for key in cost.keys():
            total_cost += cost[key] * self.weight_dict[key]

        # detaching
        Obj = total_cost
        
        assert Obj.min() >= 0, "Objective should be non-negative"

        return Obj
    
    # @time_profile
    def _compute_objective(self, k, problem_data):
        """
        Get J(k; problem_data)

        Args:
            k : normalized trajectory parameter, numpy array, this corresponds to the indeterminate of the generators
        Returns:
            Obj: objective function value
            Grad: gradient of the objective function
        """
        assert k.shape[0] == len(self.opt_dim) and k.ndim == 1

        # parse the initial condition of the backup plan
        meta_data  = problem_data["meta"]
        pos0       = meta_data["state"][:2]
        v0         = meta_data["state"][2:] 

        # transform the trajectory parameter to the original domain
        # k_orig           = k.clone()
        # k1               = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]
        # k2               = self.unnormalize_k(k)
        # assert torch.allclose(k1, k2), "The transformation is not correct"
        k                = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]
        k                = self.to_tensor(k)
        grad_k_to_beta   = torch.diag(self.FRS_info["delta_k"][self.opt_dim])


        # (1) Projection Objective
        # The objective tries to minimize the distance between the desired trajectory and the backup plan

        cost = {}
        grad_cost = {}

        # (1-1) parse the required data
        if "projection" in self.weight_dict.keys() and self.weight_dict["projection"] > 0:
            data = problem_data["objective"]["projection"]
            desired_plan_orig = data["reference_trajectory"]

            # the desired plan only projects the plan within horizon
            desired_plan = desired_plan_orig[desired_plan_orig.t_des <= self.t_f] 

            # (1-2) compute the cost and the gradient
            pos, _, gradpos_to_k   = self.model(desired_plan.t_des, 
                                                pos0, 
                                                torch.cat([v0, k]), 
                                                return_gradient=True) # (n_timestep, n_pos, n_optvar)
            pos                    = pos[0]                                                            # (n_timestep, n_pos)
            gradpos_to_k           = gradpos_to_k[0]                                                   # (n_timestep, n_pos, n_optvar)

            pos_err                = pos - desired_plan.x_des
            grad_pos_err_to_k      = gradpos_to_k

            cost_proj              = torch.sum(torch.mean(pos_err**2, dim=0))                     
            grad_cost_proj_to_k    = 2*torch.bmm(pos_err.unsqueeze(-2), grad_pos_err_to_k).squeeze(-2) # (n_timestep, n_optvar)
            grad_cost_proj_to_k    = torch.mean(grad_cost_proj_to_k, dim=0)                            # (n_optvar,)
            grad_cost_proj_to_beta = grad_cost_proj_to_k @ grad_k_to_beta                              # (n_timestep, n_optvar)

            cost["projection"] = cost_proj
            grad_cost["projection"] = grad_cost_proj_to_beta

        # (2) Goal objective
        # The objective tries to minimize the distance between the final position and the goal
        if "goal" in self.weight_dict.keys() and self.weight_dict["goal"] > 0:
            data = problem_data["objective"]["goal"]
            goal = data["target"]

            pos, _, gradpos_to_k   = self.model(torch.tensor([0, self.t_f]), 
                                                pos0, 
                                                torch.cat([v0, k]), 
                                                return_gradient=True)
            
            pos_last               = pos[0][-1]
            gradpos_to_k_last      = gradpos_to_k[0][-1]

            goal_err               = pos_last - goal
            cost_goal              = torch.sum(goal_err**2)

            grad_cost_goal_to_k    = (2*goal_err.unsqueeze(-2) @ gradpos_to_k_last).squeeze(-2)
            grad_cost_goal_to_beta = grad_cost_goal_to_k @ grad_k_to_beta

            cost["goal"] = cost_goal
            grad_cost["goal"] = grad_cost_goal_to_beta

        # weighting
        total_cost = self.to_tensor(0)
        total_grad = self.to_tensor(torch.zeros_like(k))
        for key in cost.keys():
            total_cost += cost[key] * self.weight_dict[key]
            total_grad += grad_cost[key] *self.weight_dict[key]

        # detaching
        Obj = total_cost.cpu().numpy()
        Grad = total_grad.cpu().numpy()

        return (Obj, Grad)

    # --------------------- Reachability Helper ----------------------- #
    def preload_frs(self, dtype=torch.float, device=torch.device('cpu')):
        """
        Load the forward-reachable-set pre-computed using CORA toolbox.

        It also checks if the FRS is consistent with the planning model specification.

        Returns
            FRS_tensor: (n_reachable_set, n_generators, n_dim), n_dim = n_state + n_param + n_time(1)
            FRS_info: Assumes that the trajectory parameter resides in the `box`
                dt: time step of the reachability analysis
                c_k: center of the box
                delta_k: half size of the box
        """
        FRS_tensor = []
        FRS_tensor_load = loadmat(self.FRS_tensor_path)

        # assert if the reachability setting is consistent with the planning model specification
        assert(FRS_tensor_load["t_total"].squeeze() == self.t_f)
        assert(FRS_tensor_load["t_plan"].squeeze() == self.time_pieces[0])

        # This assumes that the domain of trajectory parameter is `box`-zonotope
        c_k      = torch.zeros(self.n_param,)
        c_k[[self.param_dict["k_vx"], self.param_dict["k_vy"]]] = torch.tensor(FRS_tensor_load["c_kv"].squeeze(), 
                                                                               dtype=dtype, device=device)
        c_k[[self.param_dict["k_ax"], self.param_dict["k_ay"]]] = torch.tensor(FRS_tensor_load["c_ka"].squeeze(), 
                                                                               dtype=dtype, device=device)

        delta_k  = torch.zeros(self.n_param,)
        delta_k[[self.param_dict["k_vx"], self.param_dict["k_vy"]]] = torch.tensor(FRS_tensor_load["delta_kv"].squeeze(),
                                                                                    device=device, dtype=dtype)
        delta_k[[self.param_dict["k_ax"], self.param_dict["k_ay"]]] = torch.tensor(FRS_tensor_load["delta_ka"].squeeze(),
                                                                                    device=device, dtype=dtype)
    
        FRS_tensor = torch.tensor(FRS_tensor_load["FRS_tensor"], dtype=dtype, device=device)

        # save the configuration for reachability computation
        FRS_info = {}
        FRS_info["dt"]      = torch.tensor(FRS_tensor_load["dt"].squeeze(), device=device, dtype=dtype)
        FRS_info["c_k"]     = c_k.to(device, dtype)
        FRS_info["delta_k"] = delta_k.to(device, dtype)
        FRS_info["n_reachable_set"] = FRS_tensor.shape[0]

        return FRS_tensor, FRS_info
    
    def forward_occupancy(self, robot_zonotope, vel):
        """
        Compute the forward occupancy at R^2 given the current position and velocity

        Args
            robot_zonotope: (n_body,) zonotopes of R^2
            vel: (2,) initial velocity

        Returns
            FRS: (n_body, n_timestep, n_c + n_g, 2) polynomial zonotope, that is k_ax, k_ay-sliceable
        """
        # Shift the FRS to the current position (pos)
        FRS_batch_zono = batchZonotope(self.FRS_tensor)

        # Slice with respect to the current velocity (vel)
        slice_dim = [self.n_state + self.param_dict["k_vx"], self.n_state + self.param_dict["k_vy"]]
        slice_pt  = vel.tolist()
        FRS_batch_zono = FRS_batch_zono.slice(slice_dim, slice_pt)

        # Make it to the polyzonotope, just to manage the id
        id_dim = [self.n_state + self.param_dict["k_ax"], self.n_state + self.param_dict["k_ay"]]
        PZ_FRS = FRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope_multi_dim(id_dim, "k_a")

        # now we can drop the dimension we do not need
        PZ_FRS_2D         = PZ_FRS.project([0, 1])
        PZ_FRS_2D         = PZ_FRS_2D.deleteZerosGenerators()
        robot_zonotope_2D = robot_zonotope[0].project([0, 1])
        robot_zonotope_2D = robot_zonotope_2D.deleteZerosGenerators()
        
        PZ_FRS_2D         = PZ_FRS_2D + robot_zonotope_2D
        PZ_FRS_2D         = PZ_FRS_2D.deleteZerosGenerators()

        return PZ_FRS_2D
    
    # ------------------------------------- Helper Functions ------------------------------------- #
    def obs_dict_to_collision_center(self, obs_dict):
        """
        Args:
            obs_dict (Dict)
        Returns:
            collision_center (np.array)
        """
        collision_center = obs_dict["flat"][-1][:2] + np.array(self.offset)
        return collision_center
    
    def get_obstacles_for_collision_check(self, obs_dict):
        """
        Get the obstacles nearby the center with the radius

        obs_dict
        
        Args:
            obstacles (list): the list of zonotope representation of the obstacles
            obs_dict (dict): the observation dictionary
        
        Returns:
            obstacles_nearby (list): the list of zonotope representation of the obstacles nearby

        TODO: We should compute the reachable sphere first based on the qpos, qgripper and then filter the obstacles
        """
        obstacles = [o.project([0, 1]) for o in obs_dict["zonotope"]["obstacle"]]
        
        if self.use_reachable_scene_filter:
        # get the center and radius of the obstacles
            obstacle_centers = torch.vstack([obstacle.center for obstacle in obstacles])
            
            if "collision_center" in obs_dict.keys():
                collision_center = obs_dict["collision_center"]
            else:
                collision_center = obs_dict["zonotope"]["robot"][0].center[:2]

            obstacles_nearby_idx = torch.norm(obstacle_centers - collision_center, dim=1) <= self.reachable_radius

            obstacles_reachable = [obstacle for i, obstacle in enumerate(obstacles) if obstacles_nearby_idx[i]]
        
        else:    
            obstacles_reachable = obstacles

        return obstacles_reachable
    
    def collision_check(self, plan, obs):
        """
        Discrete collision checking algorithm.

        It checks the collision by intersecting the robot zonotopes at each planning time steps with
        the obstacles zonotopes.

        Args
            plan: ReferenceTrajectory torch array (qpos)
            obs: observation dictionary

        Returns
            is_collision: bool

        Speed-up (release, exact): the robot zonotope only translates along the plan, so the half-space
        normals C and offsets deltaD of ``obstacle - robot_i`` are the same for every waypoint i. They are
        computed once per obstacle (same code as ``zonotope.polytope``) and d_i = C @ c_i is evaluated for all
        waypoints in one product. C, deltaD and every Pb_i are bitwise equal to the original per-(waypoint, obstacle)
        loop (``_collision_check_loop``, still used with ``USE_ZONOPY=true``); tests/test_speedups.py checks this.
        """
        assert isinstance(plan, ReferenceTrajectory), "The plan should be the instance of ReferenceTrajectory"
        assert "zonotope" in obs.keys() and "robot" in obs["zonotope"].keys() and "obstacle" in obs["zonotope"].keys()
        robot_zonotope = obs["zonotope"]["robot"][0].project([0, 1])
        assert torch.allclose(robot_zonotope.center, plan[0][1]+self.offset, 1e-6), "The plan is not consistent with the current position"

        obstacle_zonos = self.get_obstacles_for_collision_check(obs)

        if use_zonopy:
            return self._collision_check_loop(plan, robot_zonotope, obstacle_zonos)

        # centers of robot_zonotope + plan[i][1] - plan[0][1] (zonotope '+'/'-' with a tensor: Z[0] += / -= in place)
        robot_centers = robot_zonotope.center.unsqueeze(0).repeat(len(plan), 1)
        robot_centers += plan.x_des
        robot_centers -= plan.x_des[0]

        for obstacle_zono in obstacle_zonos:
            # buff_i = obstacle_zono - robot_zonotope_i: center obstacle.c - robot_i.c, generators [obstacle.G; robot.G]
            generators = torch.vstack((obstacle_zono.generators, robot_zonotope.generators))
            centers = obstacle_zono.center - robot_centers
            C, deltaD = _zonotope_halfspaces_2d(generators)
            d = centers @ C.T                                    # row i == C @ c_i
            b = torch.cat((d+deltaD,-d+deltaD), dim=-1)          # row i == Pb of buff_i
            unsafe = b.min(dim=-1)[0] > 1e-6
            if unsafe.any():
                return True

        return False

    def _collision_check_loop(self, plan, robot_zonotope, obstacle_zonos):
        """Original collision check (one zonotope polytope per waypoint and obstacle)."""
        # checks if the robot at each time step is in collision with the obstacles
        for i in range(len(plan)):
            robot_zonotope_i = robot_zonotope + plan[i][1] - plan[0][1]
            for obstacle_zono in obstacle_zonos:
                buff = obstacle_zono - robot_zonotope_i
                _, b = buff.polytope(self.combs)
                unsafe = b.min(dim=-1)[0] > 1e-6
                if unsafe:
                    return True

        return False
    
    def __call__(self, obs_dict, goal_dict, random_initialization=False):
        """
        Plan the trajectory given the observation and the goal

        Args
            obs_dict: observation dictionary
            goal_dict: goal dictionary
            random_initialization: bool, whether to initialize the trajectory parameter randomly

        Returns
            plan: ReferenceTrajectory object
            actions: (n_timesteps, n_action) np.ndarray
        """
        plan, info = super().__call__(obs_dict, goal_dict, random_initialization=random_initialization)

        if info["status"] in FEASIBLE_FLAG:
            k_opt    = plan.get_trajectory_parameter()
            self.FRS = self.get_FRS_from_obs_and_optvar(obs_dict, k_opt[self.opt_dim])

        return plan, info

    def get_FRS_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Get the FRS given the optimized trajectory parameter
        """
        FRS = self.forward_occupancy(obs_dict["zonotope"]["robot"], obs_dict[self.state_key][2:])
        k_opt = k_opt/self.FRS_info["delta_k"][self.opt_dim]
        k_opt = einops.repeat(k_opt, 'n -> repeat n', repeat=FRS.Z.shape[0])
        FRS = FRS.slice_all_dep(k_opt)
        
        return FRS
    
    def test_forward_occupancy(self, FO, pos_0, vel_0, acc_0):
        """
        Args
            FO: (B, n_g, 2)
        """
        init_state = torch.tensor(pos_0, dtype=self.dtype, device=self.device)
        param      = torch.cat([torch.tensor(vel_0), torch.tensor(acc_0)]).to(dtype=self.dtype, device=self.device)
        (x_des, dx_des) = self.model(self.t_des, init_state, param)

        ka = torch.tensor(acc_0, dtype=self.dtype, device=self.device)
        ka = ka/self.FRS_info["delta_k"][3:4]
        ka = einops.repeat(ka, 'n -> repeat n', repeat=FO.Z.shape[0])
        FO_k = FO.slice_all_dep(ka)

        self.render(x_des[0], FO_k)
    
    def render(self, plan, FO=None):
        """
        Render the plan

        Args:
            plan: (N_t, 2) torch array
            FO: (N_t, n_g, 2) zonotopes
        """

        # Plot t-p_t diagrams
        fig, axs = plt.subplots(self.n_state, 1)
        axs = axs.flatten()  # Flatten in case of more than one row
        
        for i, (k, v) in enumerate(self.state_dict.items()):
            ax = axs[i]
            ax.plot(self.t_des, plan[:, v], label=k)
            ax.legend()
            ax.grid(True)
            ax.set_title(k)

        if self.render_kwargs["render_offline"]:
            os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)
            save_path = os.path.join(self.render_kwargs["save_dir"], 
                                     f"time_vs_plan.png")
            plt.savefig(save_path, format='png')
            plt.close(fig)
        

        # trajectory plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(plan[:, 0], plan[:, 1])
        ax.grid(True)
        
        if FO is not None:
            # static zonotopes
            # This assumes that static_obs has the outer walls of the environment
            patches_data = torch.vstack([zono.polyhedron_patch() for zono in FO])
            vertices = torch.vstack([patches_data.reshape(-1, 2)])
            max_V = vertices.cpu().numpy().max(axis=0)
            min_V = vertices.cpu().numpy().min(axis=0)

            ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
            ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])

            patches = []
            for patch in patches_data:
                patches.append(ax.fill(patch[:, 0], patch[:, 1],
                                    edgecolor='blue', facecolor='blue', alpha=0.5, linewidth=0.1))
                
            fig.canvas.draw()

        if self.render_kwargs["render_offline"]:
            os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)
            save_path = os.path.join(self.render_kwargs["save_dir"], 
                                     f"traj.png")
            plt.savefig(save_path, format='png')
            plt.close(fig)

class Simple2DPlannerUnderAct(Simple2DPlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "k_lb" in kwargs.keys() and "k_ub" in kwargs.keys():
            print("Restricting the trajectory parameter")
            k_lb = kwargs["k_lb"]
            k_ub = kwargs["k_ub"]
            self.k_lb = k_lb
            self.k_ub = k_ub
