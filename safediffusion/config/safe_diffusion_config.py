from robomimic.config.base_config import BaseConfig
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig

class SafeDiffusionConfig(DiffusionPolicyConfig):
    ALGO_NAME = "safediffusion"

    def __init__(self, dict_to_load=None):
        super().__init__(dict_to_load=dict_to_load)
        self.unlock_keys()
        self.safety_config()
        self.lock_keys()

    def safety_config(self):
        self.safety.exp_name    = "test"
        self.safety.output_dir  = "../{}_rollout_result".format(self.algo_name)
        self.safety.policy_path = None
        
        # zonotope configuration
        self.safety.zonotope.order    = 40
        self.safety.zonotope.max_comb = 200

        # safety-filter configuration
        self.safety.filter.n_head = 1
        self.safety.filter.verbose = True
        self.safety.filter.max_init_attempts = 10

        # safe diffusion policy configuration
        self.safety.align.mode                  = "jm2d"               # guidance mode: "jm2d" or "gibbs"
        self.safety.align.N_steps_mppi          = 3                    # mppi step for sampling-based guidance
        self.safety.align.temp_mppi             = 0.2                  # mppi temperature
        self.safety.align.opt_K_again           = True                 # whether we do K optimization at the last step (running mppi sampler)
        self.safety.align.timestep_for_guidance = 4                    # start applying guidance at this time
        self.safety.align.N_sample_action       = 8                    # number of actions to sample
        self.safety.align.N_sample_backup       = 128                  # number of backup actions to sample for alignment
        self.safety.align.N_sample_trajopt      = 512                  # number of backup actions to sample for trajectory optimization
        self.safety.align.score_comp_style      = "fast"               # score evaluation mode
        self.safety.align.ddim_eta              = 1.0                  # defaults to ddpm
        self.safety.align.num_workers           = None                 # process-pool size for the candidate rollouts (None: min(cpu_count, N_sample_action))

        self.safety.test_time_perturb.delay_observation.enabled = False
        self.safety.policy.ddim_eta = 1.0

        # trajectory optimization
        self.safety.trajopt.verbose = False
        self.safety.trajopt.nlp_time_limit = 3.0
        self.safety.trajopt.use_reachable_scene_filter = False  # only build the constraint for the obstacle that is reachable (rough-check)
        self.safety.trajopt.reachable_scene_radius     = 0.35   # the radius of the reachable scene, accounted from the robot end-effector
        
        self.safety.trajopt.use_last_few_arm_links     = False  # use the last few arm links for the constraint qualification
        self.safety.trajopt.num_last_few_arm_links     = 3      # use the last few arm links for the constraint qualification
        
        self.safety.trajopt.use_gripper_approximation  = False  # use the gripper approximation for the constraint qualification
        self.safety.trajopt.pad_radius                 = 0.02   # pad the radius of the gripper for the constraint qualification
        self.safety.trajopt.ignore_arm_active_object_collision = False # ignore the arm-active object collision for the constraint qualification
        self.safety.trajopt.ignore_gripper_desk_collision      = False # ignore the gripper-desk collision for the constraint qualification
        
        # render configuration
        self.safety.render.save_dir = None

        # Diffusion rendering
        self.safety.render.diffusion.enabled = False
        
        self.safety.render.zonotope.width = 20
        self.safety.render.zonotope.height = 20
        self.safety.render.zonotope.ticks  = True
        self.safety.render.zonotope.zoom_factor = 1.5
        self.safety.render.zonotope.grid = True
        self.safety.render.zonotope.axis = True
        
        self.safety.render.zonotope.robot.color = "black"
        self.safety.render.zonotope.robot.alpha = 0.1
        self.safety.render.zonotope.robot.linewidth = 0.5

        self.safety.render.zonotope.active_object.color = "yellow"
        self.safety.render.zonotope.active_object.alpha = 1
        self.safety.render.zonotope.active_object.linewidth = 0.1

        self.safety.render.zonotope.goal.color = "purple"
        self.safety.render.zonotope.goal.alpha = 0.3
        self.safety.render.zonotope.goal.linewidth = 0.5
        
        self.safety.render.zonotope.static_obs.color = "red"
        self.safety.render.zonotope.static_obs.alpha = 0.1
        self.safety.render.zonotope.static_obs.linewidth = 0.5
        
        self.safety.render.zonotope.dynamic_obs.color = "blue"
        self.safety.render.zonotope.dynamic_obs.alpha = 0.1
        self.safety.render.zonotope.dynamic_obs.linewidth = 0.5
        
        # FRS (optional)
        self.safety.render.zonotope.FRS.color = "green"
        self.safety.render.zonotope.FRS.alpha = 0.1
        self.safety.render.zonotope.FRS.linewidth = 0.5
        
        # FRS (optional)
        self.safety.render.zonotope.FRS2.color = "purple"
        self.safety.render.zonotope.FRS2.alpha = 0.1
        self.safety.render.zonotope.FRS2.linewidth = 0.5

        # Plan (optional)
        self.safety.render.zonotope.plan.color = "cyan"
        self.safety.render.zonotope.plan.alpha = 0.1
        self.safety.render.zonotope.plan.linewidth = 8.0
        self.safety.render.zonotope.plan.linestyle = "-"

        self.safety.render.zonotope.backup_plan.color = "green"
        self.safety.render.zonotope.backup_plan.linewidth = 1.0
        self.safety.render.zonotope.backup_plan.alpha = 0.1
        self.safety.render.zonotope.backup_plan.linestyle = "--"

        self.safety.render.zonotope.plans.linewidth = 2.0
        self.safety.render.zonotope.plans.linestyle = "-"
        self.safety.render.zonotope.plans.alpha = 0.4

        
