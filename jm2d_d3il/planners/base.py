"""Planner = (diffusion sampling variant, projector, policy) built the way the original evaluation scripts built them.

The construction mirrors SafeRobomimic@origin/mpd-improved ``scripts/eval_paper.py`` (dpcc, safediffuser, mpd)
and SafeRobomimic@origin/um_dpcc ``scripts/eval_dev_Kfunc.py`` (jm2d). The variant string of the original code is
replaced by explicit settings (``configs/planners/<name>.yaml``).
"""
from diffuser.sampling import Policy, Projector


class Planner:
    name = None
    policy_class = Policy
    diffusion_class = None          # None: the pickled GaussianDiffusion; JM2D uses a subclass

    def __init__(self, batch_size, constraint_padding, dynamics_constraints, trajectory_selection='minimum_projection_cost',
                 diffusion_timestep_threshold=1.0, horizon=8, test_ret=0, device='cuda', legacy_variant=None,
                 planner=None, **kwargs):
        if kwargs:
            raise TypeError(f"{type(self).__name__}: unknown settings {sorted(kwargs)}")
        self.batch_size = batch_size
        self.constraint_padding = constraint_padding
        self.dynamics_constraints = dynamics_constraints
        self.trajectory_selection = trajectory_selection
        self.diffusion_timestep_threshold = diffusion_timestep_threshold
        self.horizon = horizon
        self.test_ret = test_ret
        self.device = device
        self.legacy_variant = legacy_variant
        self.policy = None

    @property
    def label(self):
        """Name of the result directory (planner + variant / u-steps)."""
        return self.name

    def settings(self):
        """JSON-serialisable settings recorded with every result."""
        return dict(planner=self.name, label=self.label, legacy_variant=self.legacy_variant, batch_size=self.batch_size,
                    horizon=self.horizon, constraint_padding=self.constraint_padding,
                    dynamics_constraints=self.dynamics_constraints, trajectory_selection=self.trajectory_selection,
                    diffusion_timestep_threshold=self.diffusion_timestep_threshold, test_ret=self.test_ret)

    # -- hooks for the subclasses ------------------------------------------------------------------------------- #
    def projector_kwargs(self):
        return dict(gradient=False, gradient_weights={}, projection_variant=self.name)

    def policy_kwargs(self):
        return {}

    def prepare_diffusion(self, diffusion):
        pass

    # -- build and plan ------------------------------------------------------------------------------------------ #
    def build(self, experiment, task):
        """Create the projector and the policy for a loaded checkpoint (``experiment``) and a task scenario."""
        diffusion = experiment.diffusion
        dataset = experiment.dataset
        if self.diffusion_class is not None:
            assert isinstance(diffusion, self.diffusion_class), \
                f"{self.name}: load the checkpoint with diffusion_class={self.diffusion_class.__name__}"
        self.prepare_diffusion(diffusion)
        trajectory_dim, action_dim, _ = task.index_maps(diffusion)
        constraints = task.planner_constraints(diffusion, self.constraint_padding, self.dynamics_constraints)
        projector = Projector(horizon=self.horizon,
                              transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=diffusion.goal_dim,
                              constraint_list=constraints, normalizer=dataset.normalizer,
                              variant='states_actions', dt=task.dt,
                              cost_dims=None, device=self.device, solver='scipy',
                              diffusion_timestep_threshold=self.diffusion_timestep_threshold,
                              **self.projector_kwargs())
        self.policy = self.policy_class(model=diffusion, normalizer=dataset.normalizer, preprocess_fns=[],
                                        test_ret=self.test_ret, projector=projector,
                                        trajectory_selection=self.trajectory_selection, **self.policy_kwargs())
        return self.policy

    def __call__(self, obs):
        """One planning step: returns (action, Trajectories(actions, observations), diffusion, diffusion_logs)."""
        return self.policy(conditions={0: obs},
                           batch_size=self.batch_size,
                           horizon=self.horizon,
                           disable_projection=False,
                           return_diffusion=True)
