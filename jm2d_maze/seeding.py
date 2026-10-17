"""Seeding (the paper protocol) and the one determinism fix needed for reproducible PointMaze runs.

Paper protocol (``scripts/run.py``; legacy ``run_rail_v2.py`` + ``rollout_random_seed``)::

    set_random_seed(42)                                  # before building planner and environments
    for seed in env_seeds:                               # 0, 1, 2, ... in order
        set_random_seed(seed, deterministic_torch=False) # inside rollout_seeds
        env.reset(); rollout(...)

``seed_gym_spaces``: gym's ``MujocoEnv.__init__`` steps every new environment once with
``action_space.sample()`` from an unseeded space (OS entropy). The random step shifts
``SafeMazeEnv.robotqpos_to_worldpos`` (geom position minus qpos after the step) by O(1e-3) and therefore the goal
fed to the diffusion policy, so the research code gave different rollouts for identical seeds. Seeding unseeded
spaces with a constant removes this; it changes no algorithm (it fixes the one construction-time action).
``restore_gym_spaces()`` (or the function returned by ``seed_gym_spaces``) undoes the patch; ``seeded_gym_spaces()``
is a context manager.
"""
import contextlib

from safediffusion.utils.rand_utils import set_random_seed

GYM_SPACE_SEED = 0
_ORIGINAL_NP_RANDOM = None   # gym.spaces.Space.np_random before the first patch


def seed_gym_spaces(seed=GYM_SPACE_SEED):
    """Make unseeded ``gym.spaces.Space`` objects seed themselves with ``seed`` instead of OS entropy.

    The patch is process-wide and stays active (``build_planner`` applies it by default). Returns a function that
    undoes it, the same as ``restore_gym_spaces()``; ``seeded_gym_spaces`` is the context-manager form.
    Spaces that already drew their generator while patched keep it after the undo.
    """
    global _ORIGINAL_NP_RANDOM
    import gym

    if _ORIGINAL_NP_RANDOM is None:
        _ORIGINAL_NP_RANDOM = gym.spaces.Space.__dict__["np_random"]

    def _np_random(self):
        if self._np_random is None:
            self.seed(seed)
        return self._np_random

    gym.spaces.Space.np_random = property(_np_random)
    return restore_gym_spaces


def restore_gym_spaces():
    """Undo ``seed_gym_spaces``: unseeded spaces draw from OS entropy again (gym's behaviour). No-op if not patched."""
    global _ORIGINAL_NP_RANDOM
    if _ORIGINAL_NP_RANDOM is None:
        return
    import gym

    gym.spaces.Space.np_random = _ORIGINAL_NP_RANDOM
    _ORIGINAL_NP_RANDOM = None


@contextlib.contextmanager
def seeded_gym_spaces(seed=GYM_SPACE_SEED):
    """``with seeded_gym_spaces(): ...`` applies ``seed_gym_spaces`` inside the block and restores the previous state
    on exit (also if the patch was already active before the block, it stays active)."""
    was_patched = _ORIGINAL_NP_RANDOM is not None
    seed_gym_spaces(seed)
    try:
        yield
    finally:
        if not was_patched:
            restore_gym_spaces()


__all__ = ["set_random_seed", "seed_gym_spaces", "restore_gym_spaces", "seeded_gym_spaces", "GYM_SPACE_SEED"]
