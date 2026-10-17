"""Experiment config: ``configs/maze2d_large.json`` (the paper's ``config_ws.json`` with paths removed)."""
import os

from safediffusion.utils.file_utils import load_config_from_json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "maze2d_large.json")

# guidance mode set for the planners that use JM2DPolicy (legacy: config_ws.json / config_ws_gibbs.json)
PLANNER_ALIGN_MODE = {"jm2d": "jm2d", "gibbs": "gibbs"}


def load_config(path=None, planner=None, num_workers=None):
    """Load the experiment config (robomimic ``Config``: algo.ddim/ddpm overrides + ``safety`` block).

    Args:
        path (str): config JSON (default ``configs/maze2d_large.json``)
        planner (str): if ``jm2d`` or ``gibbs``, sets ``safety.align.mode`` accordingly
        num_workers (int): process-pool size for JM2D/Gibbs candidate rollouts (default: config value; None means
            min(cpu_count, N_sample_action) as in the research code; results do not depend on it)
    """
    config = load_config_from_json(path or DEFAULT_CONFIG)
    with config.values_unlocked():
        if planner in PLANNER_ALIGN_MODE:
            config.safety.align.mode = PLANNER_ALIGN_MODE[planner]
        if num_workers is not None:
            config.safety.align.num_workers = int(num_workers)
    return config
