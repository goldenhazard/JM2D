"""Repository paths and default locations (all overridable from the command line)."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
PLANNER_CONFIG_DIR = os.path.join(CONFIG_DIR, "planners")
THIRD_PARTY_DIR = os.path.join(REPO_ROOT, "third_party")          # holds the patched D3IL clone: third_party/d3il
DEFAULT_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "avoiding-d3il", "H8_K20_Dmodels.GaussianDiffusion")
PAPER_DIR = os.path.join(REPO_ROOT, "paper")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
