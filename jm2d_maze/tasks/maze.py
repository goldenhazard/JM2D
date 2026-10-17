"""PointMaze task: the D4RL ``maze2d-large`` environment with inflated walls and a zonotope safety wrapper.

``SafeMazeEnv`` was ``safediffusion_d4rl/environments/safe_maze_env.py`` (SafeRobomimic@robomimic 8b2371d2); unchanged.
Test-time padding w (Fig. 4 x-axis) is selected through the maze spec string of the D4RL fork
goldenhazard/D4RL@a0b30c4 (``LARGE_MAZE_PAD005`` ... ``LARGE_MAZE_PAD025`` append ``"PAD<w>"`` to the spec, and
``MazeEnv.__init__`` inflates every wall by w).
"""
import os
from copy import deepcopy

import numpy as np

from safediffusion.envs.env_zonotope import ZonotopeEnv
import torch
import matplotlib.pyplot as plt

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope

PADS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
# legacy maze names (folder names of the stored runs, ``--maze_name`` of the legacy scripts)
LEGACY_MAZE_NAMES = {0.0: "large", 0.05: "large_pad005", 0.10: "large_pad010", 0.15: "large_pad015",
                     0.20: "large_pad020", 0.25: "large_pad025"}


def pad_key(pad):
    """0.15 -> "pad015"; pads that are not a multiple of 0.01 keep all digits: 0.125 -> "pad0125" (task names)."""
    hundredths = round(float(pad) * 100, 6)
    if hundredths == int(hundredths):
        return f"pad{int(hundredths):03d}"
    return "pad" + f"{float(pad):.6f}".rstrip("0").replace(".", "")


def task_name(pad, backup):
    """Task identifier used in results: ``maze_pad015_hq``."""
    return f"maze_{pad_key(pad)}_{backup}"


def padded_maze_spec(layout, pad):
    """``layout`` with every wall inflated by ``pad``, built like the fork's ``LARGE_MAZE_PAD<w>`` specs."""
    return layout if pad == 0 else layout + "PAD" + str(pad)


def get_maze_spec(pad, layout=None):
    """Maze spec string of ``maze2d-large`` (or of ``layout``) with walls inflated by ``pad`` (needs the D4RL fork).

    Args:
        pad (float): wall padding w >= 0; the paper pads ``PADS`` resolve to the fork's ``LARGE_MAZE_PAD<w>`` specs
        layout (str): optional maze layout of the size of maze2d-large (9 rows x 12 tiles: ``#`` wall, ``O`` free,
            ``G`` goal; rows joined by a backslash, see ``read_maze_layout``); default: the maze2d-large layout
    """
    from d4rl.pointmaze import maze_model
    pad = float(pad)
    if not 0 <= pad < float("inf"):
        raise ValueError(f"padding must be a finite number >= 0, got {pad}")
    if layout is None:
        specs = {0.0: maze_model.LARGE_MAZE, 0.05: maze_model.LARGE_MAZE_PAD005, 0.10: maze_model.LARGE_MAZE_PAD010,
                 0.15: maze_model.LARGE_MAZE_PAD015, 0.20: maze_model.LARGE_MAZE_PAD020,
                 0.25: maze_model.LARGE_MAZE_PAD025}
        for w, spec in specs.items():
            if abs(pad - w) < 1e-9:
                return spec
        return padded_maze_spec(maze_model.LARGE_MAZE, pad)
    rows, ref = layout.split("\\"), maze_model.LARGE_MAZE.split("\\")
    if len(rows) != len(ref) or any(len(r) != len(ref[0]) for r in rows):
        raise ValueError(f"maze layout must have {len(ref)} rows of {len(ref[0])} tiles, got {[len(r) for r in rows]}")
    maze_model.parse_maze(layout)   # raises on unknown tiles
    return padded_maze_spec(layout, pad)


def read_maze_layout(path):
    """Layout file (one maze row per line, e.g. ``#OOOO#OOOOO#``) -> layout string for ``get_maze_spec``."""
    with open(path) as f:
        return "\\".join(line.strip() for line in f if line.strip())


def copy_obs_sharing_obstacles(obs_dict):
    """``deepcopy(obs_dict)`` except that the obstacle zonotope list is shared, not copied.

    Speed-up (release, exact): the planners deep-copy the observation for every DP candidate before shifting the
    robot zonotope; the obstacle zonotopes (references to the environment's geometry table) are only read, so
    copying them (about 40 tensors per copy) is not needed. Everything else is deep-copied as before.
    """
    zonotopes = obs_dict.get("zonotope") if isinstance(obs_dict, dict) else None
    if not isinstance(zonotopes, dict) or "obstacle" not in zonotopes:
        return deepcopy(obs_dict)
    obstacles = zonotopes["obstacle"]
    return deepcopy(obs_dict, {id(obstacles): obstacles})


def get_open_maze_spec(maze_size_str):
    """Wall-free maze used by the state predictor (legacy ``OPEN_MAZE_DICT``)."""
    from d4rl.pointmaze.maze_model import LARGE_MAZE_OPEN, MEDIUM_MAZE_OPEN
    return {"large": LARGE_MAZE_OPEN, "medium": MEDIUM_MAZE_OPEN}[maze_size_str]


class SafeMazeEnv(ZonotopeEnv):
    """
    Safety Zonotope Wrapper for Maze2D environments.

    The safety specifications are defined as not colliding with the walls.

    NOTE: get_observations() returns the robot qpos, while all the renderer functions are based on the world position.
    """
    def __init__(self, env, **kwargs):
        assert env.name.startswith('maze2d')

        super().__init__(env, **kwargs)
        
        # Set the third-party renderer
        self.janner_renderer = None

        # Transformation between the worlds
        self.d4rlmazepos_to_worldpos = np.array([1, 1])
        self.robotqpos_to_worldpos = self.get_robot_to_world_transformation()

        self.success_radius = 0.2

        self.success        = False
        self.done           = False

    # --------------------- Helper Function ------------------------------ #
    def set_state(self, pos, vel):
        """ 
        Set the state of the MuJoCo simulation using position and velocity in world frame
        
        Args:
            pos (np.ndarray): position of the agent in the world frame
            vel (np.ndarray): velocity of the agent in the world frame
        
        NOTE: this pos and vel is different from the qpos and qvel
        """
        t_dummy = np.array([0])
        qpos = pos - self.robotqpos_to_worldpos
        qvel = vel

        state_dict = {"states": np.concatenate([t_dummy, qpos, qvel])}

        return self.reset_to(state_dict)
    
    def get_robot_to_world_transformation(self):
        """
        Get the transformation matrix that transforms the robot qpos to the world position (x, y)

        In the Maze 2D environment, it is simple transformation
        """
        robot_zonotope = self.geom_table["zonotope"][self.geom_table["robot"]].values[0]
        world_robotpos = np.array(robot_zonotope.center[:2])
        body_robotpos  = self.unwrapped_env.sim.data.qpos

        offset = world_robotpos - body_robotpos

        return offset
    
    def is_success(self):
        """
        Maze 2D environment is successful when the agent reaches the target

        Pointmaze environment does not return success info, so overriden.
        """
        dict = {"task": self.success}

        return dict

    # ------------------- SafetyEnv related functions -------------------#
    def is_safe(self):
        """ For Maze 2D, the safety is defined as not colliding with the walls
        """
        return not self.collision()
    
    def is_observation_safe(self, obs):
        """
        obs: observation dictionary
        """
        PAD_MARGIN      = 0.05
        robot_center    = obs["flat"][:, :2] + self.robotqpos_to_worldpos # (B, 2)
        robot_size      = np.diag(list(self.geom_table["zonotope"][self.geom_table["robot"]])[0].generators)[:2]
        
        obs_zonos       = list(self.geom_table["zonotope"][self.geom_table["static_obs"]])
        obstacle_center = np.stack([z.center[:2] for z in obs_zonos])                      # (N_obs, 2)
        obstacle_size   = np.stack([np.diag(z.generators)[:2] for z in obs_zonos])         # (N_obs, 2)
        obstacle_size   = obstacle_size + robot_size
        obstacle_size   = obstacle_size + PAD_MARGIN # (N_obs, 2)
        
        robot_center = robot_center[:, np.newaxis, :] # (B, 1, 2)
        obstacle_center = obstacle_center[np.newaxis, :, :] # (1, N_obs, 2)
        obstacle_size  =  obstacle_size[np.newaxis, :, :] # (1, N_obs, 2)
        
        abs_diff  = np.abs(robot_center - obstacle_center) # (B, N_obs, 2)
        is_inside = np.all(abs_diff <= obstacle_size, axis=2)  # (B, N_obs)
        
        is_inside_any = np.any(is_inside, axis=1, keepdims=True)  # (B, 1)
        
        is_safe = ~is_inside_any
        
        return is_safe

    def register_robot_geoms(self):
        """ Register the robot geometry that has body name of particle
        """
        robot_body_names = ["particle"]
        
        return self.get_geom_that_body_name_is_in(robot_body_names)
    
    def register_static_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        static_obs_prefix = ["wall"]

        return self.get_geom_that_name_starts_with(static_obs_prefix)
    
    def register_dynamic_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        return []
    
    def get_observation(self, obs=None):
        obs = super().get_observation(obs)

        obs["zonotope"]["robot"][0].center[:2] = torch.tensor(self.d4rlmazepos_to_worldpos + obs["flat"][-1][:2])
        return obs
    
    def reset(self):
        """
        Reset the environment
        """
        obs = super().reset()
        self.success = False
        self.done = False

        return obs
    
    def reset_to(self, state_dict):
        """
        Reset the environment to the given state

        Args:
            state_dict (dict): dictionary containing the states to reset

        Returns:
            obs (np.ndarray): observation
        """
        obs = super().reset_to(state_dict)
        self.success = False
        self.done = False

        return obs

    def step(self, action):
        """
        Step the environment with the given action

        Args:
            action (np.ndarray): action to take

        Returns:
            obs (np.ndarray): observation
            reward (float): reward
            done (bool): done flag
            info (dict): info dictionary
        """
        obs, reward, done, info = super().step(action)
        
        # Update the success flag
        goal_qpos = self.get_goal()["flat"][:2]

        # Environment is frame stacking wrapper. Retrieve the most recent one.
        if obs["flat"][:2].ndim == 2:
            qpos = obs["flat"][-1, :2]
        else:
            qpos = obs["flat"][:2]

        if np.linalg.norm(goal_qpos - qpos) <= self.success_radius:
            self.success = True
            done = True
    
        return obs, reward, done, info
    
    def get_goal(self):
        """
        Get the goal qpos, qvel configuration for the robot
        """
        # Get goal qpos
        goal_pos     = self.unwrapped_env.get_target()          # goal d4rl pos
        goal_pos     = goal_pos + self.d4rlmazepos_to_worldpos  # goal safemazeenv pos
        goal_qpos    = goal_pos - self.robotqpos_to_worldpos    # goal robot qpos
        
        # Get goal qvel
        goal_qvel     = np.zeros(2,)
        goal_state   = np.concatenate((goal_qpos, goal_qvel))

        goal_dict = {}
        goal_dict["flat"] = goal_state
        
        return goal_dict
    
    def set_goal(self, pos):
        """ 
        Set the goal position of the agent

        Args:
            pos (np.ndarray): goal position in the world frame
        """
        
        pos = pos - self.d4rlmazepos_to_worldpos # goal d4rl pos
        self.unwrapped_env.set_target(target_location=pos)

        pos_world = pos + self.d4rlmazepos_to_worldpos
        c = torch.hstack([torch.tensor(pos_world), torch.tensor(0)])
        G = torch.eye(3) * 0.1
        Z = torch.vstack([c, G])
        goal_zonotope = zonotope(Z)

        self.goal_zonotope = goal_zonotope

        return self.get_goal()


    # ------------------- Render Helper ------------------------------ #        
    def draw_zonotopes(self, zonotopes, color, alpha, linewidth):
        """
        Draw the 2D zonotope on the plot

        Args:
            zonotopes (list): list of zonotopes
            color (str): color of the zonotope
            alpha (float): transparency of the zonotope
            linewidth (float): width of the line

        Returns:
            zonotope_patches (list): list of patches that are drawn
            zonotope_patch_data (list): list of vertices of the zonotopes
        """        
        zonotope_patch_data = torch.vstack([zonotope.polyhedron_patch() for zonotope in zonotopes])

        zonotope_patches = []
        for zonotope_patch in zonotope_patch_data:
            zonotope_patches.append(self.ax.fill(zonotope_patch[:, 0], zonotope_patch[:, 1],
                                             edgecolor=color, facecolor=color, alpha=alpha, linewidth=linewidth))
        
        return (zonotope_patches, zonotope_patch_data)

    def initialize_renderer(self):
        """
        Initialize the renderer for the zonotope visualization

        This function makes the plot to be 2D
        """
        self.fig = plt.figure()
        self.fig.set_size_inches(self.render_setting["width"], self.render_setting["height"])
        self.ax = self.fig.add_subplot(111)
        
        # static zonotopes
        # This assumes that static_obs has the outer walls of the environment
        patches_data = self.draw_zonotopes_in_geom_table("static_obs")
        vertices = torch.vstack([patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        
        # TODO
        # self.zoom(zoom_factor = self.render_setting["zoom_factor"])

    def custom_render(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        """Renders the environment.

        Args:
            mode (str): the mode to render with
        """
        if mode == "zonotope":
            img = super().custom_render(mode, height, width, camera_name, **kwargs)
            
        elif mode == "janner":
            if self.janner_renderer is None:
                from safediffusion.utils.render_utils import Maze2dRenderer
                self.janner_renderer = Maze2dRenderer(self.env.name)
            
            img = self.janner_renderer.render(self.get_observation()["flat"], **kwargs)
        
        return img
    
    def adjust_camera(self, camera_name):
        """
        For Maze2D env, ignore this
        """
        pass
