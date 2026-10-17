import abc
import os
from enum import Enum

import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope, batchZonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope
import safediffusion.utils.reachability_utils as ReachUtils
import safediffusion.utils.list_utils as ListUtils
from safediffusion.envs.env_safety import SafetyEnv

class GeomType(Enum):
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7
    SDF = 8

class ZonotopeEnv(SafetyEnv):
    """
    Zonotope Environment Class
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

        # visualization setting
        self.render_setting = kwargs["render"]["zonotope"]
        self.fig     = None
        self.ax      = None

        # zonotope objects to draw
        self.patches = {
            "robot"       : None,
            "dynamic_obs" : None,
            "static_obs"  : None,
            "FRS"         : None,
            "FRS2"        : None,
            "goal"        : None,
        }

        self.plots = {
            "plan": None,
            "backup_plan": None,
            "plans": None
        }

        self.light_source = None

    def get_observation(self, obs=None):
        """
        Add zonotope information to the observation dictionary

        Args:
            obs (dict): observation dictionary
        
        Returns:
            obs_dict (dict): updated observation dictionary with zonotope information
        
        Example:
            obs_dict["zonotope"]["obstacle"] = list of zonotopes of the obstacles
            obs_dict["zonotope"]["robot"] = list of zonotopes of the robot
        """
        obs_dict = super().get_observation(obs)

        # make sure to synchronize before getting observation
        self._sync()

        zonotope_dict = dict()
        zonotope_dict["obstacle"] = []
        zonotope_dict["obstacle"].extend(list(self.geom_table["zonotope"][self.geom_table["static_obs"]]))
        zonotope_dict["obstacle"].extend(list(self.geom_table["zonotope"][self.geom_table["dynamic_obs"]]))
        zonotope_dict["robot"] = []
        zonotope_dict["robot"].extend(list(self.geom_table["zonotope"][self.geom_table["robot"]]))

        obs_dict["zonotope"] = zonotope_dict

        return obs_dict

    def set_goal(self, **kwargs):
        obs = super().set_goal(**kwargs)
        self._sync()

        return obs
    
    @property
    def name(self):
        return "Zono" + super().name
    
    def create_geom_table(self):
        """
        Add the zonotope column that stores the zonotopic representation of the geometry
        """
        geom_table                       = super().create_geom_table()
        geom_table["zonotope_primitive"] = [self.get_zonotope_primitive_from_geom_id(geom_id) 
                                                for geom_id in geom_table.index]
        geom_table["zonotope"]           = [self.get_zonotope_from_geom_id(geom_id) 
                                                for geom_id in geom_table.index]
        
        return geom_table
    
    # ------------------------------------------------------------ #
    # ----------------- Zonotope-related functions --------------- #
    # ------------------------------------------------------------ #
    def _sync(self):
        """
        Update the zonotope of the robot and dynamic obstacles
        """
        mjdata = self.unwrapped_env.sim.data
        robot_mask = self.geom_table["robot"]
        dynamic_obs_mask = self.geom_table["dynamic_obs"]
        update_mask = robot_mask | dynamic_obs_mask
        self.geom_table.loc[update_mask, "zonotope"] = [ReachUtils.transform_zonotope(
                                                            zono = self.geom_table["zonotope_primitive"][geom_id],
                                                            pos  = mjdata.geom_xpos[geom_id],
                                                            rot  = mjdata.geom_xmat[geom_id].reshape(3, 3)
                                                        ) 
                                                        for geom_id in self.geom_table.loc[update_mask].index]

    def get_zonotope_from_geom_name(self, geom_name):
        """
        Get the zonotope from the geom_name

        Args:
            geom_name (str): name of the geom

        Returns:
            ZP (zonotope): zonotope of the geom
        """
        geom_id = self.unwrapped_env.sim.model.geom_name2id(geom_name)
        return self.get_zonotope_from_geom_id(geom_id)

    def get_zonotope_primitive_from_geom_id(self, geom_id):
        """
        Get the zonotope primitive of the geometry given `geom_id`

        This zonotope does not depend on the current pos/rot of the geometry

        Args
            geom_id (int): id of the geometry
        
        Returns
            zonotope_primitive (zonotope)
        """

        mjmodel = self.unwrapped_env.sim.model

        geom_type = mjmodel.geom_type[geom_id]
        geom_size = mjmodel.geom_size[geom_id]

        zero = torch.zeros(3,)
        eye  = torch.eye(3)

        # Get Zonotope Primitive (ZP)
        if geom_type == GeomType.PLANE.value:
            ZP = ReachUtils.get_zonotope_from_plane_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.SPHERE.value:
            ZP = ReachUtils.get_zonotope_from_sphere_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.CYLINDER.value:
            ZP = ReachUtils.get_zonotope_from_cylinder_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.BOX.value:
            ZP = ReachUtils.get_zonotope_from_box_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.MESH.value:
            mesh_id = mjmodel.geom_dataid[geom_id]
            vert_start = mjmodel.mesh_vertadr[mesh_id]
            vert_count = mjmodel.mesh_vertnum[mesh_id]
            vertices = mjmodel.mesh_vert[vert_start:vert_start + vert_count].reshape(-1, 3)
            ZP = ReachUtils.get_zonotope_from_mesh_vertices(vertices=vertices)
        else:
            print(f"Geom type {geom_type} is not supported yet.")
            raise NotImplementedError("Not Implemented")

        if ZP is not None:
            ZP = ZP.to(device=self.device, dtype=self.dtype)
            mjmodel = self.unwrapped_env.sim.model
            ZP.name = mjmodel.geom_id2name(geom_id)
        
        return ZP
    
    def get_zonotope_from_geom_id(self, geom_id):
        """
        Get the zonotope given the simulation data
        """

        mjdata = self.unwrapped_env.sim.data
        
        ZP = self.get_zonotope_primitive_from_geom_id(geom_id)

        geom_pos = mjdata.geom_xpos[geom_id]
        geom_rot = mjdata.geom_xmat[geom_id].reshape(3, 3)

        Z = ReachUtils.transform_zonotope(ZP, pos = geom_pos, rot = geom_rot)

        return Z
    
    # ------------------------------------------------------------ #
    # ----------------- Renderer-related functions --------------- #
    # ------------------------------------------------------------ #
    def clear_patches(self, patches):
        """
        Clear the patches from the figure.

        This uses recursive removing of the patches in order to support both 3d and 2d plot.

        Args:
            patches: list of patches or a single patch to remove
        """
        if type(patches) == list:
            for patch in patches:
                self.clear_patches(patch)
        else:
            patches.remove()
    
    def draw_zonotopes_in_geom_table(self, key, remove_if_exists=True):
        """
        Draw the zonotope group in the geom_table

        Args:
            key (str): key of the zonotope group
            remove_if_exists (bool): remove the existing patches if True
        
        Returns:
            patch_data (list): list of vertices of the zonotopes
        
        Example:
            draw_zonotopes_in_geom_table("static_obs")
        """
        # Input validation
        assert key in self.patches and key in self.geom_table.columns, f"Key {key} is not in the patches dictionary"
        
        # Clear patches if exists
        if self.patches[key] is not None and remove_if_exists:
            self.clear_patches(self.patches[key])
        # Get the zonotopes series data
        zonotopes = self.geom_table["zonotope"][self.geom_table[key]]

        if zonotopes.empty:
            return None
        
        self.patches[key], patch_data = self.draw_zonotopes(zonotopes, 
                                                self.render_setting[key]["color"],
                                                self.render_setting[key]["alpha"],
                                                self.render_setting[key]["linewidth"])
        return patch_data
    
    def add_circle(self, img, color):
        """
        Add a circle to the image

        Args:
            img (np.array): image to add the circle
            center (tuple): center of the circle
            radius (int): radius of the circle
        
        Returns:
            img (np.array): updated image with the circle
        """
        radius = 16
        center = [64, 64]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=color)
        return np.array(img)
    
    def draw_zonotopes(self, zonotopes, color, alpha, linewidth):
        """
        Draw the 3D zonotope on the plot

        Args:
            zonotopes (list): list of zonotopes / batchzonotopes
            color (str): color of the zonotope
            alpha (float): transparency of the zonotope
            linewidth (float): width of the line

        Returns:
            patches (list): list of patches that are drawn
            patch_data (list): list of vertices of the zonotopes
        """
        zonotopes = ListUtils.maybe_flatten_the_list(zonotopes)
        zonotopes = ReachUtils.maybe_flatten_the_zonotope(zonotopes)

        # (324, 3, 3)
        patch_data = torch.vstack([zonotope.polyhedron_patch() for zonotope in zonotopes])

        # Convert torch tensor to numpy array (if needed)
        # patch_data = patch_data.numpy() if isinstance(patch_data, torch.Tensor) else patch_data
        
        patches = self.ax.add_collection3d(Poly3DCollection(patch_data,
                                                             edgecolor=color,
                                                             facecolor=color,
                                                             alpha=alpha,
                                                             linewidth=linewidth))
        # # Create a light source object
        # ls = LightSource(azdeg=45, altdeg=45)
        # vertices = patch_data.reshape(-1, 3)
        # rgb_to_shade = np.stack([vertices[:, 0], vertices[:, 1], vertices[:, 2]], axis = 1)
        # rgb = ls.shade(rgb_to_shade, plt.cm.viridis)

        # patches.set_face_color(rgb)
        
        # for idx, patch in enumerate(patches):
        #     patch[idx].set_facecolor(rgb[idx, :, :])
        
        # # Define a basic light direction (simulates "camlight" effect)
        # light_direction = np.array([0, 1, 0])  # Light coming from above (z-axis)

        # # Simulate shading based on face normals
        # def calculate_face_normal(polygon):
        #     v0, v1, v2 = polygon[:3]
        #     normal = np.cross(v1 - v0, v2 - v0)
        #     normal /= np.linalg.norm(normal)  # Normalize the normal vector
        #     return normal

        # for patch, polygon in zip(patches.get_paths(), patch_data):
        #     # Calculate the normal vector for the face
        #     normal = calculate_face_normal(polygon)

        #     # Calculate the dot product between the light direction and the normal
        #     lighting_intensity = np.dot(normal, light_direction)

        #     # Clamp the lighting intensity between 0 and 1
        #     lighting_intensity = max(0, min(1, lighting_intensity))

        #     # Adjust the face color based on the lighting intensity
        #     shaded_color = np.array(plt.colors.to_rgb(color)) * lighting_intensity
        #     patch.set_facecolor(np.append(shaded_color, alpha))  # Preserve transparency

        return patches, patch_data

    def initialize_renderer(self):
        """
        Initialize the renderer for the zonotope visualization
        """
        self.fig = plt.figure()
        self.fig.set_size_inches(self.render_setting["width"], self.render_setting["height"])
        self.ax = plt.axes(projection='3d')
        
        # Enable shading for better 3D effect
        self.ax.set_box_aspect((1, 1, 0.8))  # Adjust the aspect ratio as needed

        # Adjust the subplot parameters to reduce padding
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        # Set the background color of the panes to grey
        grey_color = (0.5, 0.5, 0.5, 1.0)  # RGB values for grey with full opacity

        self.ax.xaxis.set_pane_color(grey_color)
        self.ax.yaxis.set_pane_color(grey_color)
        self.ax.zaxis.set_pane_color(grey_color)

        # static zonotopes
        static_obs_zono_patches_data = self.draw_zonotopes_in_geom_table("static_obs")

        vertices = torch.vstack([static_obs_zono_patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        # initialize the domain
        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        self.ax.set_zlim([min_V[2] - 0.1, max_V[2] + 0.5]) # robot height

        # zoom setting
        self.zoom(zoom_factor = self.render_setting["zoom_factor"])

        # ticks setting
        if not self.render_setting["ticks"]:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if hasattr(self.ax, 'set_zticks'):
                self.ax.set_zticks([])
        
        # grid setting
        self.ax.grid(self.render_setting["grid"])

        # Set axis visibility based on self.show_axis
        self.set_axis_visibility(self.render_setting["axis"])

        # Initialize the light source
        self.light_source = LightSource(azdeg=315, altdeg=45)

    def set_axis_visibility(self, visible):
        """
        Set the visibility of the axis.
        
        Args:
            visible (bool): If True, show the axis. If False, hide the axis.
        """
        self.show_axis = visible
        if self.ax is not None:
            if visible:
                self.ax.set_axis_on()
            else:
                self.ax.set_axis_off()

    def adjust_camera_center(self, center):
        """
        Adjusting camera center is equivalent to zoom 1x and move zoom center
        """
        self.zoom(zoom_factor = 1, zoom_center = center)
    
    def zoom(self, zoom_factor, zoom_center = None):
        """
        Zoom the figure

        Args
            zoom_factor (float)
            zoom_center: (np.array) float
        """
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim() if hasattr(self.ax, 'get_zlim') else None

        if zoom_center is not None:
            cx = zoom_center[0]
            cy = zoom_center[1]
        else:
            cx = (xlim[1] + xlim[0])/2
            cy = (ylim[1] + ylim[0])/2
        
        dx = (xlim[1] - xlim[0])/zoom_factor
        dy = (ylim[1] - ylim[0])/zoom_factor

        new_xlim = [cx - dx/2, cx + dx/2]
        new_ylim = [cy - dy/2, cy + dy/2]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        if zlim:
            if zoom_center:
                zoom_center[2]
            else:
                cz = (zlim[1] + zlim[0])/2
            dz = (zlim[1] - zlim[0])/zoom_factor
            new_zlim = [cz - dz/2, cz + dz/2]
            self.ax.set_zlim(new_zlim)
    
    def draw_zonotopes_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw the zonotope category of name `name` at the figure canvas

        Args:
            kwargs (dict), the dictionary that has a zonotope kwargs[name]
            name (str), the name of the zonotope category
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            if self.patches[name] is not None:
                self.clear_patches(self.patches[name])
            
            self.patches[name], _ = self.draw_zonotopes(kwargs[name],
                                                        self.render_setting[name]["color"],
                                                        self.render_setting[name]["alpha"],
                                                        self.render_setting[name]["linewidth"])

    def draw_multitraj3D_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw the multiple trajectory of name `name` at the figure canvas
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            # Remove existing plot if any
            if self.plots[name] is not None:
                for line in self.plots[name]:
                    line.remove()
            
            # Generate a colormap
            colormap = plt.cm.get_cmap('tab10', len(kwargs[name]))
            
            self.plots[name] = []
            # Plot each trajectory with a different color
            for i in range(len(kwargs[name])):
                color = colormap(i / len(kwargs[name]))
                line, = self.ax.plot(kwargs[name][i][:, 0], 
                                     kwargs[name][i][:, 1], 
                                     kwargs[name][i][:, 2], 
                                     color     = color, 
                                     linewidth = self.render_setting[name]["linewidth"],
                                     linestyle = self.render_setting[name]["linestyle"],
                                     alpha     = self.render_setting[name]["alpha"])
                self.plots[name].append(line)

    def draw_multitraj2D_in_kwargs_if_exists(self, kwargs, name, marker = None, weights = None):
        """
        Draw the multiple trajectory of name `name` at the figure canvas
        
        Args
            kwargs[name] (np.array): trajectory (B, N, 2)
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            # Remove existing plot if any
            if name in self.plots.keys() and self.plots[name] is not None:
                for line in self.plots[name]:
                    line.remove()
            
            # Generate a colormap
            colormap = plt.cm.get_cmap('tab10', len(kwargs[name]))
            
            # Default size if no weights provided
            if weights is None:
                weights = np.ones((len(kwargs[name])))
            else:
                weights = np.nan_to_num(weights)
                weights = weights.sum(axis=-1)
                if weights.max() > 0:
                    weights /= weights.max()
            
            zero_alpha = 0.2
            nonzero_alpha_min = 0.6
    
            # Plot each set of points with different colors
            
            self.plots[name] = []
            # Plot each trajectory with a different color
            for i in range(len(kwargs[name])):
                color = colormap(i / len(kwargs[name]))
                alpha = zero_alpha + (1-zero_alpha) * ((weights[i] > 0) * (nonzero_alpha_min + weights[i] * (1-nonzero_alpha_min)))
                line, = self.ax.plot(kwargs[name][i][:, 0], 
                                     kwargs[name][i][:, 1], 
                                     color     = color, 
                                     linewidth = self.render_setting[name]["linewidth"],
                                     linestyle = self.render_setting[name]["linestyle"],
                                     alpha     = alpha,
                                     marker    = marker)
                self.plots[name].append(line)

    def draw_traj2D_in_kwargs_if_exists(self, kwargs, name, marker = None, markersize = None):
        """
        Draw the 2D trajectory of name `name` at the figure canvas
        
        Args
            kwargs[name] (np.array): trajectory (N, 2)
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            if name in self.plots.keys() and self.plots[name] is not None:
                self.plots[name][0].remove()

            color = self.render_setting[name]["color"]

            if color == "gradient":
                # Extract the trajectory points
                points = kwargs[name]
                
                # Create segments from the points for the LineCollection
                segments = np.array([points[i:i+2] for i in range(len(points)-1)])
                
                # Define a color map (e.g., viridis)
                cmap = cm.get_cmap('viridis')
                
                # Normalize the segments for color mapping
                norm = plt.Normalize(0, len(segments))
                colors = cmap(norm(np.arange(len(segments))))
                
                # Create a LineCollection with the segments and corresponding colors
                lc = LineCollection(segments, 
                                    colors=colors, 
                                    linewidth=self.render_setting[name]["linewidth"], 
                                    linestyle=self.render_setting[name]["linestyle"],
                                    alpha = self.render_setting[name]["alpha"])
                
                # Add the LineCollection to the axis
                self.ax.add_collection(lc)
                self.plots[name] = [lc]
            
            else:
                self.plots[name] = self.ax.plot(
                    kwargs[name][:, 0], 
                    kwargs[name][:, 1], 
                    color=self.render_setting[name]["color"], 
                    linewidth=self.render_setting[name]["linewidth"],
                    linestyle=self.render_setting[name]["linestyle"],
                    alpha = self.render_setting[name]["alpha"],
                    marker = marker,
                    markersize = markersize
                )

    def draw_multivel2D_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw multiple velocity vectors at points specified by plan_next_samples
        using velocities from vel_next_samples
        
        Args
            kwargs[name] (np.array): trajectory (N, 4)
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            # Remove existing plot if any
            if name in self.plots.keys() and self.plots[name] is not None:
                for quiver in self.plots[name]:
                    quiver.remove()
            
            # Get positions where to draw vectors
            positions = kwargs[name][:, :2]
            velocities = kwargs[name][:, 2:]
            
            # Generate a colormap
            colormap = plt.cm.get_cmap('tab10', len(velocities))
            
            self.plots[name] = []
            # Plot each velocity field with a different color
            for i in range(len(velocities)):
                color = colormap(i / len(velocities))
                # Create quiver plot for velocity vectors
                quiver = self.ax.quiver(positions[i, 0],  # x positions
                                    positions[i, 1],  # y positions
                                    velocities[i, 0],  # x velocities
                                    velocities[i, 1],  # y velocities
                                    color=color,
                                    alpha=self.render_setting[name]["alpha"],
                                    scale=100,  # adjust this to scale vector lengths
                                    width=0.005)  # adjust this to change arrow width
                self.plots[name].append(quiver)    
    
    def draw_multiscatter2D_in_kwargs_if_exists(self, kwargs, name, weights=None):
        if name in kwargs.keys() and kwargs[name] is not None:
            # Remove existing plots
            if name in self.plots.keys() and self.plots[name] is not None:
                for scatter in self.plots[name]:
                    scatter.remove()
            
            # Initialize plots list
            self.plots[name] = []
            
            # Generate colormap
            colormap = plt.cm.get_cmap('tab10', len(kwargs[name]))
            
            # Default size if no weights provided
            if weights is None:
                weights = np.ones((len(kwargs[name]), kwargs[name][0].shape[0]))
            else:
                weights = np.nan_to_num(weights, nan=0.0)
                if weights.max() > 0:
                    weights /= weights.max()
    
            # Plot each set of points with different colors
            min_alpha = self.render_setting[name]["min_alpha"]
            for i in range(len(kwargs[name])):
                color = colormap(i / len(kwargs[name]))
                
                scatter = self.ax.scatter(kwargs[name][i][:, 0], 
                                        kwargs[name][i][:, 1],
                                        s=self.render_setting[name]["size"],
                                        color=color,
                                        alpha=0.02 + (1-0.02) * ((weights[i] > 0) * (min_alpha + weights[i] * (1-min_alpha))))
                self.plots[name].append(scatter)
        
    def draw_traj3D_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw the trajectory of name `name` at the figure canvas
        
        Args
            kwargs[name] (np.array): trajectory (N, 3)
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            if self.plots[name] is not None:
                self.plots[name][0].remove()

            color = self.render_setting[name]["color"]

            if color == "gradient":
                # Extract the trajectory points
                points = kwargs[name]
                
                # Create segments from the points for the Line3DCollection
                segments = np.array([points[i:i+2] for i in range(len(points)-1)])
                
                # Define a color map (e.g., viridis)
                cmap = cm.get_cmap('viridis')
                
                # Normalize the segments for color mapping
                norm = plt.Normalize(0, len(segments))
                colors = cmap(norm(np.arange(len(segments))))
                
                # Create a Line3DCollection with the segments and corresponding colors
                lc = Line3DCollection(segments, 
                                      colors    = colors, 
                                      linewidth = self.render_setting[name]["linewidth"], 
                                      linestyle = self.render_setting[name]["linestyle"])
                
                # Add the Line3DCollection to the axis
                self.ax.add_collection(lc)
                self.plots[name] = [lc]
            
            else:
                self.plots[name] = self.ax.plot(
                    kwargs[name][:, 0], 
                    kwargs[name][:, 1], 
                    kwargs[name][:, 2], 
                    color=self.render_setting[name]["color"], 
                    linewidth=self.render_setting[name]["linewidth"],
                    linestyle=self.render_setting[name]["linestyle"]
                )
    
    def draw_env(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        if self.fig is None:
            self.initialize_renderer()

        # dynamic zonotopes
        self.draw_zonotopes_in_geom_table("dynamic_obs", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("robot", remove_if_exists=True)
        self.draw_zonotopes_in_kwargs_if_exists(kwargs, "FRS")
        self.draw_zonotopes_in_kwargs_if_exists(kwargs, "goal")
        
        self.draw_traj2D_in_kwargs_if_exists(kwargs, "plan")
        self.draw_traj2D_in_kwargs_if_exists(kwargs, "backup_plan")
        
        if "plans" in kwargs.keys():
            # Remove existing plot if any
            if self.plots["plans"] is not None:
                for line in self.plots["plans"]:
                    line.remove()
            
            # Initialize list to hold plot objects
            self.plots["plans"] = []
            
            # Generate a colormap
            colormap = plt.cm.get_cmap('tab10', len(kwargs["plans"]))
            
            # Plot each trajectory with a different color
            for i in range(len(kwargs["plans"])):
                plan = kwargs["plans"][i]
                color = colormap(i / len(kwargs["plans"]))
                line, = self.ax.plot(plan[:, 0], plan[:, 1], 
                                    color     = color, 
                                    linewidth = self.render_setting["plans"]["linewidth"],
                                    linestyle = self.render_setting["plans"]["linestyle"],
                                    alpha     = self.render_setting["plans"]["alpha"])
                self.plots["plans"].append(line)


    
    def custom_render(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        """Renders the environment as 3D zonotope world.

        Args:
            mode (str): the mode to render with
        """
        # draw
        self.draw_env(mode, height, width, camera_name, **kwargs)
        
        # view setting
        self.adjust_camera(camera_name)
        if "follow_robot" in kwargs.keys() and kwargs["follow_robot"]:
            robot_center = np.array(list(self.geom_table["zonotope"][self.geom_table["robot"]])[-1].center)
            self.adjust_camera_center(center = robot_center)

        # Update the light source based on the camera angle
        # elev, azim = self.ax.elev, self.ax.azim
        # self.light_source.azdeg = azim
        # self.light_source.altdeg = elev

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if "intervened" in kwargs.keys() and kwargs["intervened"]:
            img = self.add_circle(img, color = (0, 255, 0)) # if intervened, add green circle
        else:
            img = self.add_circle(img, color = (0, 0, 255)) # if not intervened, add blue circle
        
        return img
    
    def initialize_renderer_diffusion(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        self.fig = plt.figure(figsize=(24, 12))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[12, 12], height_ratios=[4, 4, 4])
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(gs[:, 0])  # Left plot, full height
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Right top
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # Right middle
        self.ax4 = self.fig.add_subplot(gs[2, 1])  # Right bottom
        
        ########################################
        # Axis 1 Setting
        ########################################
        self.ax = self.ax1  # Keep reference to main ax for compatibility
        self.fig.set_size_inches(self.render_setting["width"], self.render_setting["height"])
        
        # static zonotopes
        # This assumes that static_obs has the outer walls of the environment
        patches_data = self.draw_zonotopes_in_geom_table("static_obs")
        vertices = torch.vstack([patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        self.zoom(zoom_factor = self.render_setting["zoom_factor"])

        ########################################
        # Axis 2 Setting
        ########################################
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')
        self.ax2.legend()
        self.ax2.set_xlim([-1.1, 1.1])
        self.ax2.set_ylim([-1.1, 1.1])
        
        
        self.ax3.set_ylim([-1.1, 1.1])
        
        self.ax4.set_ylim([-1.1, 1.1])

        # Adjust subplot spacing
        plt.tight_layout()
    
    def render_diffusion(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        """Renders the environment as 3D zonotope world with two subplots.
        Args:
            mode (str): the mode to render with
        """
        # Create figure with subplots if not already created
        if self.fig is None:
            self.initialize_renderer_diffusion()
        
        #################################################
        # Draw at Axis 1
        #################################################
        self.ax = self.ax1
        
        self.draw_env(mode, height, width, camera_name)

        self.draw_traj2D_in_kwargs_if_exists(kwargs, "plan_next_mean", 
                                             marker     = self.render_setting["plan_next_mean"]["marker"], 
                                             markersize = self.render_setting["plan_next_mean"]["markersize"])
        self.draw_traj2D_in_kwargs_if_exists(kwargs, "plan_next", 
                                             marker = self.render_setting["plan_next"]["marker"],
                                             markersize = self.render_setting["plan_next"]["markersize"])
        
        self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "plan_next_samples", weights = kwargs["K_weights"])
        self.draw_multivel2D_in_kwargs_if_exists(kwargs, "state_next_samples")
        
        
        self.adjust_camera(camera_name)
        if "follow_robot" in kwargs.keys() and kwargs["follow_robot"]:
            robot_center = np.array(list(self.geom_table["zonotope"][self.geom_table["robot"]])[-1].center)
            self.adjust_camera_center(center=robot_center)
            
        #################################################
        
        #################################################
        # Draw at Axis 2
        #################################################
        self.ax = self.ax2
        # self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "naction_samples", marker='o')
        # self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "action_samples", marker='o')
        # self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "time_ax_samples", marker='o')
        # self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "time_ay_samples", marker='o')
        # self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "time_ay_samples", marker='o')
        self.draw_multiscatter2D_in_kwargs_if_exists(kwargs, "K_samples", weights=kwargs["K_weights"])
        self.draw_traj2D_in_kwargs_if_exists(kwargs, "K_next", marker = self.render_setting["K_next"]["marker"])
        self.draw_traj2D_in_kwargs_if_exists(kwargs, "K_mean", marker = self.render_setting["K_mean"]["marker"])
        
        self.ax = self.ax3
        self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "time_ax_samples", marker='o', weights = kwargs["K_weights"])
        
        self.ax = self.ax4
        self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "time_ay_samples", marker='o', weights = kwargs["K_weights"])
        
        
        # Cheese
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        return img
    
    # def render_diffusion(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
    #     """Renders the environment as 3D zonotope world.

    #     Args:
    #         mode (str): the mode to render with
    #     """
    #     self.draw_env(mode, height, width, camera_name)

    #     #################################################
    #     # Diffusion
    #     #################################################
    #     self.draw_traj2D_in_kwargs_if_exists(kwargs, "plan_next_mean")
    #     self.draw_traj2D_in_kwargs_if_exists(kwargs, "plan_next")
    #     self.draw_multitraj2D_in_kwargs_if_exists(kwargs, "plan_next_samples")
    #     self.draw_multivel2D_in_kwargs_if_exists(kwargs, "state_next_samples")  # Add this line
    #     #################################################
    #     self.adjust_camera(camera_name)

    #     if "follow_robot" in kwargs.keys() and kwargs["follow_robot"]:
    #         robot_center = np.array(list(self.geom_table["zonotope"][self.geom_table["robot"]])[-1].center)
    #         self.adjust_camera_center(center = robot_center)

    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()

    #     img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    #     if "intervened" in kwargs.keys() and kwargs["intervened"]:
    #         img = self.add_circle(img, color = (0, 255, 0)) # if intervened, add green circle
    #     else:
    #         img = self.add_circle(img, color = (0, 0, 255)) # if not intervened, add blue circle
        
    #     return img

    def adjust_camera(self, camera_name):
        # camera view
        if camera_name == "agentview":
            self.ax.view_init(elev=40, azim=-30)
        elif camera_name == "frontview":
            self.ax.view_init(elev=5, azim=0)
        elif camera_name == "topview":
            self.ax.view_init(elev=90, azim=0)
        else:
            raise NotImplementedError
    
    # Exporter function
    def export_figure_to_json(self, filename):
        """
        Export the figure data to a JSON file.
        
        Args:
            filename (str): The name of the JSON file to save the data to.
        """
        if self.fig is None or self.ax is None:
            raise ValueError("Figure has not been initialized. Call custom_render first.")
        
        obs = self.get_observation()
        
        data = {
            "figure": {
                "size": self.fig.get_size_inches().tolist(),
            },
            "axes": {
                "xlim": self.ax.get_xlim(),
                "ylim": self.ax.get_ylim(),
                "zlim": self.ax.get_zlim(),
                "view": {
                    "elev": self.ax.elev,
                    "azim": self.ax.azim
                }
            },
            "zonotopes": {},
            "trajectories": {},
            "robot0_joint_pos": obs["robot0_joint_pos"],
            "robot0_gripper_qpos": abs(obs["robot0_gripper_qpos"])[0],
            "robot0_eef_pos": obs["robot0_eef_pos"]
        }

        # Export zonotope data
        for key, patches in self.patches.items():
            if patches is not None:
                data["zonotopes"][key] = self._extract_poly3d_data(patches)

        # Export trajectory data
        # for key, plots in self.plots.items():
        #     if plots is not None:
        #         data["trajectories"][key] = self._extract_line_data(plots)

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)

    def _extract_poly3d_data(self, poly3d):
        """Extract data from a Poly3DCollection object more efficiently."""
        if isinstance(poly3d, Poly3DCollection):
            data = {
                "vertices": poly3d._vec.tolist(),
            }
            
            # Handle facecolors
            facecolors = poly3d.get_facecolors()
            if len(facecolors) > 0:
                unique_colors = np.unique(facecolors, axis=0)
                if len(unique_colors) == 1:
                    data["facecolor"] = mcolors.to_hex(unique_colors[0])
                else:
                    data["facecolors"] = [mcolors.to_hex(c) for c in facecolors]

            # Handle edgecolors
            edgecolors = poly3d.get_edgecolors()
            if len(edgecolors) > 0:
                unique_colors = np.unique(edgecolors, axis=0)
                if len(unique_colors) == 1:
                    data["edgecolor"] = mcolors.to_hex(unique_colors[0])
                else:
                    data["edgecolors"] = [mcolors.to_hex(c) for c in edgecolors]

            # Handle linewidths
            linewidths = poly3d.get_linewidths()
            if np.isscalar(linewidths) or (isinstance(linewidths, np.ndarray) and linewidths.size == 1):
                data["linewidth"] = float(linewidths)
            else:
                data["linewidths"] = linewidths.tolist()

            # Handle alpha
            alpha = poly3d.get_alpha()
            if alpha is not None:
                data["alpha"] = alpha

            return data
        return None

    def _extract_line_data(self, lines):
        """Extract data from a Line3DCollection or list of Line2D objects."""
        if isinstance(lines, Line3DCollection):
            data = {
                "type": "Line3DCollection",
                "segments": lines.get_segments().tolist()
            }
            
            # Handle colors
            colors = lines.get_colors()
            if colors.size > 0:
                unique_colors = np.unique(colors, axis=0)
                if len(unique_colors) == 1:
                    data["color"] = mcolors.to_hex(unique_colors[0])
                else:
                    data["colors"] = [mcolors.to_hex(c) for c in colors]

            # Handle linewidths
            linewidths = lines.get_linewidths()
            if np.isscalar(linewidths) or (isinstance(linewidths, np.ndarray) and linewidths.size == 1):
                data["linewidth"] = float(linewidths)
            else:
                data["linewidths"] = linewidths.tolist()

            # Handle linestyles
            linestyles = lines.get_linestyles()
            if isinstance(linestyles, str) or len(set(linestyles)) == 1:
                data["linestyle"] = linestyles[0] if isinstance(linestyles, list) else linestyles
            else:
                data["linestyles"] = linestyles

            return data

        elif isinstance(lines, list) and all(isinstance(line, Line2D) for line in lines):
            return [{
                "type": "Line2D",
                "xdata": line.get_xdata().tolist(),
                "ydata": line.get_ydata().tolist(),
                "zdata": line.get_3d_properties().get('z', []),
                "color": mcolors.to_hex(line.get_color()),
                "linewidth": line.get_linewidth(),
                "linestyle": line.get_linestyle(),
            } for line in lines]

        return None
    
class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)