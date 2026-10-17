import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
from tqdm import tqdm

def formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices):
    """
    Construct y <= mx + b from two points and a direction, operates on x and y coordinates

    Args
        constraint: list of [(x1, y1), (x2, y2), 'above' or 'below']
        enlarge_constraints: float
        trajectory_dim: int
        act_obs_indices: dict

    Returns
        C_row: np.ndarray
        d: float

    """
    m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
    n = [-1, 1/m] / np.linalg.norm([-1, 1/m])
    if (m > 0 and constraint[2] == 'below') or (m < 0 and constraint[2] == 'above'):
        n *= -1
    points_enlarged = [constraint[0] + enlarge_constraints * n, constraint[1] + enlarge_constraints * n]
    d = points_enlarged[0][1] - m * points_enlarged[0][0]
    # d = constraint[0][1] - m * constraint[0][0]
    C_row = np.zeros(trajectory_dim)
    if constraint[2] == 'below':
        C_row[act_obs_indices['x']] = -m
        C_row[act_obs_indices['y']] = 1
    elif constraint[2] == 'above':
        C_row[act_obs_indices['x']] = m
        C_row[act_obs_indices['y']] = -1
        d *= -1
    return C_row, d

def formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices):
    """
    Construct lower and upper bounds from a list of bounds

    Args
        constraint_types: list
        bounds: list
    """
    lower_bound = -np.inf * np.ones(trajectory_dim)
    upper_bound = np.inf * np.ones(trajectory_dim)
    if 'bounds' in constraint_types:
        for bound in bounds:
            for dim_idx, dim in enumerate(bound['dimensions']):
                if bound['type'] == 'lower' and dim in act_obs_indices:
                    lower_bound[act_obs_indices[dim]] = bound['values'][dim_idx]
                elif bound['type'] == 'upper' and dim in act_obs_indices:
                    upper_bound[act_obs_indices[dim]] = bound['values'][dim_idx]
    return lower_bound, upper_bound

def formulate_dynamics_constraints(exp, act_obs_indices, action_dim):
    dynamic_constraints = []
    if 'pointmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
        ]
    if 'antmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
            ('deriv', np.array([act_obs_indices['z'], act_obs_indices['vz']])),
        ]
    if 'avoiding' in exp and action_dim > 0:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
            ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),
        ]
    return dynamic_constraints

# Plotting
def plot_environment_constraints(exp, ax, flip_xy=False):
    if exp == 'pointmaze-umaze-dense-v2':
        ax.add_patch(matplotlib.patches.Rectangle((-1.5, -0.5), 2, 1, color='k', alpha=0.2))
    if exp == 'pointmaze-medium-dense-v2':
        bottom_left_corners = [[-1, 2], [0, 2], [-1, 1], [-3, 0], [1, 0], [2, 0], [-1, -1], [-2, -2], [1, -2], [0, -3]]
        for corner in bottom_left_corners:
            ax.add_patch(matplotlib.patches.Rectangle((corner[0], corner[1]), 1, 1, color='k', alpha=0.2))
    elif exp == 'antmaze-umaze-v1':
        ax.add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))
    elif exp == 'avoiding-d3il':
        centers = [[0.5, -0.1], [0.425, 0.08], [0.575, 0.08], [0.35, 0.26], [0.5, 0.26], [0.65, 0.26]]
        for center in centers:
            if flip_xy:
                ax.add_patch(matplotlib.patches.Circle(center[::-1], 0.025, color='r'))
            ax.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
        ax.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)

def plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits, flip_xy=False):
    for constraint in polytopic_constraints:
        mat = np.vstack((constraint[:2], np.zeros(2)))
        if 'pointmaze' in exp:
            mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
        elif 'antmaze' in exp:
            mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
        elif 'avoiding' in exp:
            # Works for triangles with two vertices on the negative y-axis
            slope = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
            if slope > 0 and constraint[2] == 'above':
                mat[2] = np.array([ax_limits[0][1], ax_limits[1][0]])
            elif slope > 0 and constraint[2] == 'below':
                mat[2] = np.array([ax_limits[0][0], ax_limits[1][1]])
            elif slope < 0 and constraint[2] == 'above':
                mat[2] = np.array([ax_limits[0][0], ax_limits[1][0]])
            elif slope < 0 and constraint[2] == 'below':
                mat[2] = np.array([ax_limits[0][1], ax_limits[1][1]])
        if flip_xy:
            mat = mat[:, ::-1]
        ax.add_patch(matplotlib.patches.Polygon(mat, color='b', alpha=0.2))


def visualize_diffusion_trajectory(diffusion_trajectory, 
                                 diffusion_logs,
                                 exp,
                                 constraint_types,
                                 obstacle_constraints, 
                                 polytopic_constraints,
                                 obs_indices,
                                 ax_limits,
                                 save_path=None,
                                 cmap_name='viridis',
                                 alpha_range=(0.6, 1.0),
                                 start_val=0.3,
                                 end_val=1.0):
    """
    Visualize a diffusion trajectory as an animation.

    Args:
        diffusion_trajectory: list of numpy arrays, each of shape (B, DT+1, H, A)
        diffusion_logs: list of dicts containing projection info and optional clean trajectories
        exp: str, the name of the experiment
        constraint_types: list of str, the types of constraints to visualize
        obstacle_constraints: list of dicts, each containing 'center' and 'radius'
        polytopic_constraints: list of numpy arrays, each of shape (N, 2)
        obs_indices: dict, the indices of the observation space
        ax_limits: tuple of floats, the limits of the plot
        save_path: str, optional path to save the animation
        cmap_name: str, name of colormap to use for clean trajectories
        alpha_range: tuple of (min_alpha, max_alpha) to control transparency range
        start_val: float between 0 and 1, starting value for main line color gradient
        end_val: float between 0 and 1, ending value for main line color gradient
    """
    PT = len(diffusion_trajectory)
    B, DT, H, A = diffusion_trajectory[0].observations.shape
    
    plot_batch_idx = 0

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Add text annotation for PT/DT display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top')
    
    # Create colormap
    cmap = plt.get_cmap(cmap_name)
    cmap_noisy = plt.get_cmap('Blues')
    
    # Full trajectory
    full_trajectory = [diffusion_trajectory[i].observations[plot_batch_idx] for i in range(PT)]
    full_trajectory = np.concatenate(full_trajectory, axis=0)[..., 0, :] # (PT, DT, A)
    ax.plot(full_trajectory[:, obs_indices['x']], full_trajectory[:, obs_indices['y']], 'k', '--', label='Full trajectory')

    # Plot static environment constraints
    plot_environment_constraints(exp, ax)

    if 'halfspace' in constraint_types:
        plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits)

    if 'obstacles' in constraint_types:
        for constraint in obstacle_constraints:
            ax.add_patch(matplotlib.patches.Circle(
                constraint['center'], 
                constraint['radius'], 
                color='b', 
                alpha=0.2
            ))
    
    # Set axis limits
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
    
    # Initialize line objects for animation
    main_line_markers = []
    main_line_segments = []
    for _ in range(H):
        marker, = ax.plot([], [], 'o', alpha=0.5, markersize=8)
        # segment, = ax.plot([], [], '-', alpha=0.5)
        main_line_markers.append(marker)
        # main_line_segments.append(segment)
    
    start_point, = ax.plot([], [], 'go')
    
    # Initialize clean trajectory lines (will be updated if clean trajectories exist)
    max_clean_trajectories = 100  # Limit the number of trajectories to plot
    clean_lines = [ax.plot([], [], alpha=0.3)[0] for _ in range(max_clean_trajectories)]
    plot_clean_trajectories = False
    if 'clean_trajectory' in diffusion_logs[0].keys() and 'weights' in diffusion_logs[0].keys():
        plot_clean_trajectories = True
    
    def init():
        """Initialize animation"""
        for marker in main_line_markers:
            marker.set_data([], [])
        # for segment in main_line_segments:
        #     segment.set_data([], [])
        start_point.set_data([], [])
        time_text.set_text('')
        for line in clean_lines:
            line.set_data([], [])
        return main_line_markers + [start_point, time_text] + clean_lines

    def animate(frame):
        """Animation function called for each frame"""
        pt_idx = frame // DT
        dt_idx = frame % DT
        
        # Clear previous lines
        for marker in main_line_markers:
            marker.set_data([], [])
        for segment in main_line_segments:
            segment.set_data([], [])
        start_point.set_data([], [])
        for line in clean_lines:
            line.set_data([], [])

        # Update main trajectory
        trajectory = diffusion_trajectory[pt_idx].observations[plot_batch_idx][dt_idx]
        x_coords = trajectory[:, obs_indices['x']]
        y_coords = trajectory[:, obs_indices['y']]
        
        # Update main line with color gradient
        for i in range(H):
            color_val = start_val + (end_val - start_val) * (i / (H-1))
            color = cmap_noisy(color_val)
            
            main_line_markers[i].set_data([x_coords[i]], [y_coords[i]])
            main_line_markers[i].set_color(color)
            
            # if i < H-1:
            #     main_line_segments[i].set_data(
            #         [x_coords[i], x_coords[i+1]], 
            #         [y_coords[i], y_coords[i+1]]
            #     )
            #     main_line_segments[i].set_color(color)
        
        # Update start point
        start_point.set_data(
            [trajectory[0, obs_indices['x']]], 
            [trajectory[0, obs_indices['y']]]
        )

        # parse projection flag
        if dt_idx == 0:
            projection_flag = False
        else:
            projection_flag = diffusion_logs[pt_idx]['projection_flag'][plot_batch_idx][dt_idx-1]

        # parse diffusion logs
        if (dt_idx > 0) and (dt_idx < DT-1) and plot_clean_trajectories:
            # Plot clean trajectories if available
            clean_trajectories = diffusion_logs[pt_idx]['clean_trajectory'][plot_batch_idx][dt_idx-1] # (N, H, A)
            clean_weights = diffusion_logs[pt_idx]['weights'][plot_batch_idx][dt_idx-1] # (N)
            
            # Normalize weights to [0, 1] for colormapping
            if (clean_weights == 0).all():
                normalized_weights = np.zeros(clean_weights.shape)
            else:
                normalized_weights = (clean_weights - clean_weights.min() + 1e-8) / (clean_weights.max() - clean_weights.min() + 1e-8)
            
            # Plot up to max_clean_trajectories trajectories
            for i, (traj, weight) in enumerate(zip(clean_trajectories[:max_clean_trajectories], 
                                                normalized_weights[:max_clean_trajectories])):
                # Scale the alpha value between min_alpha and max_alpha
                alpha = alpha_range[0] + weight * (alpha_range[1] - alpha_range[0])
                
                clean_lines[i].set_data(
                    traj[:, obs_indices['x']], 
                    traj[:, obs_indices['y']]
                )
                color = cmap(weight)
                clean_lines[i].set_color(color)
                clean_lines[i].set_alpha(alpha)  # Set the scaled alpha value

        # Update time text
        time_text.set_text(f'PT: {pt_idx+1}/{PT}, DT: {dt_idx+1}/{DT}, Projection: {bool(projection_flag)}')

        return main_line_markers + [start_point, time_text] + clean_lines

    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=PT * DT,
        interval=100,
        blit=True
    )

    # Add colorbar if clean trajectories are available
    if 'clean_trajectory' in diffusion_logs[0]:
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Make space for colorbar by adjusting the main axes
        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        fig.colorbar(sm, cax=cax, label='Normalized weights')

    # Save animation
    if save_path is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        
        if not save_path.endswith('.mp4'):
            save_path = save_path + '.mp4'
            print(f"Adding .mp4 extension: {save_path}")

        # Set up the progress bar
        print("Saving animation...")
        total_frames = PT * DT
        progress_callback = lambda i, n: pbar.update(1)
        
        with tqdm(total=total_frames, desc="Rendering") as pbar:
            try:
                anim.save(save_path, 
                         writer=writer, 
                         progress_callback=progress_callback)
                print(f"Animation saved successfully to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
    
    plt.close()
    
    return anim

def prepare_visualization_data(diffusion_trajectory, 
                             diffusion_logs,
                             exp,
                             constraint_types,
                             obstacle_constraints, 
                             polytopic_constraints,
                             obs_indices,
                             ax_limits):
    """
    Prepare minimal visualization data for later video creation.
    """
    # Only save the raw data needed for visualization
    return {
        'diffusion_trajectory': diffusion_trajectory,
        'diffusion_logs': diffusion_logs,
        'exp': exp,
        'constraint_types': constraint_types,
        'obstacle_constraints': obstacle_constraints,
        'polytopic_constraints': polytopic_constraints,
        'obs_indices': obs_indices,
        'ax_limits': ax_limits
    }

def render_from_pickle(pickle_path, save_path=None, cmap_name='viridis', alpha_range=(0.6, 1.0)):
    """
    Render video from saved visualization data.
    """
    import pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return visualize_diffusion_trajectory(
        data['diffusion_trajectory'],
        data['diffusion_logs'],
        data['exp'],
        data['constraint_types'],
        data['obstacle_constraints'],
        data['polytopic_constraints'],
        data['obs_indices'],
        data['ax_limits'],
        save_path=save_path,
        cmap_name=cmap_name,
        alpha_range=alpha_range
    )