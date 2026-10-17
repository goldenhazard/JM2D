"""Sample plots for the two tasks: ``plot_joint_samples`` (donut_joint) and ``plot_constrained_samples``
(donut_constrained).

Adapted from the legacy plotting methods (``DonutJointProblem.plot_results`` donut_problem.py:717,
``Donut2DProblem.plot_results`` donut2D_problem.py:642 and ``plot_diffusion`` :804) to draw on a given
axis. The grey data background is drawn from a seeded ``np.random.RandomState`` (legacy: a fresh unseeded
dataset per plot). Marker sizes take a ``marker_scale`` because the legacy sizes assume a 10x10 inch figure.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from jm2d_toy.data import R_INNER, R_OUTER, DonutConstrainedDataset
from jm2d_toy.tasks.donut_joint import sample_points_from_circle

SUCCESS_COLOR = "royalblue"
FAILURE_COLOR = "crimson"


def donut_background(num_samples=20000, seed=0):
    """(num_samples, 2) points from the donut data distribution (seeded, for plot backgrounds)."""
    return DonutConstrainedDataset(num_samples, rng=np.random.RandomState(seed)).all_samples


def _donut_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    for r in (R_INNER, R_OUTER):
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.5)


def plot_joint_samples(ax, problem, x, k, compatible, title=None, marker_scale=1.0, background=True):
    """donut_joint samples (paper Fig. 2): x = (p1, p2) as circles, k as a star; blue = compatible (constraint
    satisfied), red = not.

    Args:
        problem: ``DonutJointProblem`` (for the obstacles)
        x: (B, 4), k: (B, 2), compatible: (B,) bool
    """
    x = np.asarray(x)
    point1, point2 = x[:, :2], x[:, 2:]
    ks = torch.tensor(np.asarray(k), dtype=torch.float32)
    center_plot = sample_points_from_circle(ks[:, 0], ks[:, 1]).cpu().numpy()

    if background:
        bg = donut_background()
        ax.scatter(bg[:, 0], bg[:, 1], c='grey', alpha=0.1, s=10 * marker_scale ** 0.5, edgecolors='none')

    for i in range(len(point1)):
        color = SUCCESS_COLOR if compatible[i] else FAILURE_COLOR
        ax.scatter(point1[i, 0], point1[i, 1], marker='o', s=1000 * marker_scale, color=color, zorder=10)
        ax.scatter(point2[i, 0], point2[i, 1], marker='o', s=1000 * marker_scale, color=color, zorder=10)
        ax.scatter(center_plot[i, 0], center_plot[i, 1], marker='*', s=2000 * marker_scale, color=color, zorder=10)
        if compatible[i]:
            ax.plot([center_plot[i, 0], point1[i, 0]], [center_plot[i, 1], point1[i, 1]], 'k', linestyle='--')
            ax.plot([center_plot[i, 0], point2[i, 0]], [center_plot[i, 1], point2[i, 1]], 'k', linestyle='--')

    for obs_center, obs_radius in problem.obstacles:
        ax.add_artist(plt.Circle(obs_center.cpu().numpy(), obs_radius, color='red', fill=True, alpha=0.3))

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_constrained_samples(ax, problem, x, valid, title=None, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), legend=True):
    """donut_constrained samples (paper Fig. 9): valid samples (green), invalid samples (red x), obstacle (red),
    data support (dashed)."""
    x = np.asarray(x)
    valid = np.asarray(valid, dtype=bool)

    bg = donut_background()
    ax.scatter(bg[:, 0], bg[:, 1], c='grey', alpha=0.1, s=10, edgecolors='none', label='Training Data')

    for obstacle in problem.obstacles:
        obstacle.plot(ax, alpha=0.15)
    obstacle_patch = Patch(facecolor='red', edgecolor='red', alpha=0.15, label='Obstacle Region')

    if problem.center is not None:
        c = problem.center.detach().cpu().numpy()
        ax.scatter(c[0], c[1], c='k', marker='*', s=150, label='Target')

    if valid.any():
        ax.scatter(x[valid, 0], x[valid, 1], c='green', edgecolors='black', s=30, marker='o', label='Valid')
    if (~valid).any():
        ax.scatter(x[~valid, 0], x[~valid, 1], c='red', s=80, marker='x', label='Invalid')

    _donut_boundary(ax)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=14)
    ax.set_ylabel(r'$\mathbf{y}$', fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(obstacle_patch)
        labels.append(obstacle_patch.get_label())
        ax.legend(handles, labels, loc='upper right', framealpha=0.9, fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_diffusion_trajectories(ax, problem, trajectories, xlim=(-3, 3), ylim=(-3, 3), cmap='viridis',
                                alpha_background=0.1):
    """Paper Fig. 9(c): reverse-diffusion trajectories (B, T, 2) coloured by step; final samples as red stars."""
    bg = donut_background()
    ax.scatter(bg[:, 0], bg[:, 1], c='grey', alpha=alpha_background, s=10, edgecolors='none')
    _donut_boundary(ax)
    for obstacle in problem.obstacles:
        obstacle.plot(ax, alpha=0.15, xlim=xlim, ylim=ylim)

    trajectories = np.asarray(trajectories)
    num_samples, num_steps, _ = trajectories.shape
    colormap = mpl.colormaps[cmap]
    colors = [colormap(t / float(num_steps - 1)) for t in range(num_steps)]
    for i in range(num_samples):
        traj = trajectories[i]
        if num_steps > 2:
            ax.scatter(traj[1:-1, 0], traj[1:-1, 1], color=colors[1:-1], s=50, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[0], s=100, marker='s', zorder=4)
        ax.scatter(traj[-1, 0], traj[-1, 1], color="red", s=300, marker='*', zorder=4)
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        ax.add_collection(LineCollection(segments, colors=colors[:-1], linewidth=0.5))

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=mpl.colors.Normalize(vmin=0, vmax=num_steps - 1))
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax).set_label('Diffusion step')
    ax.legend(handles=[Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10,
                              label='Final Sample'),
                       Patch(facecolor='red', edgecolor='red', alpha=0.15, label='Obstacle')],
              loc='upper right', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    return ax


plot_fig2_panel, plot_fig9_panel = plot_joint_samples, plot_constrained_samples
