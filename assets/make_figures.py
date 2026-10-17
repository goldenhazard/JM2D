#!/usr/bin/env python
"""README figures: ``assets/donut_joint.png`` and ``assets/donut_constrained.png`` (needs a CUDA GPU).

    python assets/make_figures.py

Each figure: (a) training data, (b) the test-time constraint, (c) JM2D samples (seed 0, configs/*.json).
"""
import io
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import jm2d_toy  # noqa: E402,F401  (sets CUBLAS_WORKSPACE_CONFIG before CUDA starts)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from jm2d_toy import samplers, tasks  # noqa: E402
from jm2d_toy.config import load_config  # noqa: E402
from jm2d_toy.data import DonutJointDataset  # noqa: E402
from jm2d_toy.plotting import (FAILURE_COLOR, SUCCESS_COLOR, donut_background,  # noqa: E402
                               plot_constrained_samples, plot_joint_samples)
from jm2d_toy.seeding import enable_determinism  # noqa: E402
from jm2d_toy.tasks.donut_joint import sample_points_from_circle  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 0
MARKER_SCALE = 0.13  # plot_joint_samples marker sizes for a 4-inch panel
PAIR_COLOR = "dimgray"


def jm2d_samples(task):
    return samplers.from_config(task, "jm2d").sample(tasks.make(task), n=load_config(task)["n"], seed=SEED)


def save(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    Image.open(buf).convert("RGB").quantize(256).save(path, optimize=True)  # palette PNG: small file
    return path


def joint_figure(path):
    problem = tasks.make("donut_joint", device="cpu")
    pairs = DonutJointDataset(5000, rng=np.random.RandomState(SEED)).all_samples
    k_param = np.array([[1 / 6, 0.5]])  # k = (-0.1, 0): |r| * 0.6 * (cos 2 pi theta, sin 2 pi theta)
    k_point = sample_points_from_circle(k_param[:, 0], k_param[:, 1]).numpy()[0]
    ok = problem.is_valid(problem.to_tensor(pairs), problem.to_tensor(np.repeat(k_param, len(pairs), 0))).numpy()
    example = pairs[[np.flatnonzero(ok)[0], np.flatnonzero(~ok)[0]]]  # one compatible pair, one not
    res = jm2d_samples("donut_joint")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0), constrained_layout=True)
    ax = axes[0]
    bg = donut_background()
    ax.scatter(bg[:, 0], bg[:, 1], c='grey', alpha=0.1, s=10 * MARKER_SCALE ** 0.5, edgecolors='none')
    for p in pairs[:3]:
        ax.plot(p[[0, 2]], p[[1, 3]], color=PAIR_COLOR, lw=1, ls=':', marker='o', ms=9)
    ax.annotate(r"$p_1$", pairs[0, :2], xytext=(8, 6), textcoords="offset points", fontsize=12)
    ax.annotate(r"$p_2$", pairs[0, 2:], xytext=(8, 6), textcoords="offset points", fontsize=12)
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect='equal', xticks=[], yticks=[])
    ax.set_title("(a) Data: two points on the donut")
    ax.set_xlabel(r"training pairs $x = (p_1, p_2)$")

    ax = axes[1]
    plot_joint_samples(ax, problem, example, np.repeat(k_param, 2, 0), [True, False],
                       title="(b) Test-time constraint", marker_scale=MARKER_SCALE)
    for p in example[1:]:  # draw the segments of the incompatible pair too
        for q in (p[:2], p[2:]):
            ax.plot([k_point[0], q[0]], [k_point[1], q[1]], color=FAILURE_COLOR, ls='--')
    ax.scatter(*k_point, marker='*', s=2000 * MARKER_SCALE, color=PAIR_COLOR, zorder=11)  # k is shared
    ax.add_artist(plt.Circle((0, 0), 0.6, fill=False, color='k', ls=':', lw=1))
    ax.annotate(r"$k$", k_point, xytext=(-14, -16), textcoords="offset points", fontsize=12)
    ax.set_xlabel(r"segments $k$–$p_1$ and $k$–$p_2$ must avoid the obstacles")

    ax = axes[2]
    n = len(res.x)
    plot_joint_samples(ax, problem, res.x, res.k, res.valid, marker_scale=MARKER_SCALE,
                       title=f"(c) JM2D samples: {int(res.valid.sum())}/{n} compatible")
    ax.set_xlabel(f"{n} joint samples of $(x, k)$, seed {SEED}")

    handles = [Line2D([], [], ls='', marker='o', color='grey', alpha=0.4, label="training data"),
               Patch(color='red', alpha=0.3, label="obstacle"),
               Line2D([], [], ls='', marker='o', color=PAIR_COLOR, label=r"data $p_1, p_2$"),
               Line2D([], [], ls='', marker='*', ms=12, color=PAIR_COLOR, label=r"parameter $k$ ($|k| \leq 0.6$)"),
               Patch(color=SUCCESS_COLOR, label="compatible"),
               Patch(color=FAILURE_COLOR, label="not compatible")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    return save(fig, path), res


def constrained_figure(path):
    problem = tasks.make("donut_constrained", device="cpu")
    no_obstacle = tasks.DonutConstrainedProblem(center=None, obs_types=[], obs_centers=np.zeros((0, 2)),
                                                obs_radius=np.zeros(0), device="cpu")
    res = jm2d_samples("donut_constrained")
    empty = np.zeros((0, 2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    plot_constrained_samples(axes[0], no_obstacle, empty, [], legend=False, title="(a) Data: one point on the donut")
    plot_constrained_samples(axes[1], problem, empty, [], legend=False, title="(b) Test-time constraint")
    axes[1].text(-0.2, 1.22, "obstacle (x must avoid it)", ha="center", va="center", color="darkred")
    n = len(res.x)
    plot_constrained_samples(axes[2], problem, res.x, res.valid,
                             title=f"(c) JM2D samples: {int(res.valid.sum())}/{n} valid")
    return save(fig, path), res


if __name__ == "__main__":
    enable_determinism()
    for task, make in (("donut_joint", joint_figure), ("donut_constrained", constrained_figure)):
        path, res = make(os.path.join(OUT_DIR, task + ".png"))
        print(f"{path}: {int(res.valid.sum())}/{len(res.x)} valid, {os.path.getsize(path) / 1024:.0f} KB")
