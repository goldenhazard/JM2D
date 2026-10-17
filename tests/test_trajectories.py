"""Trajectory files and the top-view figure of ``scripts/run.py --plot``, without a GPU."""
import os

import numpy as np
import pytest

from jm2d_d3il import tasks
from jm2d_d3il.plotting import count_outcomes, plot_run
from jm2d_d3il.results import read_traj, traj_row, write_traj


class _Diffusion:
    """Index maps of the Avoiding planners: action (vx, vy) + observation (x_des, y_des, x, y), no goal."""
    transition_dim, goal_dim, action_dim = 6, 0, 2


@pytest.mark.parametrize("scenario", tasks.SCENARIOS)
def test_xy_constraints_are_the_rows_the_runner_checks(scenario):
    task = tasks.make(scenario)
    rows, obstacles = task.evaluation_constraints(_Diffusion())
    halfspaces, discs = task.xy_constraints()
    assert len(rows) == len(halfspaces)
    ix, iy = task.observation_indices["x"], task.observation_indices["y"]
    for (kind, (c, d)), (a, b, d_xy) in zip(rows, halfspaces):
        c = c[_Diffusion.action_dim:]           # the runner checks obs @ c[action_dim:] >= d
        assert kind == "ineq" and (c[ix], c[iy], d) == (a, b, d_xy)
        assert not np.delete(c, [ix, iy]).any()
    np.testing.assert_array_equal(discs, [[*o["center"], o["radius"]] for o in obstacles])


def test_traj_file_roundtrip_and_plot(tmp_path):
    task = tasks.make("cluttered")
    halfspaces, discs = task.xy_constraints()
    meta = dict(label=np.array("jm2d"), scenario=np.array("cluttered"), seed=np.array(0), halfspaces=halfspaces,
                discs=discs, xlim=np.array(task.ax_limits[0], float), ylim=np.array(task.ax_limits[1], float))
    path = np.array([[0.525, -0.28], [0.52, -0.2], [0.45, 0.1], [0.43, 0.36]])
    results = [dict(n_success=1.0, n_success_and_constraints=1.0, n_violations=0.0, n_steps=2.0),
               dict(n_success=1.0, n_success_and_constraints=0.0, n_violations=3.0, n_steps=2.0),
               dict(n_success=0.0, n_success_and_constraints=0.0, n_violations=0.0, n_steps=2.0)]
    rows = {t: traj_row(path + 0.01 * t, r) for t, r in enumerate(results)}
    traj_file = str(tmp_path / "seed0.traj.npz")
    write_traj(traj_file, meta, rows)
    meta2, rows2 = read_traj(traj_file)
    assert sorted(rows2) == [0, 1, 2] and str(meta2["scenario"]) == "cluttered"
    np.testing.assert_array_equal(meta2["halfspaces"], halfspaces)
    for t in rows:
        np.testing.assert_array_equal(rows2[t]["path"], rows[t]["path"])
        assert {k: rows2[t][k] for k in rows[t] if k != "path"} == {k: v for k, v in rows[t].items() if k != "path"}
    assert count_outcomes(rows2) == {"ssucc": 1, "violated": 1, "failure": 1}
    png = str(tmp_path / "seed0.png")
    assert plot_run(traj_file, png) == 3 and os.path.getsize(png) > 0
