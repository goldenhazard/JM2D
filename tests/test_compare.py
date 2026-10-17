"""``scripts/compare.py`` reproduces Table 1 and the Fig. 4 numbers from the stored paper runs (no GPU needed)."""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from jm2d_maze.compare import fig4_summary, load_paper_runs, load_targets, table1_summary  # noqa: E402


def test_table1_from_stored_runs():
    targets = load_targets()["table1"]["values"]
    summary = table1_summary(load_paper_runs(figure="table1"))
    exact = []
    for planner, cells in targets.items():
        for backup, t in cells.items():
            s = summary[planner][backup]
            assert s["n"] == 50
            exact += [round(s["ssucc"], 2) == t["ssucc"], round(s["shor"], 1) == t["shor"], round(s["int"], 2) == t["int"]]
    assert sum(exact) == 17                                          # 17/18 at the paper's precision
    jm2d_xmym = summary["jm2d"]["xmym"]
    assert (jm2d_xmym["n_intervention"], jm2d_xmym["n_plan"]) == (627, 1312)  # 0.478, printed as 0.47


@pytest.mark.parametrize("metric,planner,tol", [("safe_success", "jm2d", 0.015), ("intervention", "jm2d", 0.015),
                                                ("horizon_s", "dp", 0.15)])
def test_fig4_cells_match_paper(metric, planner, tol):
    targets = load_targets()["fig4"]
    summary = fig4_summary(load_paper_runs(figure="fig4"), pads=targets["pads"])
    for pad, value in zip(targets["pads"], targets["values"][metric][planner]):
        assert abs(summary[metric][planner][pad]["mean"] - value) <= tol, (pad, summary[metric][planner][pad], value)


def test_fig4_qualitative_claims():
    summary = fig4_summary(load_paper_runs(figure="fig4"))
    for pad in summary["intervention"]["jm2d"]:
        jm2d = summary["intervention"]["jm2d"][pad]["mean"]
        assert jm2d < summary["intervention"]["rail"][pad]["mean"]
        assert jm2d < summary["intervention"]["gibbs"][pad]["mean"]
        assert summary["safe_success"]["jm2d"][pad]["mean"] >= summary["safe_success"]["rail"][pad]["mean"]
