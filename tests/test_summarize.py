"""The stored paper runs (results/paper/cummulative_data.json) reproduce the Table 2 DPCC / Ours cells."""
import os
import subprocess
import sys

from jm2d_d3il import summarize as S

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_stored_table2_matches_paper():
    table = S.table2({}, S.load_stored())
    targets = S.load_targets()
    rows = [r for r in S.check_paper(table, targets["table2"], "table2", targets.get("printed_decimals"))
            if r[0].split(".")[1] in ("dpcc", "jm2d") and not r[0].endswith("ssucc_std")]
    status = {r[0]: r[3] for r in rows}
    assert len(status) == 16
    close = sorted(k for k, v in status.items() if v == "CLOSE")
    assert close == ["table2.dpcc.cluttered.n_vio", "table2.jm2d.top_left.ssucc"], close
    assert sum(v == "EXACT" for v in status.values()) == 14
    for cid, val, tgt, st, src, n in rows:
        assert abs(val - tgt) <= 0.01 + 1e-9, cid


def test_stored_seed_counts():
    stored = S.load_stored()
    assert stored[("jm2d", "cluttered")]["n_seeds"] == 3
    assert all(c["n_seeds"] == 5 for k, c in stored.items() if k != ("jm2d", "cluttered"))


def test_summarize_cli_runs():
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "summarize.py"), "--table", "2", "--check-paper"],
                         capture_output=True, text=True, check=True).stdout
    assert "EXACT 14" in out and "CLOSE 2" in out


def test_repro_table_rerun_vs_stored_vs_paper(tmp_path):
    """Re-run DPCC / Ours cells are aggregated over seeds and shown next to the stored and paper values."""
    from jm2d_d3il.results import write_summary
    for seed, ssucc in [(0, 0.8), (1, 0.9)]:
        trials = {"n_success": [1.0], "n_success_and_constraints": [ssucc], "n_steps": [60.0], "n_violations": [1.0],
                  "total_violations": [0.0], "avg_time": [1.0], "collision_free_completed": [1.0]}
        write_summary(str(tmp_path / "dpcc" / "top_left" / f"seed{seed}.json"), label="dpcc", planner="dpcc",
                      scenario="top_left", seed=seed, n_trials=1, trials=trials,
                      metrics={"ssucc": ssucc, "n_vio": 1.0, "time_per_step_s": 1.0})
    runs = S.aggregate_runs([str(tmp_path)])
    repro = S.repro_table(runs, S.load_stored(), S.load_targets()["table2"])
    cell = repro["dpcc"]["rerun"]["top_left"]
    assert cell["n_seeds"] == 2 and abs(cell["ssucc"] - 0.85) < 1e-12 and abs(cell["ssucc_std"] - 0.05) < 1e-12
    assert abs(repro["dpcc"]["stored"]["top_left"]["ssucc"] - 0.872) < 1e-9
    assert repro["dpcc"]["paper"]["top_left"]["ssucc"] == 0.87
    assert repro["jm2d"]["rerun"]["top_left"] is None
    text = S.markdown_repro(repro, {"dpcc": "DPCC", "jm2d": "Ours"})
    assert "0.850±0.05" in text and "0.872 [0.81, 0.92]" in text


def test_aggregate_runs_include_partial(tmp_path):
    """--include-partial aggregates a seed with only a trials.jsonl (no finished seed<k>.json), marking the cell."""
    from jm2d_d3il.results import append_trial, write_summary
    trials0 = {"n_success": [1.0], "n_success_and_constraints": [0.5], "n_steps": [60.0], "n_violations": [2.0],
               "total_violations": [0.0], "avg_time": [1.0], "collision_free_completed": [1.0]}
    write_summary(str(tmp_path / "mpd" / "top_left" / "seed0.json"), label="mpd", planner="mpd", scenario="top_left",
                  seed=0, n_trials=1, trials=trials0, metrics={"ssucc": 0.5, "n_vio": 2.0, "time_per_step_s": 1.0})
    jsonl_path = str(tmp_path / "mpd" / "top_left" / "seed1.trials.jsonl")
    for trial, (ssucc, vio) in enumerate([(1.0, 0.0), (0.0, 1.0), (1.0, 2.0)]):
        append_trial(jsonl_path, trial, {"n_success": 1.0, "n_success_and_constraints": ssucc, "n_steps": 10.0,
                                         "n_violations": vio, "total_violations": 0.0, "avg_time": 1.0,
                                         "collision_free_completed": 1.0})

    cell = S.aggregate_runs([str(tmp_path)])[("mpd", "top_left")]
    assert cell["n_seeds"] == 1 and cell.get("partial") is None and abs(cell["ssucc"] - 0.5) < 1e-12

    cell = S.aggregate_runs([str(tmp_path)], include_partial=True)[("mpd", "top_left")]
    assert cell["n_seeds"] == 2 and cell["partial"] is True and cell["n_trials_per_seed"] == {0: 1, 1: 3}
    assert abs(cell["ssucc"] - (0.5 + 2.0 / 3) / 2) < 1e-12
    assert "[2s, 4t, partial]" in S.fmt_cell(cell)

    cell = S.aggregate_runs([str(tmp_path)], include_partial=True, min_partial_trials=4)[("mpd", "top_left")]
    assert cell["n_seeds"] == 1 and cell.get("partial") is None


def test_summarize_cli_shared_flags(tmp_path):
    """--paper / --figure / --out / --check-paper (shared with the other branches) and the old aliases."""
    script = os.path.join(REPO, "scripts", "summarize.py")
    md, js = tmp_path / "t2.md", tmp_path / "t.json"
    out = subprocess.run([sys.executable, script, "--paper", "--figure", "table2", "--check-paper", "--out", str(md)],
                         capture_output=True, text=True, check=True).stdout
    assert "EXACT 14" in out and "CLOSE 2" in out and md.read_text().strip() == out.strip()
    assert "Table 3" not in out
    subprocess.run([sys.executable, script, "--results", str(tmp_path), "--table", "3", "--json", str(js)],
                   capture_output=True, text=True, check=True)
    import json
    assert list(json.loads(js.read_text())) == ["table3"]
    bad = subprocess.run([sys.executable, script, "--paper", "--no-stored"], capture_output=True, text=True)
    assert bad.returncode != 0
