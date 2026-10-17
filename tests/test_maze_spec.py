"""Custom wall paddings and maze layouts (``jm2d_maze.tasks.get_maze_spec``); the paper pads are unchanged. CPU only."""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from d4rl.pointmaze import maze_model  # noqa: E402

from jm2d_maze.tasks import PADS, get_maze_spec, pad_key, padded_maze_spec, read_maze_layout  # noqa: E402

SHIPPED = {0.0: maze_model.LARGE_MAZE, 0.05: maze_model.LARGE_MAZE_PAD005, 0.10: maze_model.LARGE_MAZE_PAD010,
           0.15: maze_model.LARGE_MAZE_PAD015, 0.20: maze_model.LARGE_MAZE_PAD020, 0.25: maze_model.LARGE_MAZE_PAD025}


@pytest.mark.parametrize("pad", PADS)
def test_paper_pads_resolve_to_shipped_specs(pad):
    assert get_maze_spec(pad) == SHIPPED[pad]
    assert get_maze_spec(pad + 1e-12) == SHIPPED[pad]                       # float round-off, e.g. 0.1 + 0.05
    assert get_maze_spec(str(pad)) == SHIPPED[pad]
    assert padded_maze_spec(maze_model.LARGE_MAZE, pad) == SHIPPED[pad]   # custom pads are built the same way
    assert pad_key(pad) == f"pad{int(round(pad * 100)):03d}"               # result folder names unchanged


def test_custom_pad():
    spec = get_maze_spec(0.12)
    assert spec == maze_model.LARGE_MAZE + "PAD0.12"
    model = maze_model.MazeEnv(maze_spec=spec).sim.model
    assert model.geom_size[model.geom_name2id("wall_0_0")][:2].tolist() == pytest.approx([0.62, 0.62])
    assert pad_key(0.12) == "pad012" and pad_key(0.125) == "pad0125" and pad_key(0.005) == "pad0005"
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            get_maze_spec(bad)


def test_custom_layout(tmp_path):
    rows = maze_model.LARGE_MAZE.split("\\")
    rows[2] = rows[2][:10] + "#" + rows[2][11:]
    path = tmp_path / "my_maze.txt"
    path.write_text("\n".join(rows) + "\n")
    layout = read_maze_layout(str(path))
    assert get_maze_spec(0.0, layout) == layout
    assert get_maze_spec(0.12, layout) == layout + "PAD0.12"
    model = maze_model.MazeEnv(maze_spec=get_maze_spec(0.12, layout)).sim.model
    assert model.geom_size[model.geom_name2id("wall_2_10")][:2].tolist() == pytest.approx([0.62, 0.62])  # new wall
    with pytest.raises(ValueError):
        get_maze_spec(0.0, "\\".join(rows[:-1]))                         # wrong size
    with pytest.raises(ValueError):
        get_maze_spec(0.0, layout.replace("O", "x", 1))                  # unknown tile
