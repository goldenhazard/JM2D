# Joint Model-based Model-free Diffusion for Planning with Constraints — PointMaze

[[Project page]](https://jm2d-corl25.github.io/)
[[ArXiv]](https://arxiv.org/abs/2509.08775)
[[Code]](https://github.com/goldenhazard/JM2D)
[[Checkpoints]](https://huggingface.co/RoboWonsuhk/JM2D)

Wonsuhk Jung\*, Utkarsh A. Mishra\*, Nadun Ranawaka Arachchige, Yongxin Chen, Danfei Xu, Shreyas Kousik

Georgia Institute of Technology

This branch is the D4RL PointMaze experiment of JM2D. Other experiments are on the branches [`public_toy`](https://github.com/goldenhazard/JM2D/tree/public_toy) (2-D toy
domain), [`public_d3il`](https://github.com/goldenhazard/JM2D/tree/public_d3il) (D3IL-Avoiding) and [`public_hardware`](https://github.com/goldenhazard/JM2D/tree/public_hardware) (Franka mug pick-and-place).

---

## Overview

A diffusion policy (DP) is trained on D4RL PointMaze demonstrations (`maze2d-large-dense-v1`). At test time every maze
wall is inflated by a padding w that the DP never saw. A safety filter keeps the point robot collision-free: every 16
control steps the DP plan is executed only if a backup plan (accelerate, then brake to a stop) is verified safe with
zonotope reachability; otherwise the backup plan is executed. The planners differ in how the plan is sampled:

- `dp`: the DP alone, no filter.
- `rail`: DP plan, then backup trajectory optimisation (RAIL).
- `gibbs`: alternating updates of the DP plan and the backup parameter k (baseline).
- `jm2d` (ours): the DP plan and the backup parameter k are sampled jointly.

Experiments: **Fig. 4** (safe-success rate, intervention rate and task horizon vs w) and **Table 1** (backup planner
quality at w = 0.20).

---

## Installation

```bash
conda create -n jm2d_maze python=3.8 -y
conda activate jm2d_maze
conda install -c conda-forge cyipopt=1.4.1 -y          # IPOPT, used by RAIL
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

Install PyTorch from conda as above, not the pip wheel: JM2D and Gibbs evaluate the DP candidates in a forked
`multiprocessing` pool, and the pip wheel's GNU OpenMP deadlocks in forked workers (the run hangs at the first
JM2D step). The conda build (LLVM OpenMP, MKL) is the tested setup of the golden tests and the reported numbers.

mujoco-py needs [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0) in `~/.mujoco/mujoco210` and
its `bin/` on `LD_LIBRARY_PATH` (plus `/usr/lib/nvidia` if that directory exists; mujoco-py refuses to import
otherwise). Add the `export` line to `~/.bashrc`; the first `import mujoco_py` compiles it (~20 s):

```bash
mkdir -p ~/.mujoco && wget -qO- https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
python -c "import mujoco_py"   # if the build fails: sudo apt install libosmesa6-dev libgl1-mesa-dev libglew-dev patchelf
```

`requirements.txt` installs the D4RL fork
[goldenhazard/D4RL@a0b30c4](https://github.com/goldenhazard/D4RL/tree/a0b30c4e27c24a4ec9600b211a1b549b13c867b6), which
adds the padded mazes (stock D4RL does not work). A CUDA GPU is required.

---

## Project Structure

```
├── checkpoints/            # DP checkpoints (scripts/download_checkpoints.py)
├── configs/                # maze2d_large.json (planners), train_dp_maze2d_large.json (DP training)
├── jm2d_maze/
│   ├── planners/           # dp.py, rail.py, gibbs.py, jm2d.py; build_planner
│   ├── backup/             # backup planners hq, xpyp, xmym (simple2d.py)
│   ├── tasks/maze.py       # padded maze; SafeMazeEnv turns the walls into zonotope obstacles
│   ├── reachability/       # forward-reachable sets of the backup planner
│   ├── rollout.py          # closed-loop episodes
│   └── compare.py          # Fig. 4 / Table 1 statistics and plot
├── safediffusion/          # safety filter, JM2D and Gibbs sampler (algo/jm2d_policy.py), zonotopes
├── robomimic/              # vendored robomimic: diffusion policy and training
├── paper/                  # paper values (targets.json)
├── results/                # shipped runs, read by compare.py --paper
└── scripts/
    ├── download_checkpoints.py
    ├── train_dp.py         # train the DP
    ├── run.py              # run the planners, one JSON line per episode
    └── compare.py          # figure + table from saved runs
```

---

## Usage

Run all commands from the repository root.

### 1. Download checkpoints

```bash
python scripts/download_checkpoints.py          # epoch 1900 only (default checkpoint of scripts/run.py)
python scripts/download_checkpoints.py --all    # all five checkpoints (~2.6 GB)
```

The files come from the `pointmaze/` folder of [RoboWonsuhk/JM2D](https://huggingface.co/RoboWonsuhk/JM2D) and go to
`checkpoints/` (md5-checked). The five checkpoints (epochs 1250, 1500, 1750, 1900, 2000) are the 5 seeds of Fig. 4.

### 2. Train

```bash
python scripts/download_checkpoints.py --dataset                        # robomimic hdf5 dataset (303 MB, md5-checked)
python scripts/train_dp.py --config configs/train_dp_maze2d_large.json  # 2000 epochs
```

The first command downloads `pointmaze/dataset/maze2d_large_dense_v1.hdf5` of
[RoboWonsuhk/JM2D](https://huggingface.co/RoboWonsuhk/JM2D) to `datasets/d4rl/converted/maze2d_large_dense_v1.hdf5`,
the dataset of the training config. Alternatively, convert it from D4RL (downloads `maze2d-large-dense-v1` from the
D4RL server and writes the same file):

```bash
python robomimic/scripts/conversion/convert_d4rl.py --env maze2d-large-dense-v1 --folder datasets  # D4RL -> robomimic hdf5
```

Checkpoints are saved every 250 epochs and at each new best rollout success rate to
`robomimic/trained_models/maze2d_large_DP/<time>/models/` (robomimic resolves a relative `--output` inside
`robomimic/`); the log is `.../<time>/logs/log.txt`. Use a checkpoint with `scripts/run.py --ckpt` (add `--tag
ep<k>` for the best-rollout files `model_epoch_<k>_..._success_<s>.pth`, whose default run name would be parsed
from `<s>`).

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/train_dp_maze2d_large.json` | robomimic training config |
| `--dataset` | from the config | converted hdf5 dataset |
| `--output` | from the config (`trained_models`) | output directory |
| `--name` | from the config (`maze2d_large_DP`) | experiment name |
| `--debug` | off | short debug run (2 epochs, written to `/tmp/tmp_trained_models`) |

### 3. Run the planners

```bash
# Fig. 4: 4 planners x 6 paddings x 5 checkpoints, env seeds 0-99, HQ backup
python scripts/run.py --planner dp rail gibbs jm2d --pad 0 0.05 0.10 0.15 0.20 0.25 \
    --ckpt checkpoints/maze2d_large_ep{1250,1500,1750,1900,2000}.pth --seeds 0-99 --out results/runs/fig4 --quiet --jobs 2
# Table 1: RAIL and JM2D at w = 0.20 with three backup planners, epoch 2000, env seeds 0-49
python scripts/run.py --planner rail jm2d --pad 0.20 --backup hq xpyp xmym \
    --ckpt checkpoints/maze2d_large_ep2000.pth --seeds 0-49 --out results/runs/table1 --quiet
```

Fig. 4 takes many GPU-hours; a short try: `python scripts/run.py --planner jm2d --pad 0.15 --seeds 0-4`. Keep the two
`--out` directories apart (both commands write `maze_pad020_hq/<planner>/ep2000.jsonl`).

| Argument | Description |
|---|---|
| `--planner` | `dp`, `rail`, `gibbs`, `jm2d` (several allowed) |
| `--pad` | test-time wall padding w >= 0 (default 0; paper: `0 0.05 0.10 0.15 0.20 0.25`) |
| `--maze` | maze layout file instead of maze2d-large (Sec. 4) |
| `--backup` | backup planner: `hq` (default), `xpyp`, `xmym` (Table 1: HQ, x+y+, x-y-) |
| `--ckpt` | DP checkpoint(s) (default `checkpoints/maze2d_large_ep1900.pth`) |
| `--seeds` | env seeds: `0-99` (default), `0,2,7`; start at 0 for the paper's start/goal pairs |
| `--horizon` | max control steps per episode (default 2000 = 20 s) |
| `--config` | another experiment config (default `configs/maze2d_large.json`) |
| `--out` | output directory (default `results/runs`): `<out>/maze_pad<w>_<backup>/<planner>/<ckpt>.jsonl` |
| `--jobs` | (planner, pad, backup, checkpoint) combinations run in parallel, one process each (default 1) |
| `--quiet` | planner console output to `<ckpt>.log` |
| `--video` | also save one video per episode (`<ckpt>_videos/seed<k>.mp4`) |

JM2D hyper-parameters in `configs/maze2d_large.json` (`safety.align`): `N_sample_action` (N = 16, samples of the DP
plan per denoising step), `N_sample_backup` (N_K = 128, samples of the backup parameter k per plan) and `temp_mppi`
(temperature λ = 0.05). After denoising, k is refined by `N_steps_mppi` = 3 MPPI steps of `N_sample_trajopt` = 512
samples (`opt_K_again`). The filter replans every `safety.filter.n_head` = 16 steps; the DP uses DDIM with 10 steps
(`algo.ddim`) and `safety.policy.ddim_eta` = 1.

### 4. Add your own constraints

The constraints are the maze walls. Each wall cell of the maze is a MuJoCo box named `wall_<row>_<col>`, and
`SafeMazeEnv` (`jm2d_maze/tasks/maze.py`) turns every geom whose name starts with `wall` into a zonotope obstacle
(`obs["zonotope"]["obstacle"]`). At every replanning step the obstacles are used by the collision check of the DP plan,
by the backup planner (its reachable set must avoid them) and by the JM2D / Gibbs sample weights; an episode is unsafe
when the robot touches a wall. The DP never sees the obstacles, so they can be changed without retraining.

**Wall padding.** `--pad` takes any w >= 0, e.g.

```bash
python scripts/run.py --planner jm2d --pad 0.12 --seeds 0-4 --out results/runs/pad012
```

**Maze layout.** Write the maze as 9 lines of 12 tiles (`#` wall, `O` free, `G` goal). This `my_maze.txt` is
maze2d-large with one extra wall cell (row 2, column 10, counted from 0):

```text
############
#OOOO#OOOOO#
#O##O#O#O###
#OOOOOO#OOO#
#O####O###O#
#OO#O#OOOOO#
##O#O#O#O###
#OO#OOO#OGO#
############
```

```bash
python scripts/run.py --planner jm2d --maze my_maze.txt --pad 0.12 --seeds 0-4 --out results/runs/my_maze
```

Results go to `<out>/my_maze_pad012_hq/`. In Python: `build_planner(..., pad=0.12,
maze_layout=jm2d_maze.tasks.read_maze_layout("my_maze.txt"))`.

**Backup planner.** A plan is only accepted if a backup plan with parameter k in the box [k_lb, k_ub] avoids all
obstacles. `--backup` selects the box: `hq` [-1, 1]^2, `xpyp` [0, 1]^2, `xmym` [-1, 0]^2. To add another box:

```python
import jm2d_maze                                    # import first (sets CUBLAS_WORKSPACE_CONFIG)
from jm2d_maze.backup import BACKUPS, Simple2DPlannerUnderAct
from jm2d_maze.planners import build_planner
from jm2d_maze.rollout import rollout_seeds
from jm2d_maze.seeding import set_random_seed

BACKUPS["xp"] = (Simple2DPlannerUnderAct, {"k_lb": [0, -1], "k_ub": [1, 1]})    # k_x >= 0 only
set_random_seed(42)
planner, env, eval_env = build_planner("jm2d", "checkpoints/maze2d_large_ep1900.pth", pad=0.12, backup="xp")
stats = rollout_seeds(planner, env, eval_env, horizon=2000, seeds=[0])
print(stats["success_and_safe_rate"], stats["intervention_rate"])
```

### 5. See the results

```bash
python scripts/compare.py --results results/runs/fig4 --figure fig4 --check-paper --out results/figures/fig4.png
python scripts/compare.py --results results/runs/table1 --figure table1 --check-paper
```

This prints the safe-success rate, intervention rate and task horizon per planner and w (mean over checkpoints),
compares them with the paper (`paper/targets.json`) and draws Fig. 4. Per-episode results are in the `.jsonl` files
(schema: `jm2d_maze/results.py`).

`python scripts/compare.py --paper --check-paper` redraws Fig. 4 and Table 1 from the shipped runs in `results/` (no
GPU, no checkpoint; figure in `results/figures/`).

**Video of one episode.** The start/goal of env seed k depend on seeds 0..k-1, so run seeds `0-k` with `--video`:

```bash
python scripts/run.py --planner jm2d --pad 0.15 --ckpt checkpoints/maze2d_large_ep1900.pth --seeds 0-3 --video \
    --out results/runs/video
```

Episode 3 is then `results/runs/video/maze_pad015_hq/jm2d/ep1900_videos/seed3.mp4` (one mp4 per seed; 512x512, 20 fps,
real time; green robot, red goal, padded walls; blue-tinted frames: backup plan executed, red-tinted: collision).
No environment variable is needed: mujoco-py renders offscreen with EGL on the NVIDIA GPU (no display).

---

## Citation

```bibtex
@inproceedings{jung2025joint,
  title={Joint Model-based Model-free Diffusion for Planning with Constraints},
  author={Wonsuhk Jung and Utkarsh Aashu Mishra and Nadun Ranawaka Arachchige and Yongxin Chen and Danfei Xu and Shreyas Kousik},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025},
  url={https://openreview.net/forum?id=E9t1ekt6W9}
}
```

## License

MIT (see `LICENSE`). Third-party code and its licenses are listed in `NOTICE`.

**Acknowledgments.** This code builds on robomimic, Diffusion Policy, the ARMTD zonotope library, zonopy, D4RL,
diffusers and CORA. The training dataset is D4RL `maze2d-large-dense-v1` (Fu et al., 2020; data license CC BY 4.0),
converted to robomimic hdf5.
