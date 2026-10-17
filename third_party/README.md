# third_party: D3IL (patched)

The Avoiding environment and demonstrations come from D3IL (ALRhub/d3il, MIT License). This directory ships only
`d3il.patch`, the local changes the experiments were run with:

- every `environments.d3il...` import is rewritten to `d3il.environments.d3il...` (the clone is imported as the
  namespace package `d3il`, with `third_party/` on `sys.path`; `jm2d_d3il` adds it automatically);
- `ObstacleAvoidanceEnv.start(override_init_pos=None)` (the evaluation starts at (0.525, -0.28, 0.12)).

Base commit: `1d9c718` ("Create LICENSE"; code-identical to `0046bec`, which the original requirements pinned).
77 files changed, 281 insertions, 278 deletions.

```bash
git clone https://github.com/ALRhub/d3il.git third_party/d3il
git -C third_party/d3il checkout 1d9c718
git -C third_party/d3il apply ../d3il.patch
# Avoiding demonstrations (96 .pkl, 2.6 MB, unmodified copy on https://huggingface.co/RoboWonsuhk/JM2D, md5-checked)
# -> third_party/d3il/environments/dataset/data/avoiding/data/
python scripts/download_checkpoints.py --data
```

Alternatively, take the Avoiding folder from the D3IL data archive linked in `third_party/d3il/README.md`.

The environment writes temporary robot XML files to `third_party/d3il/environments/d3il/models/mj/robot/`.
