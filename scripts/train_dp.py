#!/usr/bin/env python
"""(Optional) retrain the PointMaze diffusion policy with robomimic.

    # 1. download the converted dataset (writes datasets/d4rl/converted/maze2d_large_dense_v1.hdf5) ...
    python scripts/download_checkpoints.py --dataset
    #    ... or download + convert it from D4RL (writes the same file)
    python robomimic/scripts/conversion/convert_d4rl.py --env maze2d-large-dense-v1 --folder datasets
    # 2. train (2000 epochs) -> robomimic/trained_models/maze2d_large_DP/<time>/models/model_epoch_<k>*.pth
    python scripts/train_dp.py --config configs/train_dp_maze2d_large.json

Checkpoints are saved every 250 epochs and at every new best rollout success rate (rollouts every 50 epochs).
robomimic resolves a relative output directory (``train.output_dir`` = ``trained_models``, ``--output``) against
``robomimic/``; pass an absolute ``--output`` to write elsewhere. Log: ``<output>/<name>/<time>/logs/log.txt``.
The script exits with status 1 if training raised (robomimic's ``main`` only prints the error).

``configs/train_dp_maze2d_large.json`` is the config of the 2025-04-16 training run that produced the epoch
1250/1500/1750/2000 checkpoints of Fig. 4 (data/output paths made relative). The epoch-1900 checkpoint
(``maze2d_large_ep1900.pth``) comes from an earlier training on a variant of the same dataset; its config is
stored inside the checkpoint (``torch.load(ckpt)["config"]``). Training is not seeded bit-for-bit (cuDNN), so a
retrained policy differs from the released checkpoints.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import robomimic.scripts.train as rmtrain  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "train_dp_maze2d_large.json"))
    parser.add_argument("--dataset", default=None, help="override train.data (converted robomimic hdf5)")
    parser.add_argument("--output", default=None, help="override train.output_dir")
    parser.add_argument("--name", default=None, help="override experiment.name")
    parser.add_argument("--debug", action="store_true", help="short debug run")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)  # the config's relative dataset path is relative to the repository root

    failed = []
    train = rmtrain.train

    def train_recording_failure(*a, **kw):  # robomimic's main() catches training errors and only prints them
        try:
            return train(*a, **kw)
        except Exception:
            failed.append(True)
            raise

    rmtrain.train = train_recording_failure
    rmtrain.main(argparse.Namespace(config=args.config, algo=None, name=args.name, dataset=args.dataset,
                                    output=args.output, debug=args.debug, auto_remove_exp=False))
    if failed:
        sys.exit("training failed (traceback above and in the run's logs/log.txt)")


if __name__ == "__main__":
    main()
