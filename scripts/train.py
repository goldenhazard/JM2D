#!/usr/bin/env python
"""Train the donut denoiser (legacy ``dev/scripts/train.py``).

    python scripts/train.py --dataset donut_constrained --out trained_models/donut_constrained   # x in R^2
    python scripts/train.py --dataset donut_joint       --out trained_models/donut_joint         # x in R^4

Checkpoints are written every 1e4 steps to ``<out>/model/model_step<k>e4.pth``; the shipped
``checkpoints/{donut_joint,donut_constrained}.pth`` are ``model_step9e4.pth`` of such runs (hidden_dim 128;
the legacy default of 256 did not match them). The legacy runs were unseeded, so retraining gives a different network.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from jm2d_toy.data import get_donut_constrained_dataloader, get_donut_joint_dataloader  # noqa: E402
from jm2d_toy.models.denoiser import DenoiserMLP  # noqa: E402
from jm2d_toy.seeding import seed_everything  # noqa: E402

_DATASET_ALIASES = {"donut": "donut_joint", "donut2d": "donut_constrained"}


def plot_samples(samples, save_path):
    plt.figure(figsize=(9, 3))
    plt.scatter(samples[:, 0], samples[:, 1], c='k', s=1)
    if samples.shape[1] == 4:
        plt.scatter(samples[:, 2], samples[:, 3], c='r', s=1)
    plt.grid(False)
    plt.savefig(save_path)
    plt.close()


def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    if args.dataset == "donut_joint":
        loader, dataset = get_donut_joint_dataloader(args.num_samples, args.batch_size, args.num_workers)
    else:
        loader, dataset = get_donut_constrained_dataloader(args.num_samples, args.batch_size, args.num_workers)

    model = DenoiserMLP(ob_dim=dataset.shape[0], hidden_dim=args.hidden_dim, device='cuda', log_dir=args.out)
    model.fit(loader, num_train_steps=args.num_train_steps, plot_fn=plot_samples)

    # sample from the trained model
    samples = []
    for _ in tqdm(range(args.eval_batch_size)):
        samples.extend(model.get_action(batch_size=args.eval_batch_size).cpu().numpy())
    plot_samples(np.array(samples), os.path.join(args.out, "final_samples.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["donut_joint", "donut_constrained"], default="donut_constrained",
                        type=lambda name: _DATASET_ALIASES.get(name, name),
                        help="donut_joint: x = (p1, p2) in R^4; donut_constrained: x in R^2")
    parser.add_argument("--out", required=True, help="output directory (model/ and plots/ are created inside)")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    parser.add_argument("--eval_batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="optional; the paper checkpoints were unseeded")
    main(parser.parse_args())
