"""Synthetic donut datasets (legacy ``dev/dataset.py``).

Points are uniform in angle and uniform in radius on the annulus R_INNER <= |p| <= R_OUTER.
``DonutConstrainedDataset``: x in R^2 (``donut_constrained`` model). ``DonutJointDataset``: x = (p1, p2) in
R^4 (``donut_joint`` model).
``rng=None`` keeps the legacy behaviour (global ``np.random``); pass a ``np.random.RandomState`` for
reproducible data (used for plot backgrounds). Removed: ``IntervalDataset``.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

R_INNER = 0.6
R_OUTER = 1.0


class DonutConstrainedDataset(Dataset):
    # generate a dataset by sampling points from a donut of inner radius r0 and outer radius r1
    def __init__(self, num_samples, rng=None):
        rng = np.random if rng is None else rng
        self.num_samples = num_samples
        self.r0 = R_INNER
        self.r1 = R_OUTER

        self.all_theta = rng.uniform(0, 2*np.pi, num_samples)
        self.all_r = rng.uniform(self.r0, self.r1, num_samples)

        self.all_samples = np.array([r*np.array([np.cos(theta), np.sin(theta)]) for r, theta in zip(self.all_r, self.all_theta)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.all_samples[idx]

    @property
    def shape(self):
        return self.all_samples[0].shape


class DonutJointDataset(Dataset):
    # generate a dataset by sampling pairs of points from a donut of inner radius r0 and outer radius r1
    def __init__(self, num_samples, rng=None):
        rng = np.random if rng is None else rng
        self.num_samples = num_samples
        self.r0 = R_INNER
        self.r1 = R_OUTER

        self.all_theta = rng.uniform(0, 2*np.pi, num_samples * 2)
        self.all_r = rng.uniform(self.r0, self.r1, num_samples * 2)

        self.all_samples = np.array([r*np.array([np.cos(theta), np.sin(theta)]) for r, theta in zip(self.all_r, self.all_theta)])

        half_indices = rng.choice(num_samples * 2, num_samples, replace=False)
        other_half_indices = np.setdiff1d(np.arange(num_samples * 2), half_indices)

        self.all_samples = np.concatenate([self.all_samples[half_indices], self.all_samples[other_half_indices]], axis=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.all_samples[idx]

    @property
    def shape(self):
        return self.all_samples[0].shape


def get_donut_constrained_dataloader(num_samples, batch_size, num_workers, rng=None):
    dataset = DonutConstrainedDataset(num_samples, rng=rng)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, dataset


def get_donut_joint_dataloader(num_samples, batch_size, num_workers, rng=None):
    dataset = DonutJointDataset(num_samples, rng=rng)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, dataset


Donut2DDataset, DonutDataset = DonutConstrainedDataset, DonutJointDataset
get_donut2d_dataloader, get_donut_dataloader = get_donut_constrained_dataloader, get_donut_joint_dataloader
