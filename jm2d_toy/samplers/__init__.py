"""Sampler registry and the public sampling protocol.

    sampler = samplers.make("jm2d", ckpt="checkpoints/donut_constrained.pth", num_mc_samples=512, temperature=0.1)
    res = sampler.sample(problem, n=64, seed=0)              # SampleResult(x, valid, ood, ...)

    sampler = samplers.make_joint("gibbs", ckpt="checkpoints/donut_joint.pth", num_param_samples=16384)
    res = sampler.sample(problem, n=16, seed=0)              # SampleResult(x, k, valid (= compatible), ...)

``Sampler.sample`` follows the protocol of the golden outputs (tests/golden): seed_everything(seed) ->
build the sampler (instantiates the denoiser and loads the checkpoint; network initialisation consumes CPU
RNG before the weights are overwritten, exactly as in the legacy code) -> run the reverse diffusion.
A fresh instance per call means results depend only on (config, checkpoint, problem, n, seed).
"""
import os
import time

import numpy as np

from jm2d_toy.config import load_config, sampler_kwargs
from jm2d_toy.results import SampleResult, md5
from jm2d_toy.samplers.base import BaseJointSampler, BaseSampler
from jm2d_toy.samplers.gibbs import GibbsSampler
from jm2d_toy.samplers.jm2d import JM2DJointSampler, JM2DSampler
from jm2d_toy.samplers.projection import ProjectionSampler
from jm2d_toy.samplers.sequential import SequentialSampler
from jm2d_toy.seeding import seed_everything
from jm2d_toy.tasks import JOINT_TASKS, canonical

# constrained generation (donut_constrained)
REGISTRY = {"projection": ProjectionSampler, "jm2d": JM2DSampler}
# joint x / k sampling (donut_joint)
JOINT_REGISTRY = {"sequential": SequentialSampler, "gibbs": GibbsSampler, "jm2d": JM2DJointSampler}


class Sampler:
    """A configured sampler: ``sample(problem, n, seed)`` -> ``SampleResult``."""

    def __init__(self, name, cls, ckpt, problem=None, use_constraints=True, device="cuda", verbose=False, **cfg):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint {ckpt} does not exist")
        self.name = name
        self.cls = cls
        self.ckpt = ckpt
        self.problem = problem
        self.use_constraints = use_constraints
        self.device = device
        self.verbose = verbose
        self.cfg = cfg
        self.joint = issubclass(cls, BaseJointSampler)

    def __repr__(self):
        return f"Sampler({self.name!r}, {self.cls.__name__}, ckpt={self.ckpt!r}, cfg={self.cfg})"

    def build(self, problem):
        """Instantiate the underlying sampler (loads the checkpoint)."""
        return self.cls(model_path=self.ckpt, problem=problem, device=self.device, verbose=self.verbose, **self.cfg)

    def sample(self, problem=None, n=16, seed=0, return_history=False):
        problem = self.problem if problem is None else problem
        if problem is None:
            raise ValueError("no problem given (pass it to make(...) or sample(...))")

        seed_everything(seed)
        impl = self.build(problem)
        t0 = time.time()
        out = impl.get_action(batch_size=n, use_constraints=self.use_constraints, return_histories=return_history)
        elapsed = time.time() - t0

        history = None
        if self.joint:
            x, k = out[0], out[1]
            if return_history:
                # legacy joint histories are lists over T of (B, ...): stack and transpose to (B, T, ...)
                history = {}
                for key, val in out[2].items():
                    arr = np.array(val)
                    history[key] = np.swapaxes(arr, 0, 1) if arr.ndim >= 2 else arr  # timesteps stay (T,)
        else:
            k = None
            if return_history:
                x, history = out
            else:
                x = out
        x = np.asarray(x).reshape(n, -1)
        k = None if k is None else np.asarray(k).reshape(n, -1)

        xt = problem.to_tensor(x)
        if self.joint:
            kt = problem.to_tensor(k)
            valid = problem.is_valid(xt, kt)
            objective = problem.objective_function(xt, kt)
        else:
            valid = problem.is_valid(xt)
            objective = problem.objective_function(xt)
        ood = problem.is_ood(xt)

        return SampleResult(
            task=getattr(problem, "task", type(problem).__name__), sampler=self.name, n=n, seed=seed,
            x=x, k=k, valid=valid.cpu().numpy(), ood=ood.cpu().numpy(), objective=objective.cpu().numpy(),
            history=history, config={"class": self.cls.__name__, **self.cfg}, use_constraints=self.use_constraints,
            global_optimum=float(problem.global_optimal_value()), ckpt=os.path.basename(self.ckpt),
            ckpt_md5=md5(self.ckpt), time_s=elapsed,
        )


def _make(registry, name, ckpt, problem=None, **cfg):
    if name not in registry:
        raise KeyError(f"unknown sampler {name!r}; choose from {sorted(registry)}")
    return Sampler(name, registry[name], ckpt, problem=problem, **cfg)


def make(name, ckpt, problem=None, **cfg):
    """Constrained-generation sampler (donut_constrained): ``projection`` or ``jm2d``."""
    return _make(REGISTRY, name, ckpt, problem=problem, **cfg)


def make_joint(name, ckpt, problem=None, **cfg):
    """Joint x/k sampler (donut_joint): ``sequential``, ``gibbs`` or ``jm2d``."""
    return _make(JOINT_REGISTRY, name, ckpt, problem=problem, **cfg)


def from_config(cfg, name, problem=None, **overrides):
    """Build sampler ``name`` from a ``configs/*.json`` dict (or a task name / path, see ``load_config``)."""
    if not isinstance(cfg, dict):
        cfg = load_config(cfg)
    kwargs = sampler_kwargs(cfg, name)
    kwargs.setdefault("use_constraints", cfg.get("use_constraints", True))
    kwargs.update(overrides)
    factory = make_joint if canonical(cfg["task"]) in JOINT_TASKS else make
    return factory(name, cfg["checkpoint"], problem=problem, **kwargs)


__all__ = ["REGISTRY", "JOINT_REGISTRY", "Sampler", "make", "make_joint", "from_config", "BaseSampler",
           "BaseJointSampler", "JM2DSampler", "JM2DJointSampler", "GibbsSampler", "SequentialSampler",
           "ProjectionSampler"]
