"""SafeDiffuser: invariance QP after every denoising step (legacy variants 'invariance-{ReS,TVS}-dpcc-c')."""
from .base import Planner

INVARIANCE_TYPES = {"res": "ReS", "tvs": "TVS"}


class SafeDiffuserPlanner(Planner):
    name = "safediffuser"

    def __init__(self, variant="res", **kwargs):
        if variant not in INVARIANCE_TYPES:
            raise ValueError(f"safediffuser variant must be one of {sorted(INVARIANCE_TYPES)}, got {variant!r}")
        legacy = kwargs.pop("legacy_variant", None)
        if isinstance(legacy, dict):
            legacy = legacy[variant]
        super().__init__(legacy_variant=legacy, **kwargs)
        self.variant = variant

    @property
    def label(self):
        return f"{self.name}-{self.variant}"

    def settings(self):
        return dict(super().settings(), variant=self.variant, invariance_type=INVARIANCE_TYPES[self.variant])

    def policy_kwargs(self):
        return dict(invariance_type=INVARIANCE_TYPES[self.variant])
