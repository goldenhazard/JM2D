"""MPD: cost-gradient guidance of the noisy mean (legacy variant 'mpd-gradient')."""
from .base import Planner


class MPDPlanner(Planner):
    name = "mpd"

    def __init__(self, gradient_weights, n_guide_steps, clip_grad, clip_grad_rule, start_guiding_steps, repeat_last,
                 **kwargs):
        super().__init__(**kwargs)
        self.gradient_weights = dict(gradient_weights)
        self.mpd_kwargs = {
            'n_guide_steps': n_guide_steps,
            'clip_grad': clip_grad,
            'start_guiding_steps': start_guiding_steps,
            'clip_grad_rule': clip_grad_rule,
            'repeat_last': repeat_last,
        }

    def settings(self):
        return dict(super().settings(), gradient_weights=self.gradient_weights, **self.mpd_kwargs)

    def projector_kwargs(self):
        return dict(gradient=True, gradient_weights=self.gradient_weights, projection_variant=self.name,
                    **self.mpd_kwargs)

    def policy_kwargs(self):
        return dict(self.mpd_kwargs)
