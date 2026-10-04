"""Samplers (Diffusers schedulers) users can choose from.

"default" keeps the scheduler the model ships with. The others are created from the
model's own scheduler config, so model-specific settings (betas, timestep spacing,
prediction type) carry over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SchedulerSpec:
    label: str
    class_name: str | None  # None = the model's original scheduler class
    options: dict[str, Any] = field(default_factory=dict)


SCHEDULERS: dict[str, SchedulerSpec] = {
    "default": SchedulerSpec("モデル既定", None),
    "euler": SchedulerSpec("Euler", "EulerDiscreteScheduler"),
    "euler_a": SchedulerSpec("Euler a", "EulerAncestralDiscreteScheduler"),
    "dpmpp_2m": SchedulerSpec("DPM++ 2M", "DPMSolverMultistepScheduler"),
    "dpmpp_2m_karras": SchedulerSpec("DPM++ 2M Karras", "DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "unipc": SchedulerSpec("UniPC", "UniPCMultistepScheduler"),
    "ddim": SchedulerSpec("DDIM", "DDIMScheduler"),
}


def build_scheduler(scheduler_id: str, original: Any) -> Any:
    """Create a fresh scheduler instance (schedulers hold per-run state, so never reuse one)."""
    spec = SCHEDULERS[scheduler_id]
    if spec.class_name is None:
        cls = type(original)
    else:
        import diffusers

        cls = getattr(diffusers, spec.class_name)
    return cls.from_config(original.config, **spec.options)
