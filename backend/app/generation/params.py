"""What one generation request asks for, after validation (see ``api/validation.py``)."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..catalog import LoraEntry, ModelEntry
from .imaging import OutputFormat

# Ranges of the per-request controls (sizes, steps, etc. come from Settings).
MIN_STRENGTH, MAX_STRENGTH, DEFAULT_STRENGTH = 0.1, 1.0, 0.6
MIN_LORA_SCALE, MAX_LORA_SCALE = 0.0, 2.0
MIN_QUALITY, MAX_QUALITY, DEFAULT_QUALITY = 1, 100, 90
MIN_CONTROL_STRENGTH, MAX_CONTROL_STRENGTH = 0.0, 1.5
DEFAULT_IDENTITY_STRENGTH, DEFAULT_POSE_STRENGTH = 0.8, 0.9


@dataclass(frozen=True)
class GenerationParams:
    # As typed (possibly Japanese); the generator translates them and adds the style's wording.
    prompt: str
    negative_prompt: str
    model: ModelEntry
    scheduler: str
    style: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None
    num_images: int
    output_format: OutputFormat
    quality: int
    init_image: Image.Image | None
    strength: float
    loras: tuple[tuple[LoraEntry, float], ...]
    face_images: tuple[Image.Image, ...] = ()
    pose_image: Image.Image | None = None
    identity_strength: float = DEFAULT_IDENTITY_STRENGTH
    pose_strength: float = DEFAULT_POSE_STRENGTH

    @property
    def uses_reference(self) -> bool:
        return bool(self.face_images) or self.pose_image is not None
