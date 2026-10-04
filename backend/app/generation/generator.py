"""Runs one generation request end to end on top of the model manager.

Order of work for a request: translate Japanese prompts -> make sure the model is loaded ->
LoRA / InstantID adapter -> pick the pipeline (text-to-image, image-to-image or ControlNet
for face/pose references) -> denoise -> encode the images.
"""

from __future__ import annotations

import contextlib
import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..catalog import Catalog, ModelEntry
from ..config import SEED_MAX, Settings
from ..errors import (
    ContentFilteredError,
    GenerationCancelledError,
    GenerationFailedError,
    ModelUnavailableError,
    is_out_of_memory,
)
from .imaging import OUTPUT_FORMATS, encode_image, fit_to
from .models import LoadedModel, ModelManager, ModelStatus
from .params import GenerationParams
from .reference import ReferenceControl
from .runtime import Runtime
from .schedulers import build_scheduler
from .styles import apply_style
from .translate import PromptTranslation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime_type: str
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    images: list[GeneratedImage]
    filtered_count: int = 0
    # English versions of Japanese prompts (None when nothing was translated).
    translated_prompt: str | None = None
    translated_negative_prompt: str | None = None


class ImageGenerator(Protocol):
    @property
    def status(self) -> ModelStatus: ...

    @property
    def cached_models(self) -> frozenset[str]:
        """Catalog ids whose weights are already on disk (usable without a download)."""
        ...

    @property
    def identity_cached(self) -> bool:
        """Whether the face/pose reference assets are already on disk."""
        ...

    @property
    def translator_cached(self) -> bool:
        """Whether the Japanese prompt translator is already on disk."""
        ...

    def load(self) -> None: ...

    def generate(self, params: GenerationParams, cancel: threading.Event | None = None) -> GenerationResult: ...


def batch_seeds(seed: int | None, count: int) -> list[int]:
    """One seed per image: the given (or a random) seed, then +1, +2, ... wrapping at 2^32."""
    base = seed if seed is not None else secrets.randbelow(SEED_MAX + 1)
    return [(base + i) % (SEED_MAX + 1) for i in range(count)]


class DiffusersGenerator:
    def __init__(self, settings: Settings, catalog: Catalog) -> None:
        self._settings = settings
        self._catalog = catalog
        self._runtime = Runtime(settings)
        self._models = ModelManager(catalog, self._runtime)
        self._references = ReferenceControl(catalog.identity, self._runtime)
        self._translation = PromptTranslation(catalog.translator, self._runtime)
        # Diffusers pipelines are not thread-safe, and model switching must not overlap a run.
        self._run_lock = threading.Lock()

    @property
    def status(self) -> ModelStatus:
        return self._models.status

    @property
    def cached_models(self) -> frozenset[str]:
        return self._models.cached_models

    @property
    def identity_cached(self) -> bool:
        return self._references.cached

    @property
    def translator_cached(self) -> bool:
        return self._translation.cached

    def load(self) -> None:
        """Load the default model (called once at startup, in a background thread)."""
        self._models.scan_cache()
        self._references.scan_cache()
        self._translation.scan_cache()
        # A failure is already logged and reflected in status.
        with self._run_lock, contextlib.suppress(ModelUnavailableError):
            self._ensure_model(self._catalog.default)

    def _ensure_model(self, entry: ModelEntry) -> LoadedModel:
        loaded = self._models.ensure(entry)
        self._references.release_unless_supported(loaded.entry.family)
        return loaded

    def generate(self, params: GenerationParams, cancel: threading.Event | None = None) -> GenerationResult:
        """Run one request. Setting `cancel` stops it at the next denoising step (GenerationCancelledError)."""
        with self._run_lock:
            if cancel is not None and cancel.is_set():  # cancelled while waiting for the previous job
                raise GenerationCancelledError()
            logger.info(
                "Generating %d image(s): model=%s %dx%d steps=%d faces=%d pose=%s",
                params.num_images,
                params.model.id,
                params.width,
                params.height,
                params.num_inference_steps,
                len(params.face_images),
                params.pose_image is not None,
            )
            prompt, negative_prompt, translated = self._translation.prompts(params.prompt, params.negative_prompt)
            loaded = self._ensure_model(params.model)
            self._models.apply_loras(loaded, params.loras)
            self._references.sync_ip_adapter(loaded, needed=bool(params.face_images))
            pipe, kwargs = self._prepare_call(loaded, params, *apply_style(params.style, prompt, negative_prompt))
            seeds = batch_seeds(params.seed, params.num_images)
            # CPU generators give the same image for the same seed regardless of device.
            kwargs["generator"] = [self._runtime.torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            kwargs["callback_on_step_end"] = self._step_callback(loaded, params, cancel)
            output = self._run(pipe, kwargs, loaded, params)

        result = self._encode(output, params, seeds)
        return GenerationResult(
            images=result.images,
            filtered_count=result.filtered_count,
            translated_prompt=prompt if translated else None,
            translated_negative_prompt=negative_prompt if translated and params.negative_prompt else None,
        )

    def _prepare_call(
        self, loaded: LoadedModel, params: GenerationParams, prompt: str, negative_prompt: str
    ) -> tuple[Any, dict[str, Any]]:
        """The pipeline to use and its call arguments (except generators and the step callback)."""
        extra: dict[str, Any] = {}
        if params.uses_reference:
            pipe, extra = self._references.pipeline(loaded, params)
        elif params.init_image is not None:
            pipe = loaded.img2img
        else:
            pipe = loaded.text2img
        pipe.scheduler = build_scheduler(params.scheduler, loaded.original_scheduler)
        if self._settings.enable_cpu_offload and self._runtime.is_cuda:
            pipe.enable_model_cpu_offload()  # (re)install hooks on the pipeline actually used

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "num_images_per_prompt": params.num_images,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if params.init_image is not None:
            kwargs["image"] = fit_to(params.init_image, params.width, params.height)
            kwargs["strength"] = params.strength
        else:
            kwargs["width"] = params.width
            kwargs["height"] = params.height
        kwargs.update(extra)
        return pipe, kwargs

    def _step_callback(
        self, loaded: LoadedModel, params: GenerationParams, cancel: threading.Event | None
    ) -> Callable[..., dict[str, Any]]:
        def on_step_end(running: Any, step: int, _timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
            if step == 0:  # the prompt is encoded by now
                self._models.move_text_encoders(loaded, on_gpu=False)
            if cancel is not None and cancel.is_set():
                raise GenerationCancelledError()
            if params.uses_reference and step >= getattr(running, "num_timesteps", 0) - 1:
                # The ControlNets (~5GB) are done; free them before the VAE decode needs its peak memory.
                self._references.park()
            return tensors

        return on_step_end

    def _run(self, pipe: Any, kwargs: dict[str, Any], loaded: LoadedModel, params: GenerationParams) -> Any:
        torch = self._runtime.torch
        with self._runtime.track_peak_memory():
            try:
                self._models.move_text_encoders(loaded, on_gpu=True)
                with torch.inference_mode():
                    return pipe(**kwargs)
            except Exception as exc:
                if is_out_of_memory(exc):
                    self._runtime.release_memory()
                raise
            finally:
                self._models.move_text_encoders(loaded, on_gpu=False)
                if params.uses_reference:
                    self._references.park()

    @staticmethod
    def _encode(output: Any, params: GenerationParams, seeds: list[int]) -> GenerationResult:
        images = getattr(output, "images", None)
        if not images:
            raise GenerationFailedError()
        nsfw = getattr(output, "nsfw_content_detected", None) or [False] * len(images)

        spec = OUTPUT_FORMATS[params.output_format]
        kept = [
            GeneratedImage(encode_image(img, params.output_format, params.quality), spec.mime_type, seed)
            for img, seed, flagged in zip(images, seeds, nsfw, strict=False)
            if not flagged
        ]
        if not kept:
            raise ContentFilteredError()
        return GenerationResult(images=kept, filtered_count=len(images) - len(kept))
