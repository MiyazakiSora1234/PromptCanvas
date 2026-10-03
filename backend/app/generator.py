"""Diffusers pipeline wrapper: loads the model once and generates PNG images.

``torch`` and ``diffusers`` are imported lazily inside :meth:`DiffusersGenerator.load`
so that the API layer and tests can run without them installed.
"""

from __future__ import annotations

import io
import logging
import secrets
import threading
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from .config import SEED_MAX, Settings
from .errors import (
    ConfigurationError,
    ContentFilteredError,
    GenerationFailedError,
    ModelLoadingError,
    ModelUnavailableError,
    describe_load_error,
    is_out_of_memory,
)
from .schemas import GenerationParams

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelStatus:
    state: ModelState
    device: str | None = None
    dtype: str | None = None
    # Safe, fixed hint text (never raw exception output).
    message: str | None = None

    def ensure_ready(self) -> None:
        """Raise the user-facing error for any state other than READY."""
        if self.state in (ModelState.NOT_LOADED, ModelState.LOADING):
            raise ModelLoadingError()
        if self.state is ModelState.FAILED:
            detail = f"（{self.message}）" if self.message else ""
            raise ModelUnavailableError(ModelUnavailableError.default_message + detail)


@dataclass(frozen=True)
class GenerationResult:
    png: bytes
    seed: int


class ImageGenerator(Protocol):
    @property
    def status(self) -> ModelStatus: ...

    def load(self) -> None: ...

    def generate(self, params: GenerationParams) -> GenerationResult: ...


def resolve_device(requested: str, *, cuda_available: bool, mps_available: bool) -> str:
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise ConfigurationError(
            "PROMPTCANVAS_DEVICE=cuda が指定されていますが CUDA が利用できません。"
            "NVIDIAドライバとCUDA版PyTorchのインストールを確認するか、auto / cpu を指定してください。"
        )
    if requested == "mps" and not mps_available:
        raise ConfigurationError("PROMPTCANVAS_DEVICE=mps が指定されていますが MPS が利用できません。")
    return requested


def resolve_dtype(requested: str, device: str) -> str:
    if requested == "auto":
        return "float16" if device in ("cuda", "mps") else "float32"
    if device == "cpu" and requested == "float16":
        raise ConfigurationError(
            "CPU では float16 を使用できません。PROMPTCANVAS_TORCH_DTYPE を float32 か auto にしてください。"
        )
    return requested


class DiffusersGenerator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipe: Any = None
        self._torch: Any = None
        self._status = ModelStatus(ModelState.NOT_LOADED)
        self._status_lock = threading.Lock()
        # Diffusers pipelines are not thread-safe; serialize calls even if the
        # limiter is ever configured to allow more than one.
        self._run_lock = threading.Lock()

    @property
    def status(self) -> ModelStatus:
        with self._status_lock:
            return self._status

    def _set_status(self, status: ModelStatus) -> None:
        with self._status_lock:
            self._status = status

    def load(self) -> None:
        s = self._settings
        self._set_status(ModelStatus(ModelState.LOADING))
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            mps = getattr(torch.backends, "mps", None)
            device = resolve_device(
                s.device,
                cuda_available=torch.cuda.is_available(),
                mps_available=bool(mps and mps.is_available()),
            )
            dtype_name = resolve_dtype(s.torch_dtype, device)
            self._set_status(ModelStatus(ModelState.LOADING, device=device, dtype=dtype_name))
            if device == "cpu":
                logger.warning(
                    "GPU が見つからないため CPU で実行します。1枚の生成に数分以上かかる場合があります。"
                )
            if device == "cuda":
                logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
            logger.info("Loading model %s on %s (%s)...", s.model_id, device, dtype_name)

            kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, dtype_name)}
            if s.model_revision:
                kwargs["revision"] = s.model_revision
            if s.model_variant:
                kwargs["variant"] = s.model_variant
            if s.hf_token is not None:
                kwargs["token"] = s.hf_token.get_secret_value()

            pipeline_cls: Any = AutoPipelineForText2Image  # diffusers is only partially typed
            pipe = pipeline_cls.from_pretrained(s.model_id, **kwargs)
            if s.enable_cpu_offload and device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                if s.enable_cpu_offload:
                    logger.warning("PROMPTCANVAS_ENABLE_CPU_OFFLOAD is only effective on CUDA; ignored.")
                pipe = pipe.to(device)
            if s.enable_attention_slicing:
                pipe.enable_attention_slicing()
            pipe.set_progress_bar_config(disable=True)
        except Exception as exc:
            hint = describe_load_error(exc)
            logger.exception("Failed to load model %s. %s", s.model_id, hint)
            self._set_status(replace(self.status, state=ModelState.FAILED, message=hint))
            return

        self._torch = torch
        self._pipe = pipe
        self._set_status(replace(self.status, state=ModelState.READY))
        logger.info("Model ready.")

    def generate(self, params: GenerationParams) -> GenerationResult:
        pipe, torch = self._pipe, self._torch
        if pipe is None or torch is None:
            raise ModelLoadingError()

        seed = params.seed if params.seed is not None else secrets.randbelow(SEED_MAX + 1)
        # A CPU generator gives the same image for the same seed regardless of device.
        rng = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": params.prompt,
            "width": params.width,
            "height": params.height,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "generator": rng,
        }
        if params.negative_prompt:
            kwargs["negative_prompt"] = params.negative_prompt

        with self._run_lock:
            try:
                with torch.inference_mode():
                    output = pipe(**kwargs)
            except Exception as exc:
                if is_out_of_memory(exc):
                    self._release_gpu_memory()
                raise

        images = getattr(output, "images", None)
        if not images:
            raise GenerationFailedError()
        nsfw = getattr(output, "nsfw_content_detected", None)
        if nsfw and nsfw[0]:
            raise ContentFilteredError()

        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        return GenerationResult(png=buffer.getvalue(), seed=seed)

    def _release_gpu_memory(self) -> None:
        torch = self._torch
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("empty_cache failed", exc_info=True)
