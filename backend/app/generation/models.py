"""Model management: keeps exactly one catalog model on the device (switching on demand).

Text-to-image and image-to-image share the same weights. Also owns the per-model extras:
LoRA adapters, the replacement VAE and keeping the text encoders in CPU RAM.
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..catalog import Catalog, LoraEntry, ModelEntry
from ..errors import LoraUnavailableError, ModelLoadingError, ModelUnavailableError, describe_load_error
from .model_cache import are_files_cached, is_model_cached
from .runtime import Runtime

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelStatus:
    state: ModelState
    model: str | None = None  # catalog id loaded / being loaded / that failed
    device: str | None = None
    dtype: str | None = None
    # Safe, fixed hint text (never raw exception output).
    message: str | None = None

    def ensure_ready(self) -> None:
        """Reject requests while a model is being loaded.

        A FAILED state does not block: the next request retries the load (or picks
        another model), and reports the failure itself if it happens again.
        """
        if self.state in (ModelState.NOT_LOADED, ModelState.LOADING):
            raise ModelLoadingError()


def adapter_name(lora: LoraEntry) -> str:
    return lora.id.replace("-", "_")


@dataclass
class LoadedModel:
    entry: ModelEntry
    text2img: Any
    img2img: Any
    original_scheduler: Any
    loras: set[str] = field(default_factory=set)
    # InstantID's IP-Adapter is patched into the shared UNet only while it is needed.
    ip_adapter_loaded: bool = False


class ModelManager:
    """Not thread-safe by itself: callers serialize loading and generation (DiffusersGenerator's run lock)."""

    def __init__(self, catalog: Catalog, runtime: Runtime) -> None:
        self._catalog = catalog
        self._runtime = runtime
        self._loaded: LoadedModel | None = None
        self._status = ModelStatus(ModelState.NOT_LOADED, model=catalog.default_model)
        self._cached: frozenset[str] = frozenset()
        self._status_lock = threading.Lock()  # status is read from request handlers

    @property
    def status(self) -> ModelStatus:
        with self._status_lock:
            return self._status

    @property
    def cached_models(self) -> frozenset[str]:
        """Catalog ids whose weights are already on disk (usable without a download)."""
        with self._status_lock:
            return self._cached

    def scan_cache(self) -> None:
        cached = frozenset(
            m.id
            for m in self._catalog.models
            if is_model_cached(m.repo, m.revision) and (m.vae is None or are_files_cached(m.vae.files()))
        )
        with self._status_lock:
            self._cached = self._cached | cached
        logger.info("Models available offline: %s", ", ".join(sorted(cached)) or "none")

    def _set_status(self, state: ModelState, model: str | None, message: str | None = None) -> None:
        rt = self._runtime
        with self._status_lock:
            self._status = ModelStatus(state, model, rt.device, rt.dtype_name, message)
            if state is ModelState.READY and model is not None:
                self._cached = self._cached | {model}

    # --- Loading -----------------------------------------------------------

    def ensure(self, entry: ModelEntry) -> LoadedModel:
        """The loaded pipelines for `entry`, switching models if needed."""
        if self._loaded is not None and self._loaded.entry.id == entry.id:
            return self._loaded

        self.unload()
        self._set_status(ModelState.LOADING, entry.id)
        try:
            self._runtime.ensure()
            loaded = self._load_pipelines(entry)
        except Exception as exc:
            hint = describe_load_error(exc)
            logger.exception("Failed to load model %s (%s). %s", entry.id, entry.repo, hint)
            self.unload()
            self._set_status(ModelState.FAILED, entry.id, hint)
            raise ModelUnavailableError(f"モデル「{entry.label}」を読み込めませんでした。{hint}") from None

        self._loaded = loaded
        self._set_status(ModelState.READY, entry.id)
        logger.info("Model ready: %s", entry.id)
        return loaded

    def _load_pipelines(self, entry: ModelEntry) -> LoadedModel:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        rt, s = self._runtime, self._runtime.settings
        logger.info("Loading model %s (%s) on %s (%s)...", entry.id, entry.repo, rt.device, rt.dtype_name)
        kwargs: dict[str, Any] = {"torch_dtype": rt.dtype}
        if entry.revision:
            kwargs["revision"] = entry.revision
        if entry.variant:
            kwargs["variant"] = entry.variant
        if rt.hf_token is not None:
            kwargs["token"] = rt.hf_token

        if entry.vae is not None:
            from diffusers import AutoencoderKL

            vae_cls: Any = AutoencoderKL
            vae = entry.vae
            kwargs["vae"] = vae_cls.from_pretrained(
                vae.repo, revision=vae.revision, torch_dtype=rt.dtype, token=rt.hf_token
            )

        text2img_cls: Any = AutoPipelineForText2Image  # diffusers is only partially typed
        img2img_cls: Any = AutoPipelineForImage2Image
        text2img = text2img_cls.from_pretrained(entry.repo, **kwargs)
        if not s.enable_cpu_offload or not rt.is_cuda:
            if s.enable_cpu_offload:
                logger.warning("PROMPTCANVAS_ENABLE_CPU_OFFLOAD is only effective on CUDA; ignored.")
            text2img = text2img.to(rt.device)
        if s.enable_attention_slicing:
            text2img.enable_attention_slicing()
        # Decode batches one image at a time: same output, much lower peak VRAM.
        if hasattr(text2img, "vae") and hasattr(text2img.vae, "enable_slicing"):
            text2img.vae.enable_slicing()
        text2img.set_progress_bar_config(disable=True)

        # Shares every component (no extra memory) with the text-to-image pipeline.
        # from_pipe casts the shared components to float32 unless a dtype is given.
        img2img = img2img_cls.from_pipe(text2img, torch_dtype=rt.dtype)
        img2img.set_progress_bar_config(disable=True)
        loaded = LoadedModel(entry, text2img, img2img, original_scheduler=text2img.scheduler)
        self.move_text_encoders(loaded, on_gpu=False)
        return loaded

    def unload(self) -> None:
        if self._loaded is None:
            return
        logger.info("Unloading model %s", self._loaded.entry.id)
        self._loaded = None
        gc.collect()
        self._runtime.release_memory()

    # --- Text encoders -----------------------------------------------------

    def _offloads_text_encoders(self) -> bool:
        s = self._runtime.settings
        return s.offload_text_encoders and self._runtime.is_cuda and not s.enable_cpu_offload

    def move_text_encoders(self, loaded: LoadedModel, *, on_gpu: bool) -> None:
        """Text encoders are only needed to encode the prompt; park them in CPU RAM otherwise."""
        if not self._offloads_text_encoders():
            return
        device = str(self._runtime.device) if on_gpu else "cpu"
        for name in ("text_encoder", "text_encoder_2"):
            encoder = getattr(loaded.text2img, name, None)
            if encoder is not None and encoder.device.type != device:
                encoder.to(device)
        if not on_gpu:
            self._runtime.release_memory()

    # --- LoRA --------------------------------------------------------------

    def apply_loras(self, loaded: LoadedModel, loras: tuple[tuple[LoraEntry, float], ...]) -> None:
        pipe = loaded.text2img  # img2img shares the same UNet/text encoders, so adapters apply to both
        for lora, _ in loras:
            if lora.id in loaded.loras:
                continue
            logger.info("Loading LoRA %s (%s)", lora.id, lora.repo)
            kwargs: dict[str, Any] = {"adapter_name": adapter_name(lora)}
            if lora.weight_name:
                kwargs["weight_name"] = lora.weight_name
            if lora.revision:
                kwargs["revision"] = lora.revision
            if self._runtime.hf_token is not None:
                kwargs["token"] = self._runtime.hf_token
            try:
                pipe.load_lora_weights(lora.repo, **kwargs)
            except Exception as exc:
                hint = describe_load_error(exc)
                logger.exception("Failed to load LoRA %s. %s", lora.id, hint)
                raise LoraUnavailableError(f"LoRA「{lora.label}」を読み込めませんでした。{hint}") from None
            loaded.loras.add(lora.id)

        if loras:
            pipe.enable_lora()
            pipe.set_adapters([adapter_name(lora) for lora, _ in loras], adapter_weights=[w for _, w in loras])
        elif loaded.loras:
            pipe.disable_lora()
