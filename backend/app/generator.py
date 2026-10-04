"""Diffusers pipeline wrapper.

Keeps exactly one catalog model on the device at a time (switching on demand),
and serves text-to-image and image-to-image from the same weights.

``torch`` and ``diffusers`` are imported lazily so that the API layer and tests
can run without them installed.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import secrets
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .catalog import Catalog, LoraEntry, ModelEntry
from .config import SEED_MAX, Settings
from .errors import (
    AppError,
    ConfigurationError,
    ContentFilteredError,
    GenerationCancelledError,
    GenerationFailedError,
    LoraUnavailableError,
    ModelLoadingError,
    ModelUnavailableError,
    ReferenceUnavailableError,
    TranslationUnavailableError,
    describe_load_error,
    is_out_of_memory,
)
from .identity import IdentityConditioner
from .imaging import OUTPUT_FORMATS, encode_image, fit_to
from .model_cache import are_files_cached, is_model_cached
from .schedulers import build_scheduler
from .schemas import GenerationParams
from .styles import apply_style
from .translate import PromptTranslator, has_japanese

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


def batch_seeds(seed: int | None, count: int) -> list[int]:
    """One seed per image: the given (or a random) seed, then +1, +2, ... wrapping at 2^32."""
    base = seed if seed is not None else secrets.randbelow(SEED_MAX + 1)
    return [(base + i) % (SEED_MAX + 1) for i in range(count)]


def adapter_name(lora: LoraEntry) -> str:
    return lora.id.replace("-", "_")


@dataclass
class _LoadedModel:
    entry: ModelEntry
    text2img: Any
    img2img: Any
    original_scheduler: Any
    loras: set[str] = field(default_factory=set)
    # InstantID's IP-Adapter is patched into the shared UNet only while it is needed.
    ip_adapter_loaded: bool = False


class DiffusersGenerator:
    def __init__(self, settings: Settings, catalog: Catalog) -> None:
        self._settings = settings
        self._catalog = catalog
        self._torch: Any = None
        self._device: str | None = None
        self._dtype_name: str | None = None
        self._loaded: _LoadedModel | None = None
        self._status = ModelStatus(ModelState.NOT_LOADED, model=catalog.default_model)
        self._cached: frozenset[str] = frozenset()
        self._identity_cached = False
        self._identity: IdentityConditioner | None = None
        self._translator: PromptTranslator | None = None
        self._translator_cached = False
        self._status_lock = threading.Lock()
        # Diffusers pipelines are not thread-safe, and model switching must not overlap a run.
        self._run_lock = threading.Lock()

    @property
    def status(self) -> ModelStatus:
        with self._status_lock:
            return self._status

    @property
    def cached_models(self) -> frozenset[str]:
        with self._status_lock:
            return self._cached

    @property
    def identity_cached(self) -> bool:
        return self._identity_cached

    @property
    def translator_cached(self) -> bool:
        return self._translator_cached

    def _set_status(self, state: ModelState, model: str | None, message: str | None = None) -> None:
        with self._status_lock:
            self._status = ModelStatus(state, model, self._device, self._dtype_name, message)
            if state is ModelState.READY and model is not None:
                self._cached = self._cached | {model}

    # --- Loading -----------------------------------------------------------

    def load(self) -> None:
        """Load the default model (called once at startup, in a background thread)."""
        cached = frozenset(
            m.id
            for m in self._catalog.models
            if is_model_cached(m.repo, m.revision) and (m.vae is None or are_files_cached(m.vae.files()))
        )
        with self._status_lock:
            self._cached = self._cached | cached
        logger.info("Models available offline: %s", ", ".join(sorted(cached)) or "none")
        if self._catalog.identity is not None:
            self._identity_cached = are_files_cached(self._catalog.identity.files())
        if self._catalog.translator is not None:
            self._translator_cached = are_files_cached(self._catalog.translator.files())
        # A failure is already logged and reflected in status.
        with self._run_lock, contextlib.suppress(ModelUnavailableError):
            self._ensure_model(self._catalog.default)

    def _init_runtime(self) -> None:
        if self._torch is not None:
            return
        import torch

        mps = getattr(torch.backends, "mps", None)
        device = resolve_device(
            self._settings.device,
            cuda_available=torch.cuda.is_available(),
            mps_available=bool(mps and mps.is_available()),
        )
        self._dtype_name = resolve_dtype(self._settings.torch_dtype, device)
        self._device = device
        self._torch = torch
        if device == "cpu":
            logger.warning("GPU が見つからないため CPU で実行します。1枚の生成に数分以上かかる場合があります。")
        elif device == "cuda":
            logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
            fraction = self._settings.cuda_memory_fraction
            if fraction is not None:
                torch.cuda.set_per_process_memory_fraction(fraction)
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("GPU memory cap: %.0f%% of %.1f GB", fraction * 100, total_gb)

    def _ensure_model(self, entry: ModelEntry) -> _LoadedModel:
        """Return the loaded pipeline for `entry`, switching models if needed. Caller holds _run_lock."""
        if self._loaded is not None and self._loaded.entry.id == entry.id:
            return self._loaded

        self._unload()
        self._set_status(ModelState.LOADING, entry.id)
        try:
            self._init_runtime()
            loaded = self._load_pipelines(entry)
        except Exception as exc:
            hint = describe_load_error(exc)
            logger.exception("Failed to load model %s (%s). %s", entry.id, entry.repo, hint)
            self._unload()
            self._set_status(ModelState.FAILED, entry.id, hint)
            raise ModelUnavailableError(f"モデル「{entry.label}」を読み込めませんでした。{hint}") from None

        identity = self._catalog.identity
        if self._identity is not None and identity is not None and entry.family not in identity.families:
            self._identity.release()  # its ControlNets only fit other model families; free the VRAM
            self._release_gpu_memory()
        self._loaded = loaded
        self._set_status(ModelState.READY, entry.id)
        logger.info("Model ready: %s", entry.id)
        return loaded

    def _load_pipelines(self, entry: ModelEntry) -> _LoadedModel:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        s, torch = self._settings, self._torch
        logger.info("Loading model %s (%s) on %s (%s)...", entry.id, entry.repo, self._device, self._dtype_name)
        kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, str(self._dtype_name))}
        if entry.revision:
            kwargs["revision"] = entry.revision
        if entry.variant:
            kwargs["variant"] = entry.variant
        if s.hf_token is not None:
            kwargs["token"] = s.hf_token.get_secret_value()

        if entry.vae is not None:
            from diffusers import AutoencoderKL

            vae_cls: Any = AutoencoderKL
            vae = entry.vae
            kwargs["vae"] = vae_cls.from_pretrained(
                vae.repo, revision=vae.revision, torch_dtype=kwargs["torch_dtype"], token=kwargs.get("token")
            )

        text2img_cls: Any = AutoPipelineForText2Image  # diffusers is only partially typed
        img2img_cls: Any = AutoPipelineForImage2Image
        text2img = text2img_cls.from_pretrained(entry.repo, **kwargs)
        if not s.enable_cpu_offload or self._device != "cuda":
            if s.enable_cpu_offload:
                logger.warning("PROMPTCANVAS_ENABLE_CPU_OFFLOAD is only effective on CUDA; ignored.")
            text2img = text2img.to(self._device)
        if s.enable_attention_slicing:
            text2img.enable_attention_slicing()
        # Decode batches one image at a time: same output, much lower peak VRAM.
        if hasattr(text2img, "vae") and hasattr(text2img.vae, "enable_slicing"):
            text2img.vae.enable_slicing()
        text2img.set_progress_bar_config(disable=True)

        # Shares every component (no extra memory) with the text-to-image pipeline.
        # from_pipe casts the shared components to float32 unless a dtype is given.
        img2img = img2img_cls.from_pipe(text2img, torch_dtype=kwargs["torch_dtype"])
        img2img.set_progress_bar_config(disable=True)
        loaded = _LoadedModel(entry, text2img, img2img, original_scheduler=text2img.scheduler)
        self._move_text_encoders(loaded, on_gpu=False)
        return loaded

    def _offloads_text_encoders(self) -> bool:
        s = self._settings
        return s.offload_text_encoders and self._device == "cuda" and not s.enable_cpu_offload

    def _move_text_encoders(self, loaded: _LoadedModel, *, on_gpu: bool) -> None:
        """Text encoders are only needed to encode the prompt; park them in CPU RAM otherwise."""
        if not self._offloads_text_encoders():
            return
        device = self._device if on_gpu else "cpu"
        for name in ("text_encoder", "text_encoder_2"):
            encoder = getattr(loaded.text2img, name, None)
            if encoder is not None and encoder.device.type != device:
                encoder.to(device)
        if not on_gpu:
            self._release_gpu_memory()

    def _unload(self) -> None:
        if self._loaded is None:
            return
        logger.info("Unloading model %s", self._loaded.entry.id)
        self._loaded = None
        gc.collect()
        self._release_gpu_memory()

    # --- LoRA --------------------------------------------------------------

    def _apply_loras(self, loaded: _LoadedModel, loras: tuple[tuple[LoraEntry, float], ...]) -> None:
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
            if self._settings.hf_token is not None:
                kwargs["token"] = self._settings.hf_token.get_secret_value()
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

    # --- Face / pose reference (InstantID + OpenPose) --------------------------

    def _identity_conditioner(self) -> IdentityConditioner:
        if self._identity is None:
            assert self._catalog.identity is not None  # request validation guarantees this
            token = self._settings.hf_token.get_secret_value() if self._settings.hf_token else None
            dtype = getattr(self._torch, str(self._dtype_name))
            self._identity = IdentityConditioner(self._catalog.identity, str(self._device), dtype, token)
        return self._identity

    def _sync_ip_adapter(self, loaded: _LoadedModel, needed: bool) -> None:
        """Patch InstantID's adapter into the UNet only for face requests (it changes every attention layer)."""
        pipe = loaded.text2img
        if needed and not loaded.ip_adapter_loaded:
            logger.info("Loading InstantID adapter into %s", loaded.entry.id)
            state = self._identity_conditioner().adapter_state()
            pipe.load_ip_adapter(state, subfolder="", weight_name="", image_encoder_folder=None)
            # The adapter's projection/attention weights arrive as float32; match the UNet's dtype.
            pipe.unet.to(dtype=getattr(self._torch, str(self._dtype_name)))
            loaded.ip_adapter_loaded = True
        elif not needed and loaded.ip_adapter_loaded:
            pipe.unload_ip_adapter()
            loaded.ip_adapter_loaded = False

    def _reference_pipeline(self, loaded: _LoadedModel, params: GenerationParams) -> tuple[Any, dict[str, Any]]:
        from diffusers import StableDiffusionXLControlNetPipeline
        from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

        try:
            setup = self._identity_conditioner().prepare(
                loaded.text2img.unet,
                face_images=params.face_images,
                pose_image=params.pose_image,
                width=params.width,
                height=params.height,
                identity_strength=params.identity_strength,
                pose_strength=params.pose_strength,
                num_images=params.num_images,
                guidance=params.guidance_scale > 1,
            )
        except AppError:
            self._identity_conditioner().park()  # e.g. no face found after the nets moved to the GPU
            raise
        except Exception as exc:
            self._identity_conditioner().park()
            hint = describe_load_error(exc)
            logger.exception("Failed to prepare face/pose reference. %s", hint)
            raise ReferenceUnavailableError(f"顔・ポーズ参照用のモデルを読み込めませんでした。{hint}") from None

        cls: Any = StableDiffusionXLControlNetPipeline  # diffusers is only partially typed
        # from_pipe casts the shared components to float32 unless a dtype is given.
        dtype = getattr(self._torch, str(self._dtype_name))
        pipe = cls.from_pipe(loaded.text2img, controlnet=MultiControlNetModel(setup.controlnets), torch_dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        extra: dict[str, Any] = {
            "image": setup.images,
            "controlnet_conditioning_scale": setup.scales,
            "control_guidance_start": [0.0] * len(setup.guidance_ends),
            "control_guidance_end": setup.guidance_ends,
        }
        if setup.ip_adapter_embeds is not None:
            # Alongside IdentityNet the adapter is toned down (more natural skin); alone it carries the likeness.
            identity = self._catalog.identity
            ratio = identity.ip_adapter_ratio if identity and setup.identity_net else 1.0
            loaded.text2img.set_ip_adapter_scale(params.identity_strength * ratio)
            extra["ip_adapter_image_embeds"] = [setup.ip_adapter_embeds]
        logger.info("Reference: identity_net=%s pose=%s", setup.identity_net, setup.pose_mode)
        return pipe, extra

    # --- Generation ----------------------------------------------------------

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
            prompt, negative_prompt, translated = self._translate(params)
            styled_prompt, styled_negative = apply_style(params.style, prompt, negative_prompt)
            loaded = self._ensure_model(params.model)
            self._apply_loras(loaded, params.loras)
            self._sync_ip_adapter(loaded, needed=bool(params.face_images))
            torch = self._torch

            extra: dict[str, Any] = {}
            if params.uses_reference:
                pipe, extra = self._reference_pipeline(loaded, params)
            elif params.init_image is not None:
                pipe = loaded.img2img
            else:
                pipe = loaded.text2img
            pipe.scheduler = build_scheduler(params.scheduler, loaded.original_scheduler)
            if self._settings.enable_cpu_offload and self._device == "cuda":
                pipe.enable_model_cpu_offload()  # (re)install hooks on the pipeline actually used

            seeds = batch_seeds(params.seed, params.num_images)
            # CPU generators give the same image for the same seed regardless of device.
            generators = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            kwargs: dict[str, Any] = {
                "prompt": styled_prompt,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "num_images_per_prompt": params.num_images,
                "generator": generators,
            }
            if styled_negative:
                kwargs["negative_prompt"] = styled_negative
            if params.init_image is not None:
                kwargs["image"] = fit_to(params.init_image, params.width, params.height)
                kwargs["strength"] = params.strength
            else:
                kwargs["width"] = params.width
                kwargs["height"] = params.height
            kwargs.update(extra)

            def on_step_end(running: Any, step: int, _timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
                if step == 0:  # the prompt is encoded by now
                    self._move_text_encoders(loaded, on_gpu=False)
                if cancel is not None and cancel.is_set():
                    raise GenerationCancelledError()
                if params.uses_reference and step >= getattr(running, "num_timesteps", 0) - 1:
                    # The ControlNets (~5GB) are done; free them before the VAE decode needs its peak memory.
                    self._park_identity()
                return tensors

            kwargs["callback_on_step_end"] = on_step_end
            cuda = self._device == "cuda"
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            try:
                self._move_text_encoders(loaded, on_gpu=True)
                with torch.inference_mode():
                    output = pipe(**kwargs)
            except Exception as exc:
                if is_out_of_memory(exc):
                    self._release_gpu_memory()
                raise
            finally:
                self._move_text_encoders(loaded, on_gpu=False)
                if params.uses_reference:
                    self._park_identity()
                if cuda:
                    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                    # Hand the per-request working memory back so other apps (and Task Manager) see it free.
                    self._release_gpu_memory()
                    logger.info(
                        "VRAM: peak %.1f GB during generation, %.1f GB held after",
                        peak_gb,
                        torch.cuda.memory_reserved() / 1024**3,
                    )

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
        return GenerationResult(
            images=kept,
            filtered_count=len(images) - len(kept),
            translated_prompt=prompt if translated else None,
            translated_negative_prompt=negative_prompt if translated and params.negative_prompt else None,
        )

    # --- Japanese prompts ------------------------------------------------------

    def _translate(self, params: GenerationParams) -> tuple[str, str, bool]:
        """(prompt, negative prompt, whether anything was translated). Caller holds _run_lock."""
        config = self._catalog.translator
        if config is None or not (has_japanese(params.prompt) or has_japanese(params.negative_prompt)):
            return params.prompt, params.negative_prompt, False
        self._init_runtime()
        if self._translator is None:
            token = self._settings.hf_token.get_secret_value() if self._settings.hf_token else None
            dtype = getattr(self._torch, str(self._dtype_name))
            self._translator = PromptTranslator(config, str(self._device), dtype, token)
        try:
            prompt = self._translator.translate(params.prompt)
            negative_prompt = self._translator.translate(params.negative_prompt)
        except Exception as exc:
            if is_out_of_memory(exc):
                self._release_gpu_memory()
                raise
            logger.exception("Failed to translate the prompt. %s", describe_load_error(exc))
            raise TranslationUnavailableError() from None
        self._translator_cached = True
        return prompt, negative_prompt, True

    def _park_identity(self) -> None:
        if self._identity is not None:
            self._identity.park()
            self._release_gpu_memory()

    def _release_gpu_memory(self) -> None:
        torch = self._torch
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("empty_cache failed", exc_info=True)
