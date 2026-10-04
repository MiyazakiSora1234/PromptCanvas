"""Face / pose references in a generation: the InstantID adapter in the UNet and the ControlNet pipeline."""

from __future__ import annotations

import logging
from typing import Any

from ..catalog import IdentityConfig
from ..errors import AppError, ReferenceUnavailableError, describe_load_error
from .identity import IdentityConditioner
from .model_cache import are_files_cached
from .models import LoadedModel
from .params import GenerationParams
from .runtime import Runtime

logger = logging.getLogger(__name__)


class ReferenceControl:
    def __init__(self, config: IdentityConfig | None, runtime: Runtime) -> None:
        self._cfg = config
        self._runtime = runtime
        self._conditioner: IdentityConditioner | None = None
        self.cached = False  # assets on disk

    def scan_cache(self) -> None:
        if self._cfg is not None:
            self.cached = are_files_cached(self._cfg.files())

    def _identity(self) -> IdentityConditioner:
        if self._conditioner is None:
            assert self._cfg is not None  # request validation guarantees this
            rt = self._runtime
            self._conditioner = IdentityConditioner(self._cfg, str(rt.device), rt.dtype, rt.hf_token)
        return self._conditioner

    def sync_ip_adapter(self, loaded: LoadedModel, *, needed: bool) -> None:
        """Patch InstantID's adapter into the UNet only for face requests (it changes every attention layer)."""
        pipe = loaded.text2img
        if needed and not loaded.ip_adapter_loaded:
            logger.info("Loading InstantID adapter into %s", loaded.entry.id)
            state = self._identity().adapter_state()
            pipe.load_ip_adapter(state, subfolder="", weight_name="", image_encoder_folder=None)
            # The adapter's projection/attention weights arrive as float32; match the UNet's dtype.
            pipe.unet.to(dtype=self._runtime.dtype)
            loaded.ip_adapter_loaded = True
        elif not needed and loaded.ip_adapter_loaded:
            pipe.unload_ip_adapter()
            loaded.ip_adapter_loaded = False

    def pipeline(self, loaded: LoadedModel, params: GenerationParams) -> tuple[Any, dict[str, Any]]:
        """The ControlNet pipeline (sharing the model's weights) and its extra call arguments."""
        from diffusers import StableDiffusionXLControlNetPipeline
        from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

        try:
            setup = self._identity().prepare(
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
            self.park()  # e.g. no face found after the nets moved to the GPU
            raise
        except Exception as exc:
            self.park()
            hint = describe_load_error(exc)
            logger.exception("Failed to prepare face/pose reference. %s", hint)
            raise ReferenceUnavailableError(f"顔・ポーズ参照用のモデルを読み込めませんでした。{hint}") from None

        cls: Any = StableDiffusionXLControlNetPipeline  # diffusers is only partially typed
        # from_pipe casts the shared components to float32 unless a dtype is given.
        controlnet = MultiControlNetModel(setup.controlnets)
        pipe = cls.from_pipe(loaded.text2img, controlnet=controlnet, torch_dtype=self._runtime.dtype)
        pipe.set_progress_bar_config(disable=True)
        extra: dict[str, Any] = {
            "image": setup.images,
            "controlnet_conditioning_scale": setup.scales,
            "control_guidance_start": [0.0] * len(setup.guidance_ends),
            "control_guidance_end": setup.guidance_ends,
        }
        if setup.ip_adapter_embeds is not None:
            # Alongside IdentityNet the adapter is toned down (more natural skin); alone it carries the likeness.
            ratio = self._cfg.ip_adapter_ratio if self._cfg and setup.identity_net else 1.0
            loaded.text2img.set_ip_adapter_scale(params.identity_strength * ratio)
            extra["ip_adapter_image_embeds"] = [setup.ip_adapter_embeds]
        logger.info("Reference: identity_net=%s pose=%s", setup.identity_net, setup.pose_mode)
        return pipe, extra

    def park(self) -> None:
        """Move the ControlNets back to CPU RAM and free their VRAM."""
        if self._conditioner is not None:
            self._conditioner.park()
            self._runtime.release_memory()

    def release_unless_supported(self, family: str) -> None:
        """Drop the ControlNets when switching to a model family they don't fit."""
        if self._conditioner is not None and self._cfg is not None and family not in self._cfg.families:
            self._conditioner.release()
            self._runtime.release_memory()
