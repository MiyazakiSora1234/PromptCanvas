"""torch, the device and the dtype (resolved once, on first use), plus GPU memory housekeeping."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..config import Settings
from ..errors import ConfigurationError

logger = logging.getLogger(__name__)


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


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.torch: Any = None
        self.device: str | None = None
        self.dtype_name: str | None = None

    def ensure(self) -> None:
        """Import torch and pick the device / dtype (raises ConfigurationError for impossible settings)."""
        if self.torch is not None:
            return
        import torch

        mps = getattr(torch.backends, "mps", None)
        device = resolve_device(
            self.settings.device,
            cuda_available=torch.cuda.is_available(),
            mps_available=bool(mps and mps.is_available()),
        )
        self.dtype_name = resolve_dtype(self.settings.torch_dtype, device)
        self.device = device
        self.torch = torch
        if device == "cpu":
            logger.warning("GPU が見つからないため CPU で実行します。1枚の生成に数分以上かかる場合があります。")
        elif device == "cuda":
            logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
            fraction = self.settings.cuda_memory_fraction
            if fraction is not None:
                torch.cuda.set_per_process_memory_fraction(fraction)
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("GPU memory cap: %.0f%% of %.1f GB", fraction * 100, total_gb)

    @property
    def dtype(self) -> Any:
        """The torch dtype object (call ensure() first)."""
        return getattr(self.torch, str(self.dtype_name))

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def hf_token(self) -> str | None:
        token = self.settings.hf_token
        return token.get_secret_value() if token is not None else None

    def release_memory(self) -> None:
        """Hand cached-but-unused GPU memory back to the driver."""
        torch = self.torch
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("empty_cache failed", exc_info=True)

    @contextmanager
    def track_peak_memory(self) -> Iterator[None]:
        """Log the peak VRAM of the enclosed work, then free its working memory."""
        if not self.is_cuda:
            yield
            return
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            # Hand the per-request working memory back so other apps (and Task Manager) see it free.
            self.release_memory()
            logger.info(
                "VRAM: peak %.1f GB during generation, %.1f GB held after",
                peak_gb,
                torch.cuda.memory_reserved() / 1024**3,
            )
