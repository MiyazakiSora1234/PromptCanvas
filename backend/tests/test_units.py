from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.errors import (
    ConfigurationError,
    GenerationFailedError,
    GpuOutOfMemoryError,
    QueueTimeoutError,
    ServerBusyError,
    classify_generation_error,
    describe_load_error,
)
from app.generator import resolve_device, resolve_dtype
from app.limiter import ConcurrencyLimiter

from .conftest import make_settings

# --- Limiter -----------------------------------------------------------------


def test_limiter_rejects_when_queue_full() -> None:
    async def scenario() -> None:
        limiter = ConcurrencyLimiter(max_waiting=0, timeout_seconds=1)
        async with limiter.slot():
            assert limiter.running == 1
            with pytest.raises(ServerBusyError):
                async with limiter.slot():
                    pass
        assert limiter.running == 0

    asyncio.run(scenario())


def test_limiter_times_out_waiting_request() -> None:
    async def scenario() -> None:
        limiter = ConcurrencyLimiter(max_waiting=1, timeout_seconds=0.05)
        async with limiter.slot():
            with pytest.raises(QueueTimeoutError):
                async with limiter.slot():
                    pass
        assert limiter.waiting == 0

    asyncio.run(scenario())


def test_limiter_runs_waiting_request_after_release() -> None:
    async def scenario() -> list[str]:
        limiter = ConcurrencyLimiter(max_waiting=1, timeout_seconds=1)
        order: list[str] = []

        async def job(name: str) -> None:
            async with limiter.slot():
                order.append(f"{name}-start")
                await asyncio.sleep(0.01)
                order.append(f"{name}-end")

        await asyncio.gather(job("a"), job("b"))
        return order

    assert asyncio.run(scenario()) == ["a-start", "a-end", "b-start", "b-end"]


# --- Error classification ---------------------------------------------------


class OutOfMemoryError(RuntimeError):
    """Same name as torch.cuda.OutOfMemoryError."""


def test_classify_out_of_memory() -> None:
    assert isinstance(classify_generation_error(OutOfMemoryError("x")), GpuOutOfMemoryError)
    assert isinstance(classify_generation_error(RuntimeError("MPS backend out of memory")), GpuOutOfMemoryError)
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = OutOfMemoryError("inner")
    assert isinstance(classify_generation_error(wrapped), GpuOutOfMemoryError)


def test_classify_other_errors_as_generic() -> None:
    assert isinstance(classify_generation_error(ValueError("boom")), GenerationFailedError)


class GatedRepoError(Exception):
    pass


def test_describe_load_error_hides_raw_text() -> None:
    message = describe_load_error(GatedRepoError("401 https://huggingface.co/... secret"))
    assert "HF_TOKEN" in message
    assert "secret" not in message
    assert "secret" not in describe_load_error(OSError("some secret path"))


# --- Device / dtype ---------------------------------------------------------


def test_resolve_device() -> None:
    assert resolve_device("auto", cuda_available=True, mps_available=False) == "cuda"
    assert resolve_device("auto", cuda_available=False, mps_available=True) == "mps"
    assert resolve_device("auto", cuda_available=False, mps_available=False) == "cpu"
    with pytest.raises(ConfigurationError):
        resolve_device("cuda", cuda_available=False, mps_available=False)


def test_resolve_dtype() -> None:
    assert resolve_dtype("auto", "cuda") == "float16"
    assert resolve_dtype("auto", "cpu") == "float32"
    with pytest.raises(ConfigurationError):
        resolve_dtype("float16", "cpu")


# --- Settings ---------------------------------------------------------------


def test_settings_reject_inconsistent_defaults() -> None:
    with pytest.raises(ValidationError):
        make_settings(default_width=500)
    with pytest.raises(ValidationError):
        make_settings(default_steps=60, max_steps=50)


def test_hf_token_is_not_exposed_in_repr() -> None:
    settings = make_settings(HF_TOKEN="hf_supersecret")
    assert settings.hf_token is not None
    assert "hf_supersecret" not in repr(settings)
