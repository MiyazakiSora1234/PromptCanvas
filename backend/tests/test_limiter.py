from __future__ import annotations

import asyncio

import pytest

from app.api.limiter import ConcurrencyLimiter
from app.errors import (
    QueueTimeoutError,
    ServerBusyError,
)

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
