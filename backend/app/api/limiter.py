"""Bounded queue in front of the (single, non-thread-safe) diffusion pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..errors import QueueTimeoutError, ServerBusyError


class ConcurrencyLimiter:
    """Runs at most ``max_running`` generations; at most ``max_waiting`` requests may queue.

    Requests beyond the queue size are rejected immediately (429) instead of piling up,
    and queued requests give up after ``timeout_seconds`` (503).
    Must be created inside the running event loop (e.g. in the app lifespan).
    """

    def __init__(self, *, max_waiting: int, timeout_seconds: float, max_running: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(max_running)
        self._max_waiting = max_waiting
        self._timeout = timeout_seconds
        self._waiting = 0
        self._running = 0

    @property
    def running(self) -> int:
        return self._running

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def max_waiting(self) -> int:
        return self._max_waiting

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._semaphore.locked() and self._waiting >= self._max_waiting:
            raise ServerBusyError()

        self._waiting += 1
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
        except TimeoutError:
            raise QueueTimeoutError() from None
        finally:
            self._waiting -= 1

        self._running += 1
        try:
            yield
        finally:
            self._running -= 1
            self._semaphore.release()
