"""Cancellation flags for in-flight generations, keyed by a client-chosen job id."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class CancelRegistry:
    """The client sends a random job id with its request and may later cancel it by that id.

    The generator thread polls the returned event between denoising steps.
    """

    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    @contextmanager
    def register(self, job_id: str | None) -> Iterator[threading.Event]:
        event = threading.Event()
        if job_id is None:  # not cancellable, but the caller can use the event the same way
            yield event
            return
        with self._lock:
            self._events[job_id] = event
        try:
            yield event
        finally:
            with self._lock:
                if self._events.get(job_id) is event:
                    del self._events[job_id]

    def cancel(self, job_id: str) -> bool:
        """Flag the job; False if no such job is running or waiting."""
        with self._lock:
            event = self._events.get(job_id)
        if event is None:
            return False
        event.set()
        return True
