"""Resilience helpers: retry with backoff, timeouts, admission gate."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    timeout_s: float | None = None,
) -> T:
    """
    Retry an async callable on transient failure.

    Args:
        fn: Zero-arg async callable.
        attempts: Total attempts (1 = no retry).
        base_delay_s: Initial backoff; doubles each retry with jitter.
        max_delay_s: Backoff cap.
        timeout_s: Per-attempt timeout (None = no timeout).

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: BaseException | None = None
    delay = base_delay_s
    for _ in range(max(1, attempts)):
        try:
            if timeout_s is not None:
                return await asyncio.wait_for(fn(), timeout_s)
            return await fn()
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as e:
            last_exc = e
            await asyncio.sleep(min(delay, max_delay_s) + random.uniform(0, 0.05))
            delay *= 2
    assert last_exc is not None
    raise last_exc


class AdmissionGate:
    """Bounded concurrency gate for connections (CCU shedding)."""

    def __init__(self, max_concurrent: int) -> None:
        self._max = max(1, max_concurrent)
        self._current = 0
        self._lock = asyncio.Lock()
        self._high_watermark = 0

    async def acquire(self) -> bool:
        """Try to admit one connection. Returns False when full (shed)."""
        async with self._lock:
            if self._current >= self._max:
                return False
            self._current += 1
            self._high_watermark = max(self._high_watermark, self._current)
            return True

    async def release(self) -> None:
        async with self._lock:
            self._current = max(0, self._current - 1)

    @property
    def current(self) -> int:
        return self._current

    @property
    def high_watermark(self) -> int:
        return self._high_watermark

    @property
    def limit(self) -> int:
        return self._max

    def set_limit(self, new_max: int) -> None:
        """Update the max concurrent connections (startup knob sync)."""
        self._max = max(1, new_max)

    async def __aenter__(self) -> bool:
        return await self.acquire()

    async def __aexit__(self, *args: Any) -> None:
        await self.release()
