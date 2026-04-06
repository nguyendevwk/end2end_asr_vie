"""Structured logging with performance tracking."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

P = ParamSpec("P")
R = TypeVar("R")


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """
    Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON logs (for production)
    """
    # Configure structlog
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        >>> with Timer() as t:
        ...     do_something()
        >>> print(f"Took {t.elapsed_ms}ms")
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._end == 0.0:
            return (time.perf_counter() - self._start) * 1000
        return (self._end - self._start) * 1000


@contextmanager
def log_latency(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    **extra: Any,
) -> Generator[Timer, None, None]:
    """
    Context manager that logs operation latency.

    Args:
        logger: Logger instance
        operation: Name of the operation
        **extra: Additional fields to log

    Example:
        >>> with log_latency(logger, "asr_transcribe", audio_size=1024):
        ...     result = await transcribe(audio)
    """
    timer = Timer()
    try:
        timer.__enter__()
        yield timer
    finally:
        timer.__exit__(None, None, None)
        logger.info(
            f"{operation}_complete",
            latency_ms=round(timer.elapsed_ms, 2),
            **extra,
        )


def track_latency(
    operation: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to track and log function latency.

    Args:
        operation: Name of the operation for logging

    Example:
        >>> @track_latency("asr_transcribe")
        ... async def transcribe(self, audio: bytes) -> str:
        ...     ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        logger = get_logger(func.__module__)

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Timer() as t:
                result = await func(*args, **kwargs)  # type: ignore[misc]
            logger.debug(
                f"{operation}_complete",
                latency_ms=round(t.elapsed_ms, 2),
            )
            return result

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Timer() as t:
                result = func(*args, **kwargs)
            logger.debug(
                f"{operation}_complete",
                latency_ms=round(t.elapsed_ms, 2),
            )
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator
