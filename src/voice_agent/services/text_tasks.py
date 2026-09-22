"""Pluggable text tasks: ASR -> TTS is the core pipeline, LLM is middleware.

Each deployment wires a different task without touching the orchestrator:

- ``passthrough``: repeat transcript straight to TTS (no LLM, lowest latency).
- ``llm``: stream an LLM response (default Groq service, swappable).
- ``router``: keyword/intent rules dispatch to sub-tasks per system needs.

Custom tasks plug in via the registry::

    from voice_agent.services.text_tasks import register_text_task

    @register_text_task("support-bot")
    def make_support_bot(**kwargs) -> TextTask:
        return MyTask()
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Protocol


class TextTask(Protocol):
    """A unit of work turning a transcript into speakable text chunks."""

    name: str

    async def stream(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield response chunks (sentences or clauses) for TTS."""
        ...


Factory = Callable[..., TextTask]

_registry: dict[str, Factory] = {}


def register_text_task(name: str) -> Callable[[Factory], Factory]:
    """Decorator registering a text-task factory."""

    def decorator(factory: Factory) -> Factory:
        _registry[name] = factory
        return factory

    return decorator


def create_text_task(name: str, **kwargs: object) -> TextTask:
    """Instantiate a registered text task."""
    try:
        factory = _registry[name]
    except KeyError:
        available = sorted(_registry)
        raise ValueError(f"Unknown text task {name!r}. Available: {available}") from None
    return factory(**kwargs)  # type: ignore[arg-type]


def available_tasks() -> list[str]:
    """List registered text-task names."""
    return sorted(_registry)


class PassthroughTask:
    """Repeat the transcript to TTS (ASR -> TTS direct, no LLM call)."""

    name = "passthrough"

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    async def stream(
        self, transcript: str, history: list[dict[str, str]]  # noqa: ARG002 - interface arg
    ) -> AsyncIterator[str]:
        text = f"{self._prefix}{transcript}".strip()
        if text:
            yield text


class LLMTask:
    """Stream an LLM response; works with any ILLMService-compatible object."""

    name = "llm"

    def __init__(self, llm: object) -> None:
        self._llm = llm

    async def stream(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        async for sentence in self._llm.stream(transcript, history):  # type: ignore[union-attr]
            yield sentence


class RouterTask:
    """Dispatch transcripts to sub-tasks by predicate rules.

    Rules are (predicate, task) pairs evaluated in order; the first
    predicate returning True wins, otherwise the default task runs.
    """

    name = "router"

    def __init__(
        self,
        rules: list[tuple[Callable[[str], bool], TextTask]] | None = None,
        default: TextTask | None = None,
    ) -> None:
        self._rules = rules or []
        self._default = default or PassthroughTask()

    def add_rule(self, predicate: Callable[[str], bool], task: TextTask) -> None:
        self._rules.append((predicate, task))

    async def stream(
        self, transcript: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        task = self._default
        lowered = transcript.lower()
        for predicate, candidate in self._rules:
            try:
                if predicate(lowered):
                    task = candidate
                    break
            except Exception:
                continue
        async for chunk in task.stream(transcript, history):
            yield chunk


def keyword_rule(*keywords: str) -> Callable[[str], bool]:
    """Build a predicate matching any keyword (case-insensitive)."""
    lowered = [k.lower() for k in keywords]

    def predicate(transcript_lower: str) -> bool:
        return any(k in transcript_lower for k in lowered)

    return predicate


@register_text_task("passthrough")
def _make_passthrough(**kwargs: object) -> TextTask:
    prefix = str(kwargs.get("prefix", ""))
    return PassthroughTask(prefix=prefix)


@register_text_task("llm")
def _make_llm(**kwargs: object) -> TextTask:
    llm = kwargs.get("llm")
    if llm is None:
        raise ValueError("'llm' task requires an llm= service instance")
    return LLMTask(llm=llm)


@register_text_task("router")
def _make_router(**kwargs: object) -> TextTask:
    default = kwargs.get("default")
    if default is not None and not hasattr(default, "stream"):
        raise ValueError("'router' default must be a TextTask")
    return RouterTask(default=default)  # type: ignore[arg-type]


async def collect(task: TextTask, transcript: str, history: list[dict[str, str]]) -> str:
    """Gather all chunks (helper for non-streaming callers/tests)."""
    parts: list[str] = []
    async for chunk in task.stream(transcript, history):
        parts.append(chunk)
    return " ".join(parts).strip()
