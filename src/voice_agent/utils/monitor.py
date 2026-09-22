"""Performance monitoring and metrics collection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from voice_agent.utils.logger import Timer, get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Statistics for a latency metric."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)

    @property
    def avg_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count > 0 else 0.0,
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class PipelineMonitor:
    """
    Monitor for tracking pipeline performance metrics.

    Tracks latencies for each component and overall pipeline.
    """

    session_id: str
    turn_id: int = 0

    # Latency stats per component
    vad_stats: LatencyStats = field(default_factory=LatencyStats)
    asr_stats: LatencyStats = field(default_factory=LatencyStats)
    llm_stats: LatencyStats = field(default_factory=LatencyStats)
    tts_stats: LatencyStats = field(default_factory=LatencyStats)
    ttfa_stats: LatencyStats = field(default_factory=LatencyStats)
    e2e_stats: LatencyStats = field(default_factory=LatencyStats)

    # Current turn tracking
    _turn_start: float = field(default=0.0, repr=False)
    _first_audio_time: float = field(default=0.0, repr=False)

    def start_turn(self) -> None:
        """Mark the start of a new turn."""
        self.turn_id += 1
        self._turn_start = time.perf_counter()
        self._first_audio_time = 0.0
        logger.info(
            "turn_started",
            session_id=self.session_id,
            turn_id=self.turn_id,
        )

    def mark_processing_start(self) -> None:
        """
        Reset turn timer at speech end (processing start).

        TTFA/E2E must be measured from when the user *stops* speaking,
        not from speech start — otherwise metrics include seconds of
        user talk time and become useless for latency tuning.
        """
        self._turn_start = time.perf_counter()
        self._first_audio_time = 0.0

    def record_vad(self, latency_ms: float, is_speech: bool) -> None:
        """Record VAD detection latency."""
        self.vad_stats.record(latency_ms)
        logger.debug(
            "vad_detected",
            session_id=self.session_id,
            turn_id=self.turn_id,
            latency_ms=round(latency_ms, 2),
            is_speech=is_speech,
        )

    def record_asr(
        self,
        latency_ms: float,
        audio_duration_ms: float,
        transcript_length: int,
    ) -> None:
        """Record ASR transcription latency."""
        self.asr_stats.record(latency_ms)
        rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        logger.info(
            "asr_complete",
            session_id=self.session_id,
            turn_id=self.turn_id,
            latency_ms=round(latency_ms, 2),
            audio_duration_ms=round(audio_duration_ms, 2),
            transcript_length=transcript_length,
            rtf=round(rtf, 3),
        )

    def record_llm(
        self,
        ttft_ms: float,
        total_ms: float,
        token_count: int,
    ) -> None:
        """Record LLM generation latency."""
        self.llm_stats.record(total_ms)
        tokens_per_sec = token_count / (total_ms / 1000) if total_ms > 0 else 0
        logger.info(
            "llm_complete",
            session_id=self.session_id,
            turn_id=self.turn_id,
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            token_count=token_count,
            tokens_per_sec=round(tokens_per_sec, 1),
        )

    def record_tts(
        self,
        latency_ms: float,
        text_length: int,
        audio_duration_ms: float,
    ) -> None:
        """Record TTS synthesis latency."""
        self.tts_stats.record(latency_ms)
        rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        logger.info(
            "tts_complete",
            session_id=self.session_id,
            turn_id=self.turn_id,
            latency_ms=round(latency_ms, 2),
            text_length=text_length,
            audio_duration_ms=round(audio_duration_ms, 2),
            rtf=round(rtf, 3),
        )

    def record_first_audio(self) -> float:
        """Record time to first audio and return TTFA."""
        if self._turn_start == 0.0:
            return 0.0

        self._first_audio_time = time.perf_counter()
        ttfa_ms = (self._first_audio_time - self._turn_start) * 1000
        self.ttfa_stats.record(ttfa_ms)
        logger.info(
            "ttfa",
            session_id=self.session_id,
            turn_id=self.turn_id,
            ttfa_ms=round(ttfa_ms, 2),
        )
        return ttfa_ms

    def end_turn(self) -> float:
        """Mark end of turn and return E2E latency."""
        if self._turn_start == 0.0:
            return 0.0

        e2e_ms = (time.perf_counter() - self._turn_start) * 1000
        self.e2e_stats.record(e2e_ms)
        logger.info(
            "turn_complete",
            session_id=self.session_id,
            turn_id=self.turn_id,
            e2e_ms=round(e2e_ms, 2),
        )
        return e2e_ms

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all metrics."""
        return {
            "vad": self.vad_stats.to_dict(),
            "asr": self.asr_stats.to_dict(),
            "llm": self.llm_stats.to_dict(),
            "tts": self.tts_stats.to_dict(),
            "ttfa": self.ttfa_stats.to_dict(),
            "e2e": self.e2e_stats.to_dict(),
        }

    def log_summary(self) -> None:
        """Log summary of all metrics."""
        logger.info(
            "session_metrics",
            session_id=self.session_id,
            total_turns=self.turn_id,
            **self.get_summary(),
        )


class MetricsContext:
    """
    Context manager for tracking a complete pipeline execution.

    Example:
        >>> async with MetricsContext(monitor) as ctx:
        ...     ctx.record_asr(...)
        ...     ctx.record_llm(...)
        ...     ctx.record_tts(...)
    """

    def __init__(self, monitor: PipelineMonitor) -> None:
        self._monitor = monitor
        self._timer = Timer()

    async def __aenter__(self) -> MetricsContext:
        self._monitor.start_turn()
        self._timer.__enter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._timer.__exit__(*args)
        self._monitor.end_turn()

    def record_asr(
        self,
        latency_ms: float,
        audio_duration_ms: float,
        transcript_length: int,
    ) -> None:
        """Record ASR metrics."""
        self._monitor.record_asr(latency_ms, audio_duration_ms, transcript_length)

    def record_llm(
        self,
        ttft_ms: float,
        total_ms: float,
        token_count: int,
    ) -> None:
        """Record LLM metrics."""
        self._monitor.record_llm(ttft_ms, total_ms, token_count)

    def record_tts(
        self,
        latency_ms: float,
        text_length: int,
        audio_duration_ms: float,
    ) -> None:
        """Record TTS metrics."""
        self._monitor.record_tts(latency_ms, text_length, audio_duration_ms)

    def record_first_audio(self) -> float:
        """Record TTFA."""
        return self._monitor.record_first_audio()
