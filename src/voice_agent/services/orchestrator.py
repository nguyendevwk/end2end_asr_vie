"""Pipeline Orchestrator - Coordinates all services for voice conversation."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from voice_agent.core import (
    PipelineError,
    PipelineState,
)
from voice_agent.services.text_tasks import LLMTask, PassthroughTask, TextTask
from voice_agent.utils import (
    AudioBuffer,
    PipelineMonitor,
    Timer,
    get_audio_duration_ms,
    get_logger,
    get_postprocessor,
    get_registry,
    retry_async,
)
from voice_agent.utils.session_store import SessionSnapshot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from voice_agent.services.asr import ASRService
    from voice_agent.services.llm import LLMService
    from voice_agent.services.tts import TTSService
    from voice_agent.services.vad import VADService

logger = get_logger(__name__)


@dataclass
class PipelineRuntime:
    """Production knobs: timeouts, retries, GPU admission control."""

    asr_timeout_s: float = 15.0
    tts_timeout_s: float = 20.0
    llm_timeout_s: float = 30.0
    llm_first_token_timeout_s: float = 10.0
    max_retries: int = 1  # extra attempts after the first try
    # Shared semaphore bounding concurrent GPU inference (None = unbounded).
    infer_semaphore: asyncio.Semaphore | None = None
    # Streaming ASR threshold: audio longer than this uses chunked streaming.
    streaming_threshold_ms: float = 2000.0
    # Prefix for passthrough task echo (configurable per deployment).
    passthrough_prefix: str = "You said: "


class Orchestrator:
    """
    Pipeline orchestrator for voice conversation.

    Coordinates VAD → ASR → LLM → TTS pipeline with:
    - Real-time audio processing
    - Streaming responses
    - Interrupt handling
    - Performance monitoring

    Services can be None for testing - disabled services will be skipped.

    Attributes:
        session_id: Unique session identifier
        state: Current pipeline state
    """

    def __init__(
        self,
        vad: VADService,
        asr: ASRService | None = None,
        llm: LLMService | None = None,
        tts: TTSService | None = None,
        session_id: str | None = None,
        text_task: TextTask | None = None,
        runtime: PipelineRuntime | None = None,
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            vad: VAD service instance (required; per-session state is
                derived via ``vad.session()`` when available).
            asr: ASR service instance (optional - disabled if None)
            llm: LLM service instance (optional - wrapped as the text task
                when ``text_task`` is not given; kept for backward compat)
            tts: TTS service instance (optional - disabled if None)
            session_id: Optional session ID (auto-generated if None)
            text_task: Pluggable transcript->response task. Defaults to the
                LLM task when ``llm`` is given, else direct ASR->TTS echo.
            runtime: Timeouts/retries/GPU admission control.
        """
        # Per-connection VAD state; falls back to the shared instance for
        # test doubles without a session() factory.
        session_factory = getattr(vad, "session", None)
        self._vad = session_factory() if callable(session_factory) else vad
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._runtime = runtime or PipelineRuntime()
        if text_task is not None:
            self._task = text_task
        elif llm is not None:
            self._task = LLMTask(llm=llm)
        else:
            self._task = PassthroughTask(prefix=self._runtime.passthrough_prefix)

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._state = PipelineState.IDLE
        self._monitor = PipelineMonitor(self.session_id)
        self._metrics = get_registry()

        # Audio buffer for current utterance
        self._audio_buffer = AudioBuffer(max_duration_ms=30000)

        # Conversation history
        self._history: list[dict[str, str]] = []
        self._last_transcript = ""

        # Pipeline task (for cancellation)
        self._pipeline_task: asyncio.Task[None] | None = None

        # Interrupt flag
        self._interrupted = False

        # Audio postprocessor for TTS output
        self._postprocessor = get_postprocessor()
        # Short commands ("xin chào" ~600ms) must pass; VAD config is source
        # of truth, 500ms floor guards against noise blips.
        self._min_asr_audio_ms = max(500, getattr(self._vad, "min_speech_ms", 500))
        # Cap history to avoid unbounded growth (20 msgs = ~10 turns)
        self._max_history = 20

        # Log service status
        logger.info(
            "orchestrator_init",
            session_id=self.session_id,
            vad_enabled=vad is not None,
            asr_enabled=asr is not None,
            llm_enabled=llm is not None,
            tts_enabled=tts is not None,
            text_task=self._task.name,
        )

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

    def _finish_turn(self) -> float:
        """Close the turn in both the session monitor and global registry."""
        e2e_ms = self._monitor.end_turn()
        if e2e_ms:
            self._metrics.observe("e2e_ms", e2e_ms)
        return e2e_ms

    async def process_audio(
        self,
        audio_chunk: bytes,
    ) -> AsyncIterator[bytes | str]:
        """
        Process incoming audio chunk.

        This is the main entry point for audio processing.
        Yields audio responses or text events.

        Args:
            audio_chunk: PCM S16LE audio chunk

        Yields:
            bytes: TTS audio response
            str: Event messages (e.g., "TRANSCRIPT:...", "SPEAKING", "LISTENING")

        Raises:
            PipelineError: If processing fails
        """
        try:
            # Run VAD on raw audio
            with Timer() as vad_timer:
                vad_result = await self._vad.detect(audio_chunk)
            self._monitor.record_vad(vad_timer.elapsed_ms, vad_result.is_speech)

            # Handle VAD events
            if vad_result.event == "start":
                # Speech started
                self._state = PipelineState.LISTENING
                self._audio_buffer.clear()
                self._monitor.start_turn()
                yield "LISTENING"
                logger.info("speech_started", session_id=self.session_id)

            if vad_result.is_speech:
                # Accumulate raw audio (will be preprocessed by ASR later)
                self._audio_buffer.add(audio_chunk)

            if vad_result.event == "end" and self._audio_buffer:
                # Speech ended - validate utterance duration first
                # Get buffered audio
                dropped = self._audio_buffer.dropped_chunks
                if dropped:
                    self._metrics.inc("audio_chunks_dropped", dropped)
                    logger.warning(
                        "audio_buffer_overflow",
                        session_id=self.session_id,
                        dropped_chunks=dropped,
                    )
                audio_data = self._audio_buffer.get_all()
                audio_duration_ms = get_audio_duration_ms(audio_data)

                if audio_duration_ms < self._min_asr_audio_ms:
                    logger.info(
                        "speech_too_short_skip_asr",
                        session_id=self.session_id,
                        audio_duration_ms=round(audio_duration_ms, 2),
                        min_asr_audio_ms=self._min_asr_audio_ms,
                    )
                    self._audio_buffer.clear()
                    self._vad.reset()
                    self._state = PipelineState.IDLE
                    yield "IDLE"
                    return

                self._state = PipelineState.PROCESSING
                yield "PROCESSING"

                logger.info(
                    "speech_ended",
                    session_id=self.session_id,
                    audio_duration_ms=round(audio_duration_ms, 2),
                )

                # Latency metrics start at speech end, not speech start
                self._monitor.mark_processing_start()
                self._metrics.inc("turns_started")

                # Process pipeline and yield results
                async for result in self._run_pipeline(audio_data):
                    yield result

                # Reset for next utterance
                self._audio_buffer.clear()
                self._vad.reset()
                self._state = PipelineState.IDLE
                yield "IDLE"

        except Exception as e:
            logger.error("process_audio_failed", error=str(e))
            self._state = PipelineState.IDLE
            raise PipelineError(f"Audio processing failed: {e}") from e

    async def _run_pipeline(
        self,
        audio: bytes,
    ) -> AsyncIterator[bytes | str]:
        """
        Run ASR → LLM → TTS pipeline.

        Supports streaming ASR for faster TTFA.
        Skips disabled services gracefully.

        Args:
            audio: Complete utterance audio

        Yields:
            Pipeline results (transcripts, audio responses)
        """
        audio_duration_ms = get_audio_duration_ms(audio)

        # === ASR ===
        if self._asr is None:
            # ASR disabled - just log and skip
            logger.info(
                "asr_skipped",
                session_id=self.session_id,
                audio_duration_ms=round(audio_duration_ms, 2),
            )
            yield "TRANSCRIPT:[ASR disabled - audio received]"
            return

        # Use streaming ASR if enabled and audio is long enough
        use_streaming = self._asr.streaming and audio_duration_ms > self._runtime.streaming_threshold_ms

        transcript = ""
        asr_latency_ms = 0.0
        rt = self._runtime

        async def _single_transcribe() -> str:
            async def _call() -> str:
                if rt.infer_semaphore is None:
                    res = await self._asr.transcribe(  # type: ignore[union-attr]
                        audio,
                        preprocess=self._asr.preprocess_enabled,  # type: ignore[union-attr]
                    )
                else:
                    async with rt.infer_semaphore:
                        res = await self._asr.transcribe(  # type: ignore[union-attr]
                            audio,
                            preprocess=self._asr.preprocess_enabled,  # type: ignore[union-attr]
                        )
                return res.text.strip()

            return await retry_async(
                _call,
                attempts=rt.max_retries + 1,
                timeout_s=rt.asr_timeout_s,
            )

        if use_streaming:
            # Streaming ASR - accumulate ALL final chunks.
            logger.debug(
                "asr_streaming_start",
                session_id=self.session_id,
                audio_duration_ms=round(audio_duration_ms, 2),
            )

            stream_start = asyncio.get_running_loop().time()
            final_parts: list[str] = []
            try:
                if rt.infer_semaphore is None:
                    async for asr_result in self._asr.transcribe_stream(
                        audio,
                        chunk_duration_ms=2000,
                    ):
                        if not asr_result.text:
                            continue
                        if asr_result.is_final:
                            final_parts.append(asr_result.text.strip())
                else:
                    async with rt.infer_semaphore:
                        async for asr_result in self._asr.transcribe_stream(
                            audio,
                            chunk_duration_ms=2000,
                        ):
                            if not asr_result.text:
                                continue
                            if asr_result.is_final:
                                final_parts.append(asr_result.text.strip())
                transcript = " ".join(final_parts).strip()
            except Exception as e:
                # Degrade to single-shot instead of dropping the turn.
                logger.warning("asr_stream_fallback", session_id=self.session_id, error=str(e))
                self._metrics.inc("asr_stream_fallbacks")
                try:
                    transcript = await _single_transcribe()
                except Exception as e2:
                    logger.error("asr_failed", session_id=self.session_id, error=str(e2))
            if not transcript:
                logger.info("empty_streaming_transcript", session_id=self.session_id)
            asr_latency_ms = (asyncio.get_running_loop().time() - stream_start) * 1000

        else:
            # Standard ASR - single pass with retry
            try:
                with Timer() as asr_timer:
                    transcript = await _single_transcribe()
                asr_latency_ms = asr_timer.elapsed_ms
            except Exception as e:
                logger.error("asr_failed", session_id=self.session_id, error=str(e))
                self._metrics.inc("errors_asr")
                yield f"ERROR:ASR failed: {e}"
                self._finish_turn()
                return

        self._monitor.record_asr(
            asr_latency_ms,
            audio_duration_ms,
            len(transcript),
        )
        self._metrics.observe("asr_ms", asr_latency_ms)

        if not transcript:
            logger.info("empty_transcript", session_id=self.session_id)
            self._finish_turn()
            return

        yield f"TRANSCRIPT:{transcript}"
        self._last_transcript = transcript
        logger.info(
            "transcript",
            session_id=self.session_id,
            text=transcript[:100],
            streaming=use_streaming,
        )

        # === TEXT TASK (default ASR -> TTS; LLM is pluggable middleware) ===
        self._state = PipelineState.SPEAKING
        yield "SPEAKING"

        first_audio = True
        llm_start = asyncio.get_running_loop().time()
        ttft_ms = 0.0
        ttfa_ms = 0.0
        total_tokens = 0
        response_parts: list[str] = []

        try:
            task_stream = self._task.stream(transcript, self._history)
            # First-chunk timeout so a hung task fails fast instead of
            # stalling the turn until the global timeout.
            try:
                first = await asyncio.wait_for(
                    task_stream.__anext__(), rt.llm_first_token_timeout_s
                )
            except StopAsyncIteration:
                first = None
        except asyncio.CancelledError:
            logger.info("pipeline_cancelled", session_id=self.session_id)
            return
        except Exception as e:
            logger.error("task_failed", session_id=self.session_id, error=str(e))
            self._metrics.inc("errors_task")
            yield f"ERROR:Text task failed: {e}"
            self._finish_turn()
            return

        async def _task_sentences() -> AsyncIterator[str]:
            if first is not None:
                yield first
            async for sentence in task_stream:
                yield sentence

        try:
            _task_deadline = asyncio.get_running_loop().time() + rt.llm_timeout_s
            async for sentence in _task_sentences():
                if asyncio.get_running_loop().time() > _task_deadline:
                    logger.warning("task_stream_timeout", session_id=self.session_id)
                    self._metrics.inc("errors_task_timeout")
                    yield "ERROR:Text task timed out"
                    break
                if self._interrupted:
                    logger.info("pipeline_interrupted", session_id=self.session_id)
                    self._interrupted = False
                    break

                total_tokens += len(sentence.split())
                response_parts.append(sentence)

                # Record LLM TTFT
                if first_audio:
                    ttft_ms = (asyncio.get_running_loop().time() - llm_start) * 1000

                # Always emit text so clients degrade to captions when TTS fails.
                yield f"RESPONSE_TEXT:{sentence}"

                # === TTS ===
                if self._tts is None:
                    logger.info(
                        "tts_skipped",
                        session_id=self.session_id,
                        sentence=sentence[:50],
                    )
                    if first_audio:
                        self._monitor.record_first_audio()
                        first_audio = False
                    continue

                # Stream TTS audio chunks with per-sentence timeout
                tts_start = asyncio.get_running_loop().time()
                chunk_count = 0
                audio_bytes_total = 0
                chunk_ms = self._tts.stream_chunk_ms

                try:
                    tts_stream = self._tts.synthesize_stream(
                        sentence, chunk_duration_ms=chunk_ms
                    )
                    while True:
                        try:
                            if rt.infer_semaphore is None:
                                audio_chunk = await asyncio.wait_for(
                                    tts_stream.__anext__(), rt.tts_timeout_s
                                )
                            else:
                                async with rt.infer_semaphore:
                                    audio_chunk = await asyncio.wait_for(
                                        tts_stream.__anext__(), rt.tts_timeout_s
                                    )
                        except StopAsyncIteration:
                            break
                        chunk_count += 1
                        audio_bytes_total += len(audio_chunk)

                        if first_audio:
                            ttfa_ms = (asyncio.get_running_loop().time() - tts_start) * 1000
                            self._monitor.record_first_audio()
                            first_audio = False
                            logger.info(
                                "first_audio_chunk",
                                session_id=self.session_id,
                                ttft_ms=round(ttft_ms, 2),
                                ttfa_ms=round(ttfa_ms, 2),
                            )

                        output_audio = self._postprocessor.process(audio_chunk)
                        yield output_audio
                except (TimeoutError, Exception) as e:
                    # Skip this sentence's audio, keep the turn alive.
                    logger.warning(
                        "tts_sentence_failed", session_id=self.session_id, error=str(e)
                    )
                    self._metrics.inc("errors_tts")
                    yield f"ERROR:TTS failed for a sentence: {e}"
                    continue

                tts_ms = (asyncio.get_running_loop().time() - tts_start) * 1000
                # PCM S16LE: 2 bytes per sample at SAMPLE_RATE
                sentence_audio_ms = (audio_bytes_total / (SAMPLE_RATE * 2)) * 1000
                self._monitor.record_tts(tts_ms, len(sentence), sentence_audio_ms)
                self._metrics.observe("tts_ms", tts_ms)

                logger.debug(
                    "sentence_complete",
                    session_id=self.session_id,
                    chunks=chunk_count,
                    tts_ms=round(tts_ms, 2),
                )
        except (TimeoutError, Exception) as e:
            logger.error("task_stream_failed", session_id=self.session_id, error=str(e))
            self._metrics.inc("errors_task")
            yield f"ERROR:Text task failed: {e}"

        # Record task metrics
        llm_total_ms = (asyncio.get_running_loop().time() - llm_start) * 1000
        self._monitor.record_llm(ttft_ms, llm_total_ms, total_tokens)
        self._metrics.observe("task_ms", llm_total_ms)
        self._metrics.inc("turns_completed")

        # Update history (both sides, capped to avoid unbounded growth)
        self._history.append({"role": "user", "content": transcript})
        full_response = " ".join(response_parts).strip()
        if full_response:
            self._history.append({"role": "assistant", "content": full_response})
        # Trim to max_history while preserving user/assistant pair coherence
        if len(self._history) > self._max_history:
            # Ensure we always keep an even number of messages (pairs)
            excess = len(self._history) - self._max_history
            # Round up to even to avoid breaking a pair
            trim = excess + (excess % 2)
            self._history = self._history[trim:]

        self._finish_turn()

    async def interrupt(self) -> None:
        """Interrupt current pipeline execution."""
        if self._state == PipelineState.SPEAKING:
            self._interrupted = True
            self._state = PipelineState.INTERRUPTED
            if self._pipeline_task is not None and not self._pipeline_task.done():
                self._pipeline_task.cancel()
            logger.info("interrupt_requested", session_id=self.session_id)

    def reset(self) -> None:
        """Reset orchestrator state."""
        if self._pipeline_task is not None and not self._pipeline_task.done():
            self._pipeline_task.cancel()
        self._audio_buffer.clear()
        self._vad.reset()
        self._state = PipelineState.IDLE
        self._interrupted = False
        logger.info("orchestrator_reset", session_id=self.session_id)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        logger.info("history_cleared", session_id=self.session_id)

    def set_task(self, task: TextTask) -> None:
        """Swap the text task at runtime (per-deployment customization)."""
        self._task = task
        logger.info("task_switched", session_id=self.session_id, task=task.name)

    def snapshot(self) -> SessionSnapshot:
        """Capture resumable session state (for reconnect recovery)."""
        return SessionSnapshot(
            session_id=self.session_id,
            history=list(self._history),
            turn_id=self._monitor.turn_id,
            last_transcript=getattr(self, "_last_transcript", ""),
            last_state=self._state.name,
        )

    def restore(self, snapshot: SessionSnapshot) -> None:
        """Restore session state after a reconnect."""
        self._history = list(snapshot.history)
        self._monitor.turn_id = snapshot.turn_id
        self.reset()
        logger.info(
            "session_restored",
            session_id=self.session_id,
            turns=snapshot.turn_id,
        )

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics summary."""
        return self._monitor.get_summary()

    def log_metrics(self) -> None:
        """Log performance metrics."""
        self._monitor.log_summary()
