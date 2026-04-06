"""Pipeline Orchestrator - Coordinates all services for voice conversation."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncIterator

from voice_agent.core import (
    PipelineError,
    PipelineState,
    VADResult,
)
from voice_agent.utils import (
    AudioBuffer,
    PipelineMonitor,
    Timer,
    get_audio_duration_ms,
    get_logger,
    get_postprocessor,
)

if TYPE_CHECKING:
    from voice_agent.services.asr import ASRService
    from voice_agent.services.llm import LLMService
    from voice_agent.services.tts import TTSService
    from voice_agent.services.vad import VADService

logger = get_logger(__name__)


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
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            vad: VAD service instance (required)
            asr: ASR service instance (optional - disabled if None)
            llm: LLM service instance (optional - disabled if None)
            tts: TTS service instance (optional - disabled if None)
            session_id: Optional session ID (auto-generated if None)
        """
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._tts = tts

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._state = PipelineState.IDLE
        self._monitor = PipelineMonitor(self.session_id)

        # Audio buffer for current utterance
        self._audio_buffer = AudioBuffer(max_duration_ms=30000)

        # Conversation history
        self._history: list[dict[str, str]] = []

        # Pipeline task (for cancellation)
        self._pipeline_task: asyncio.Task[None] | None = None

        # Interrupt flag
        self._interrupted = False

        # Audio postprocessor for TTS output
        self._postprocessor = get_postprocessor()
        self._min_asr_audio_ms = max(1000, vad._config.min_speech_ms)

        # Log service status
        logger.info(
            "orchestrator_init",
            session_id=self.session_id,
            vad_enabled=vad is not None,
            asr_enabled=asr is not None,
            llm_enabled=llm is not None,
            tts_enabled=tts is not None,
        )

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

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
        use_streaming = (
            self._asr._config.streaming
            and audio_duration_ms > 2000  # Only stream for >2s audio
        )

        transcript = ""
        asr_latency_ms = 0.0

        if use_streaming:
            # Streaming ASR - get partial results for faster TTFA
            logger.debug(
                "asr_streaming_start",
                session_id=self.session_id,
                audio_duration_ms=round(audio_duration_ms, 2),
            )

            async for asr_result in self._asr.transcribe_stream(
                audio,
                chunk_duration_ms=2000,
            ):
                if not asr_result.text:
                    continue
                if asr_result.is_final:
                    transcript = asr_result.text.strip()
            if not transcript:
                logger.info("empty_streaming_transcript", session_id=self.session_id)
            asr_latency_ms = audio_duration_ms  # Approximation for streaming

        else:
            # Standard ASR - single pass
            with Timer() as asr_timer:
                asr_result = await self._asr.transcribe(
                    audio,
                    preprocess=self._asr._config.preprocess,
                )

            transcript = asr_result.text.strip()
            asr_latency_ms = asr_timer.elapsed_ms

        self._monitor.record_asr(
            asr_latency_ms,
            audio_duration_ms,
            len(transcript),
        )

        if not transcript:
            logger.info("empty_transcript", session_id=self.session_id)
            return

        yield f"TRANSCRIPT:{transcript}"
        logger.info(
            "transcript",
            session_id=self.session_id,
            text=transcript[:100],
            streaming=use_streaming,
        )

        # === LLM ===
        if self._llm is None:
            # LLM disabled - echo transcript
            logger.info("llm_skipped", session_id=self.session_id)
            if self._tts is not None:
                # Still do TTS with the transcript
                self._state = PipelineState.SPEAKING
                yield "SPEAKING"
                tts_result = await self._tts.synthesize(f"Bạn nói: {transcript}")
                output_audio = self._postprocessor.process(tts_result.audio)
                yield output_audio
            return

        # === LLM + TTS Streaming ===
        self._state = PipelineState.SPEAKING
        yield "SPEAKING"

        first_audio = True
        llm_start = asyncio.get_event_loop().time()
        ttft_ms = 0.0
        ttfa_ms = 0.0
        total_tokens = 0

        async for sentence in self._llm.stream(transcript, self._history):
            if self._interrupted:
                logger.info("pipeline_interrupted", session_id=self.session_id)
                self._interrupted = False
                break

            total_tokens += len(sentence.split())

            # Record LLM TTFT
            if first_audio:
                ttft_ms = (asyncio.get_event_loop().time() - llm_start) * 1000

            # === TTS Streaming ===
            if self._tts is None:
                # TTS disabled - just log response
                logger.info(
                    "tts_skipped",
                    session_id=self.session_id,
                    sentence=sentence[:50],
                )
                if first_audio:
                    self._monitor.record_first_audio()
                    first_audio = False
                continue

            # Stream TTS audio chunks
            tts_start = asyncio.get_event_loop().time()
            chunk_count = 0
            chunk_ms = self._tts._config.stream_chunk_ms

            async for audio_chunk in self._tts.synthesize_stream(
                sentence, chunk_duration_ms=chunk_ms
            ):
                chunk_count += 1

                # Record TTFA (Time to First Audio)
                if first_audio:
                    ttfa_ms = (asyncio.get_event_loop().time() - tts_start) * 1000
                    self._monitor.record_first_audio()
                    first_audio = False
                    logger.info(
                        "first_audio_chunk",
                        session_id=self.session_id,
                        ttft_ms=round(ttft_ms, 2),
                        ttfa_ms=round(ttfa_ms, 2),
                    )

                # Postprocess and yield audio chunk
                output_audio = self._postprocessor.process(audio_chunk)
                yield output_audio

            # Record TTS metrics for this sentence
            tts_ms = (asyncio.get_event_loop().time() - tts_start) * 1000
            self._monitor.record_tts(tts_ms, len(sentence), tts_ms)

            logger.debug(
                "sentence_complete",
                session_id=self.session_id,
                chunks=chunk_count,
                tts_ms=round(tts_ms, 2),
            )

        # Record LLM metrics
        llm_total_ms = (asyncio.get_event_loop().time() - llm_start) * 1000
        self._monitor.record_llm(ttft_ms, llm_total_ms, total_tokens)

        # Update history
        self._history.append({"role": "user", "content": transcript})
        # Note: We'd need to accumulate LLM response to add to history

        self._monitor.end_turn()

    async def interrupt(self) -> None:
        """Interrupt current pipeline execution."""
        if self._state == PipelineState.SPEAKING:
            self._interrupted = True
            self._state = PipelineState.INTERRUPTED
            logger.info("interrupt_requested", session_id=self.session_id)

    def reset(self) -> None:
        """Reset orchestrator state."""
        self._audio_buffer.clear()
        self._vad.reset()
        self._state = PipelineState.IDLE
        self._interrupted = False
        logger.info("orchestrator_reset", session_id=self.session_id)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        logger.info("history_cleared", session_id=self.session_id)

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics summary."""
        return self._monitor.get_summary()

    def log_metrics(self) -> None:
        """Log performance metrics."""
        self._monitor.log_summary()
