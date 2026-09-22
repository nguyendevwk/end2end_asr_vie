"""ASR Service using Qwen3-ASR with streaming and audio preprocessing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from voice_agent.core import ASRConfig, ASRError, SAMPLE_RATE, TranscriptionResult
from voice_agent.utils import AudioPreprocessor, Timer, get_audio_duration_ms, get_logger, pcm_to_numpy

if TYPE_CHECKING:
    from qwen_asr import Qwen3ASRModel

logger = get_logger(__name__)


class ASRService:
    """
    Automatic Speech Recognition using Qwen3-ASR.

    Features:
    - Two backends: transformers (less VRAM) or vLLM (faster)
    - Audio preprocessing for improved accuracy
    - Streaming with interim/final results
    - Context overlapping for better accuracy
    - Error recovery and timeout handling

    Streaming Modes:
    - transcribe(): Single-shot transcription
    - transcribe_stream(): Process complete audio in chunks
    - transcribe_realtime(): Live streaming with interim results

    Attributes:
        config: ASR configuration
        model_name: Name of the ASR model
    """

    # Streaming parameters (tuned for Vietnamese)
    INTERIM_RESULTS_ENABLED = True      # Emit interim results
    SINGLE_UTTERANCE_MODE = False       # Stop on silence
    OVERLAP_SAMPLES = 1600              # 100ms overlap at 16kHz
    MIN_STABILITY_THRESHOLD = 0.8       # Stability threshold for interim results
    MAX_ALTERNATIVES = 1                # Number of alternatives

    def __init__(self, config: ASRConfig | None = None) -> None:
        """
        Initialize ASR service.

        Args:
            config: ASR configuration (uses defaults if None)
        """
        self._config = config or ASRConfig()
        self._model: Qwen3ASRModel | None = None
        self._started = False
        self._preprocessor = AudioPreprocessor()

    async def start(self) -> None:
        """Load ASR model with specified backend."""
        if self._started:
            return

        backend = self._config.backend.lower()
        logger.info(
            "loading_asr_model",
            model=self._config.model_name,
            backend=backend,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
        )

        try:
            from qwen_asr import Qwen3ASRModel

            if backend == "vllm":
                # vLLM backend: Higher throughput, needs more VRAM
                self._model = Qwen3ASRModel.LLM(
                    model=self._config.model_name,
                    gpu_memory_utilization=self._config.gpu_memory_utilization,
                    max_inference_batch_size=8,
                    max_new_tokens=self._config.max_new_tokens,
                )
            else:
                # Transformers backend: Lower VRAM, suitable for small GPUs
                import torch

                self._model = Qwen3ASRModel.from_pretrained(
                    self._config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            self._started = True
            logger.info(
                "asr_model_loaded",
                model=self._config.model_name,
                backend=backend,
            )

        except ImportError as e:
            logger.error("qwen_asr_not_installed", error=str(e))
            raise ASRError(
                "qwen-asr package not installed. Run: pip install qwen-asr"
            ) from e
        except Exception as e:
            logger.error("asr_load_failed", error=str(e))
            raise ASRError(f"Failed to load ASR model: {e}") from e

    async def stop(self) -> None:
        """Cleanup ASR resources."""
        if not self._started:
            return

        self._model = None
        self._started = False
        logger.info("asr_stopped")

    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        preprocess: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: PCM S16LE audio bytes
            language: Language hint (None for auto-detect)
                      Options: "Vietnamese", "English", "Chinese", etc.
            preprocess: Apply audio preprocessing (default True)

        Returns:
            TranscriptionResult with text, language, and latency

        Raises:
            ASRError: If transcription fails
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        if len(audio) == 0:
            return TranscriptionResult(text="", language="", latency_ms=0.0)

        timer = Timer()
        try:
            with timer:
                # Convert PCM to numpy
                audio_np = pcm_to_numpy(audio)

                # Preprocess for better accuracy
                if preprocess:
                    audio_np = self._preprocessor.process_numpy(audio_np)

                if len(audio_np) == 0:
                    return TranscriptionResult(text="", language="", latency_ms=0.0)

                # Transcribe (blocking HF/vLLM call -> worker thread
                # so the FastAPI event loop stays responsive)
                lang = language or self._config.language
                model = self._model
                results = await asyncio.to_thread(
                    model.transcribe,
                    audio=(audio_np, SAMPLE_RATE),
                    language=lang,
                )

            result = results[0] if results else None
            text = result.text if result else ""
            detected_lang = result.language if result else ""

            # Clean up text
            text = self._clean_text(text)

            audio_duration_ms = get_audio_duration_ms(audio)

            logger.info(
                "asr_transcribe",
                latency_ms=round(timer.elapsed_ms, 2),
                audio_duration_ms=round(audio_duration_ms, 2),
                text_length=len(text),
                language=detected_lang,
                rtf=round(timer.elapsed_ms / audio_duration_ms, 3) if audio_duration_ms > 0 else 0,
            )

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                latency_ms=timer.elapsed_ms,
            )

        except Exception as e:
            logger.error("asr_transcribe_failed", error=str(e))
            raise ASRError(f"Transcription failed: {e}") from e

    async def transcribe_stream(
        self,
        audio: bytes,
        language: str | None = None,
        chunk_duration_ms: int = 2000,
        enable_interim: bool = True,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Streaming transcription with interim and final results.

        Process complete audio in chunks, yielding both interim (unstable) and
        final (stable) results. Uses context overlapping to avoid word boundary issues.

        Args:
            audio: PCM S16LE audio bytes
            language: Language hint (None for auto-detect)
            chunk_duration_ms: Duration of each chunk in ms (default 2000ms)
            enable_interim: Emit interim results for lower perceived latency

        Yields:
            TranscriptionResult with is_final=False for interim, True for final

        Example:
            async for result in asr.transcribe_stream(audio):
                if result.is_final:
                    final_transcript += result.text
                else:
                    show_interim(result.text)  # Update UI with interim
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        if len(audio) == 0:
            return

        # Convert to numpy
        audio_np = pcm_to_numpy(audio)

        # Preprocess entire audio for consistency
        audio_np = self._preprocessor.process_numpy(audio_np)

        if len(audio_np) == 0:
            return

        # Calculate chunk parameters
        chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)
        overlap_samples = self.OVERLAP_SAMPLES  # 100ms overlap
        total_samples = len(audio_np)

        # Calculate effective step (chunk - overlap)
        step_samples = chunk_samples - overlap_samples
        n_chunks = max(1, (total_samples - overlap_samples + step_samples - 1) // step_samples)

        logger.debug(
            "asr_stream_start",
            total_samples=total_samples,
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
            n_chunks=n_chunks,
        )

        accumulated_text = ""
        prev_text = ""
        audio_offset_ms = 0.0

        for i in range(n_chunks):
            # Calculate chunk boundaries with overlap
            start = max(0, i * step_samples - (overlap_samples // 2 if i > 0 else 0))
            end = min(total_samples, start + chunk_samples)
            chunk = audio_np[start:end]

            is_last_chunk = (i == n_chunks - 1)
            audio_offset_ms = start / SAMPLE_RATE * 1000

            # Skip very short chunks (< 100ms)
            if len(chunk) < SAMPLE_RATE * 0.1:
                continue

            timer = Timer()
            try:
                with timer:
                    lang = language or self._config.language
                    model = self._model
                    results = await asyncio.to_thread(
                        model.transcribe,
                        audio=(chunk, SAMPLE_RATE),
                        language=lang,
                    )

                result = results[0] if results else None
                text = result.text if result else ""
                text = self._clean_text(text)

                if not text:
                    continue

                # Calculate stability based on text similarity with previous
                stability = self._calculate_stability(prev_text, text)

                # Determine if this should be final result
                # Final if: last chunk OR high stability
                is_final = is_last_chunk or stability > self.MIN_STABILITY_THRESHOLD

                # Emit interim result first (if enabled and not final)
                if enable_interim and not is_final:
                    yield TranscriptionResult(
                        text=text,
                        language=result.language if result else "vi",
                        latency_ms=timer.elapsed_ms,
                        is_final=False,
                        stability=stability,
                        confidence=0.0,  # Model doesn't provide confidence
                        result_end_offset_ms=audio_offset_ms + len(chunk) / SAMPLE_RATE * 1000,
                    )
                    logger.debug(
                        "asr_interim_result",
                        chunk=i + 1,
                        text=text[:50],
                        stability=round(stability, 2),
                    )

                # Emit final result
                if is_final:
                    # For final, use accumulated context
                    final_text = self._merge_overlapping_text(accumulated_text, text)
                    accumulated_text = final_text

                    yield TranscriptionResult(
                        text=final_text,
                        language=result.language if result else "vi",
                        latency_ms=timer.elapsed_ms,
                        is_final=True,
                        stability=1.0,
                        confidence=0.0,
                        result_end_offset_ms=audio_offset_ms + len(chunk) / SAMPLE_RATE * 1000,
                    )
                    logger.debug(
                        "asr_final_result",
                        chunk=i + 1,
                        n_chunks=n_chunks,
                        text=text[:50],
                        latency_ms=round(timer.elapsed_ms, 2),
                    )

                prev_text = text

            except Exception as e:
                logger.error("asr_stream_chunk_failed", chunk=i, error=str(e))
                # Continue processing - don't break the stream
                continue

        logger.info(
            "asr_stream_complete",
            n_chunks=n_chunks,
            total_text_length=len(accumulated_text),
        )

    def _calculate_stability(self, prev_text: str, current_text: str) -> float:
        """
        Calculate stability score (0-1) based on text similarity.

        Higher stability means the transcript is less likely to change.
        This is uses this to indicate how "stable" interim results are.
        """
        if not prev_text:
            return 0.5  # First result, medium stability

        if not current_text:
            return 0.0

        # Simple overlap-based stability
        prev_words = prev_text.lower().split()
        curr_words = current_text.lower().split()

        if not prev_words or not curr_words:
            return 0.5

        # Count matching words from the start
        matching = 0
        for p, c in zip(prev_words, curr_words):
            if p == c:
                matching += 1
            else:
                break

        # Stability = ratio of matching prefix
        stability = matching / max(len(prev_words), len(curr_words))

        # Boost stability if texts are very similar
        if prev_text.strip() == current_text.strip():
            stability = 1.0

        return min(1.0, stability)

    def _merge_overlapping_text(self, accumulated: str, new_text: str) -> str:
        """
        Merge overlapping text segments intelligently.

        Handles the overlap between chunks to avoid duplicate words.
        """
        if not accumulated:
            return new_text

        accumulated_words = accumulated.split()
        new_words = new_text.split()

        if not accumulated_words or not new_words:
            return accumulated + " " + new_text

        # Find overlap point
        # Look for common suffix in accumulated and prefix in new_text
        max_overlap = min(5, len(accumulated_words), len(new_words))

        for overlap_len in range(max_overlap, 0, -1):
            if accumulated_words[-overlap_len:] == new_words[:overlap_len]:
                # Found overlap, merge without duplicates
                return " ".join(accumulated_words + new_words[overlap_len:])

        # No overlap found, just concatenate
        return accumulated + " " + new_text

    async def transcribe_realtime(
        self,
        audio_stream: AsyncIterator[bytes],
        min_chunk_ms: int = 500,
        max_chunk_ms: int = 3000,
        silence_timeout_ms: int = 1000,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Real-time streaming transcription.

        Processes live audio stream with:
        - Interim results for immediate feedback
        - Final results when speech segment ends
        - Automatic silence detection and segmentation
        - Buffer management and error recovery

        This is the recommended method for live microphone input.

        Args:
            audio_stream: Async generator yielding PCM S16LE audio chunks
            min_chunk_ms: Minimum audio to accumulate before processing (500ms)
            max_chunk_ms: Maximum audio before forced processing (3000ms)
            silence_timeout_ms: Silence duration to trigger final result (1000ms)

        Yields:
            TranscriptionResult with is_final flag:
            - is_final=False: Interim result (may change)
            - is_final=True: Final result (stable, won't change)

        Example:
            async for result in asr.transcribe_realtime(mic_stream()):
                if result.is_final:
                    process_final(result.text)
                else:
                    update_display(result.text)  # Show interim
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        # Buffer for accumulating audio
        buffer = bytearray()
        min_chunk_bytes = int(SAMPLE_RATE * 2 * min_chunk_ms / 1000)
        max_chunk_bytes = int(SAMPLE_RATE * 2 * max_chunk_ms / 1000)

        # State tracking
        prev_text = ""
        total_processed_ms = 0.0
        chunks_processed = 0
        last_speech_time = 0.0

        logger.info(
            "asr_realtime_start",
            min_chunk_ms=min_chunk_ms,
            max_chunk_ms=max_chunk_ms,
            silence_timeout_ms=silence_timeout_ms,
        )

        try:
            async for audio_chunk in audio_stream:
                buffer.extend(audio_chunk)
                current_time_ms = len(buffer) / (SAMPLE_RATE * 2) * 1000

                # Check if we should process
                should_process = False
                force_final = False

                if len(buffer) >= max_chunk_bytes:
                    # Max buffer reached - force processing
                    should_process = True
                    force_final = True
                    logger.debug("asr_max_buffer_reached", buffer_ms=current_time_ms)
                elif len(buffer) >= min_chunk_bytes:
                    # Min buffer reached - process for interim
                    should_process = True

                if not should_process:
                    continue

                # Process current buffer
                chunk_to_process = bytes(buffer)
                chunk_duration_ms = len(chunk_to_process) / (SAMPLE_RATE * 2) * 1000

                timer = Timer()
                try:
                    with timer:
                        result = await self.transcribe(
                            chunk_to_process,
                            preprocess=self._config.preprocess,
                        )

                    text = result.text.strip()
                    chunks_processed += 1

                    if text:
                        last_speech_time = current_time_ms

                        # Calculate stability
                        stability = self._calculate_stability(prev_text, text)

                        # Determine if final
                        # Final if: forced OR high stability with significant content
                        is_final = force_final or (
                            stability > self.MIN_STABILITY_THRESHOLD
                            and len(text) > 10
                        )

                        yield TranscriptionResult(
                            text=text,
                            language=result.language or "vi",
                            latency_ms=timer.elapsed_ms,
                            is_final=is_final,
                            stability=stability,
                            confidence=0.0,
                            result_end_offset_ms=total_processed_ms + chunk_duration_ms,
                        )

                        logger.debug(
                            "asr_realtime_result",
                            is_final=is_final,
                            stability=round(stability, 2),
                            text=text[:50],
                            latency_ms=round(timer.elapsed_ms, 2),
                        )

                        prev_text = text

                        # If final, clear buffer completely
                        if is_final:
                            buffer.clear()
                            prev_text = ""
                        else:
                            # Keep overlap for context
                            overlap_bytes = min(len(buffer), self.OVERLAP_SAMPLES * 2)
                            buffer = bytearray(buffer[-overlap_bytes:])

                    elif force_final and prev_text:
                        # No new text but forced - emit final for previous
                        yield TranscriptionResult(
                            text=prev_text,
                            language="vi",
                            latency_ms=timer.elapsed_ms,
                            is_final=True,
                            stability=1.0,
                            confidence=0.0,
                            result_end_offset_ms=total_processed_ms + chunk_duration_ms,
                        )
                        buffer.clear()
                        prev_text = ""

                    total_processed_ms += chunk_duration_ms

                except Exception as e:
                    logger.error("asr_realtime_chunk_failed", error=str(e))
                    # Don't clear buffer on error - retry with more audio
                    continue

            # Process remaining buffer as final
            if len(buffer) > SAMPLE_RATE * 2 * 0.1:  # > 100ms
                try:
                    result = await self.transcribe(bytes(buffer), preprocess=self._config.preprocess)
                    if result.text.strip():
                        yield TranscriptionResult(
                            text=result.text.strip(),
                            language=result.language or "vi",
                            latency_ms=result.latency_ms,
                            is_final=True,
                            stability=1.0,
                            confidence=0.0,
                            result_end_offset_ms=total_processed_ms + len(buffer) / (SAMPLE_RATE * 2) * 1000,
                        )
                except Exception as e:
                    logger.error("asr_realtime_final_failed", error=str(e))

        finally:
            logger.info(
                "asr_realtime_complete",
                chunks_processed=chunks_processed,
                total_processed_ms=round(total_processed_ms, 2),
            )

    # Alias for backwards compatibility
    transcribe_progressive = transcribe_realtime

    async def transcribe_numpy(
        self,
        audio_np: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        preprocess: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe numpy audio array to text.

        Args:
            audio_np: Audio as float32 numpy array
            sample_rate: Audio sample rate
            language: Language hint (None for auto-detect)
            preprocess: Apply audio preprocessing

        Returns:
            TranscriptionResult with text, language, and latency

        Raises:
            ASRError: If transcription fails
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        if len(audio_np) == 0:
            return TranscriptionResult(text="", language="", latency_ms=0.0)

        timer = Timer()
        try:
            with timer:
                # Preprocess
                if preprocess:
                    audio_np = self._preprocessor.process_numpy(audio_np)

                if len(audio_np) == 0:
                    return TranscriptionResult(text="", language="", latency_ms=0.0)

                # Transcribe directly from numpy (blocking -> worker thread)
                lang = language or self._config.language
                model = self._model
                results = await asyncio.to_thread(
                    model.transcribe,
                    audio=(audio_np, sample_rate),
                    language=lang,
                )

            result = results[0] if results else None
            text = result.text if result else ""
            text = self._clean_text(text)
            detected_lang = result.language if result else ""

            audio_duration_ms = len(audio_np) / sample_rate * 1000

            logger.info(
                "asr_transcribe",
                latency_ms=round(timer.elapsed_ms, 2),
                audio_duration_ms=round(audio_duration_ms, 2),
                text_length=len(text),
                language=detected_lang,
            )

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                latency_ms=timer.elapsed_ms,
            )

        except Exception as e:
            logger.error("asr_transcribe_failed", error=str(e))
            raise ASRError(f"Transcription failed: {e}") from e

    def _clean_text(self, text: str) -> str:
        """Clean up transcribed text."""
        if not text:
            return ""

        # Strip whitespace
        text = text.strip()

        # Remove repeated spaces (O(n) instead of O(n^2))
        text = " ".join(text.split())

        # Remove common ASR artifacts
        artifacts = ["<|notimestamps|>", "<unk>", "[UNK]", "<s>", "</s>"]
        for artifact in artifacts:
            text = text.replace(artifact, "")

        return text.strip()

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._config.model_name

    @property
    def streaming(self) -> bool:
        """Whether chunked streaming transcription is enabled."""
        return self._config.streaming

    @property
    def preprocess_enabled(self) -> bool:
        """Whether audio preprocessing is enabled."""
        return self._config.preprocess
