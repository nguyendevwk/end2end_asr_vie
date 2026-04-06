"""ASR Service using Qwen3-ASR with streaming and audio preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

from voice_agent.core import ASRConfig, ASRError, SAMPLE_RATE, TranscriptionResult
from voice_agent.utils import Timer, get_audio_duration_ms, get_logger, pcm_to_numpy

if TYPE_CHECKING:
    from qwen_asr import Qwen3ASRModel

logger = get_logger(__name__)


class ASRPreprocessor:
    """
    Audio preprocessor optimized for ASR.

    Applies normalization and cleanup to improve transcription accuracy.
    """

    def __init__(self, target_sample_rate: int = SAMPLE_RATE) -> None:
        self._target_sr = target_sample_rate

    def process(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for ASR.

        Steps:
        1. Remove DC offset
        2. Trim silence at start/end
        3. Normalize to [-1, 1] range
        4. Apply pre-emphasis filter (boost high frequencies for speech clarity)

        Args:
            audio_np: Input audio as float32 numpy array

        Returns:
            Preprocessed audio
        """
        if len(audio_np) == 0:
            return audio_np

        # 1. Remove DC offset
        audio_np = audio_np - np.mean(audio_np)

        # 2. Trim silence (simple energy-based)
        audio_np = self._trim_silence(audio_np)

        if len(audio_np) == 0:
            return audio_np

        # 3. Normalize to peak amplitude
        max_amp = np.max(np.abs(audio_np))
        if max_amp > 0:
            audio_np = audio_np / max_amp * 0.95  # Leave headroom

        # 4. Pre-emphasis filter (improves speech recognition)
        audio_np = self._pre_emphasis(audio_np, coef=0.97)

        return audio_np.astype(np.float32)

    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        frame_length: int = 512,
    ) -> np.ndarray:
        """Trim leading/trailing silence based on energy threshold."""
        if len(audio) < frame_length:
            return audio

        # Calculate frame energies
        threshold = 10 ** (threshold_db / 20)
        n_frames = len(audio) // frame_length

        # Find first frame above threshold
        start_frame = 0
        for i in range(n_frames):
            frame = audio[i * frame_length : (i + 1) * frame_length]
            if np.max(np.abs(frame)) > threshold:
                start_frame = i
                break

        # Find last frame above threshold
        end_frame = n_frames
        for i in range(n_frames - 1, -1, -1):
            frame = audio[i * frame_length : (i + 1) * frame_length]
            if np.max(np.abs(frame)) > threshold:
                end_frame = i + 1
                break

        # Add padding (keep some context)
        pad_frames = 2
        start_frame = max(0, start_frame - pad_frames)
        end_frame = min(n_frames, end_frame + pad_frames)

        start_sample = start_frame * frame_length
        end_sample = min(len(audio), end_frame * frame_length)

        return audio[start_sample:end_sample]

    def _pre_emphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies."""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])


class ASRService:
    """
    Automatic Speech Recognition using Qwen3-ASR.

    Features:
    - Two backends: transformers (less VRAM) or vLLM (faster)
    - Audio preprocessing for improved accuracy
    - Streaming transcription support

    Attributes:
        config: ASR configuration
        model_name: Name of the ASR model
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        """
        Initialize ASR service.

        Args:
            config: ASR configuration (uses defaults if None)
        """
        self._config = config or ASRConfig()
        self._model: Qwen3ASRModel | None = None
        self._started = False
        self._preprocessor = ASRPreprocessor()

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
                    audio_np = self._preprocessor.process(audio_np)

                if len(audio_np) == 0:
                    return TranscriptionResult(text="", language="", latency_ms=0.0)

                # Transcribe
                lang = language or self._config.language
                results = self._model.transcribe(
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
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Streaming transcription - process audio in chunks.

        Yields partial results as audio is processed for lower latency.

        Args:
            audio: PCM S16LE audio bytes
            language: Language hint
            chunk_duration_ms: Duration of each chunk in ms

        Yields:
            TranscriptionResult for each chunk
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        if len(audio) == 0:
            return

        # Convert to numpy
        audio_np = pcm_to_numpy(audio)

        # Preprocess entire audio
        audio_np = self._preprocessor.process(audio_np)

        if len(audio_np) == 0:
            return

        # Calculate chunk size
        chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)
        total_samples = len(audio_np)
        n_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        accumulated_text = ""

        for i in range(n_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_samples)
            chunk = audio_np[start:end]

            if len(chunk) < SAMPLE_RATE * 0.1:  # Skip chunks < 100ms
                continue

            timer = Timer()
            try:
                with timer:
                    lang = language or self._config.language
                    results = self._model.transcribe(
                        audio=(chunk, SAMPLE_RATE),
                        language=lang,
                    )

                result = results[0] if results else None
                text = result.text if result else ""
                text = self._clean_text(text)

                if text:
                    accumulated_text += " " + text

                    yield TranscriptionResult(
                        text=text,
                        language=result.language if result else "",
                        latency_ms=timer.elapsed_ms,
                    )

                    logger.debug(
                        "asr_stream_chunk",
                        chunk=i + 1,
                        total_chunks=n_chunks,
                        text=text[:50],
                        latency_ms=round(timer.elapsed_ms, 2),
                    )

            except Exception as e:
                logger.error("asr_stream_chunk_failed", chunk=i, error=str(e))
                # Continue with next chunk

    async def transcribe_progressive(
        self,
        audio_stream: AsyncIterator[bytes],
        min_chunk_ms: int = 500,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Progressive transcription - transcribe as audio arrives.

        Processes audio stream in real-time, yielding partial transcripts
        as they become available. Useful for live transcription.

        Args:
            audio_stream: Stream of PCM audio chunks
            min_chunk_ms: Minimum chunk duration to process (default 500ms)

        Yields:
            TranscriptionResult with partial/final transcripts

        Note:
            This provides better real-time feedback than transcribe_stream,
            but may have lower accuracy due to smaller context windows.
        """
        if not self._started or self._model is None:
            raise ASRError("ASR service not started")

        buffer = bytearray()
        min_chunk_bytes = int(SAMPLE_RATE * 2 * min_chunk_ms / 1000)  # 2 bytes per sample

        logger.debug(
            "asr_progressive_start",
            min_chunk_ms=min_chunk_ms,
            min_chunk_bytes=min_chunk_bytes,
        )

        async for audio_chunk in audio_stream:
            buffer.extend(audio_chunk)

            # Process when buffer reaches minimum size
            if len(buffer) >= min_chunk_bytes:
                chunk_to_process = bytes(buffer)
                buffer.clear()

                # Transcribe chunk
                try:
                    result = await self.transcribe(
                        chunk_to_process,
                        preprocess=self._config.preprocess,
                    )

                    if result.text.strip():
                        logger.debug(
                            "asr_progressive_chunk",
                            text=result.text[:50],
                            chunk_bytes=len(chunk_to_process),
                        )
                        yield result

                except Exception as e:
                    logger.error("asr_progressive_chunk_failed", error=str(e))
                    # Continue processing next chunks

        # Process remaining buffer
        if len(buffer) > SAMPLE_RATE * 2 * 0.1:  # > 100ms
            try:
                result = await self.transcribe(bytes(buffer), preprocess=self._config.preprocess)
                if result.text.strip():
                    yield result
            except Exception as e:
                logger.error("asr_progressive_final_failed", error=str(e))

        logger.debug("asr_progressive_complete")
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
                    audio_np = self._preprocessor.process(audio_np)

                if len(audio_np) == 0:
                    return TranscriptionResult(text="", language="", latency_ms=0.0)

                # Transcribe directly from numpy
                lang = language or self._config.language
                results = self._model.transcribe(
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

        # Remove repeated spaces
        while "  " in text:
            text = text.replace("  ", " ")

        # Remove common ASR artifacts
        artifacts = ["<|endoftext|>", "<unk>", "[UNK]", "<s>", "</s>"]
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
