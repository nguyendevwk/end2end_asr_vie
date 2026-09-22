"""VAD Service using Silero VAD."""

from __future__ import annotations

import asyncio

import numpy as np
import torch

from voice_agent.core import SAMPLE_RATE, VADConfig, VADError, VADResult
from voice_agent.utils import Timer, get_logger, pcm_to_numpy

logger = get_logger(__name__)

# Silero VAD requires exactly 512 samples at 16kHz
VAD_CHUNK_SAMPLES = 512


class VADService:
    """
    Voice Activity Detection using Silero VAD.

    Detects speech segments in audio stream with low latency.
    Uses VADIterator for streaming detection with start/end events.

    Note: Silero VAD requires exactly 512 samples at 16kHz.
    This service buffers incoming audio and processes in 512-sample chunks.

    Attributes:
        config: VAD configuration
        sample_rate: Audio sample rate (16000)
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        """
        Initialize VAD service.

        Args:
            config: VAD configuration (uses defaults if None)
        """
        self._config = config or VADConfig()
        self._model: torch.nn.Module | None = None
        self._iterator: _VADIterator | None = None
        self._started = False
        # Guards torch model + iterator state across concurrent sessions.
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Load Silero VAD model."""
        if self._started:
            return

        logger.info("loading_vad_model")
        try:
            # Load Silero VAD from torch.hub
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model.eval()

            # Initialize iterator for streaming
            self._iterator = _VADIterator(
                model=self._model,
                threshold=self._config.threshold,
                min_speech_duration_ms=self._config.min_speech_ms,
                min_silence_duration_ms=self._config.min_silence_ms,
                speech_pad_ms=self._config.speech_pad_ms,
            )

            self._started = True
            logger.info("vad_model_loaded")

        except Exception as e:
            logger.error("vad_load_failed", error=str(e))
            raise VADError(f"Failed to load VAD model: {e}") from e

    async def stop(self) -> None:
        """Cleanup VAD resources."""
        if not self._started:
            return

        self._model = None
        self._iterator = None
        self._started = False
        logger.info("vad_stopped")

    async def detect(self, audio: bytes) -> VADResult:
        """
        Detect voice activity in audio chunk.

        Args:
            audio: PCM S16LE audio bytes (any size, will be buffered)

        Returns:
            VADResult with is_speech, confidence, and event
            (aggregated result from all processed 512-sample chunks)

        Raises:
            VADError: If detection fails
        """
        if not self._started or self._iterator is None:
            raise VADError("VAD service not started")

        timer = Timer()
        try:
            with timer:
                # Convert to numpy
                audio_np = pcm_to_numpy(audio)

            # Torch inference blocks the event loop (~5ms x N chunks);
            # run it in a worker thread so WebSocket serving stays responsive.
            # The lock serializes model access across concurrent sessions.
            async with self._lock:
                iterator = self._iterator
                if iterator is None:
                    raise VADError("VAD service stopped during inference")
                result = await asyncio.to_thread(iterator, audio_np)

            logger.debug(
                "vad_detect",
                latency_ms=round(timer.elapsed_ms, 2),
                is_speech=result.is_speech,
                vad_event=result.event,
            )

            return result

        except Exception as e:
            logger.error("vad_detect_failed", error=str(e))
            raise VADError(f"VAD detection failed: {e}") from e

    def reset(self) -> None:
        """Reset VAD state for new utterance."""
        if self._iterator is not None:
            self._iterator.reset_states()
            logger.debug("vad_reset")

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started

    @property
    def min_speech_ms(self) -> int:
        """Minimum speech duration (ms) before a 'start' event fires."""
        return self._config.min_speech_ms

    @property
    def threshold(self) -> float:
        """Speech probability threshold."""
        return self._config.threshold

    def session(self) -> VADSession:
        """
        Create an isolated per-connection session sharing the loaded model.

        Each WebSocket connection must use its own session; sharing one
        iterator across connections corrupts speech/silence state.
        """
        if not self._started or self._model is None:
            raise VADError("VAD service not started")
        return VADSession(service=self)


class VADSession:
    """Per-connection VAD state (iterator + buffer) over a shared model."""

    def __init__(self, service: VADService) -> None:
        self._service = service
        config = service._config
        # _VADIterator is module-global by call time; no import needed.
        self._iterator = _VADIterator(
            model=service._model,
            threshold=config.threshold,
            min_speech_duration_ms=config.min_speech_ms,
            min_silence_duration_ms=config.min_silence_ms,
            speech_pad_ms=config.speech_pad_ms,
        )

    @property
    def min_speech_ms(self) -> int:
        return self._service.min_speech_ms

    @property
    def threshold(self) -> float:
        return self._service.threshold

    async def detect(self, audio: bytes) -> VADResult:
        """Detect voice activity (model access serialized by service lock)."""
        timer = Timer()
        try:
            with timer:
                audio_np = pcm_to_numpy(audio)
            async with self._service._lock:
                result = await asyncio.to_thread(self._iterator, audio_np)
            logger.debug(
                "vad_detect",
                latency_ms=round(timer.elapsed_ms, 2),
                is_speech=result.is_speech,
                vad_event=result.event,
            )
            return result
        except Exception as e:
            logger.error("vad_detect_failed", error=str(e))
            raise VADError(f"VAD detection failed: {e}") from e

    def reset(self) -> None:
        """Reset session state for a new utterance."""
        if self._iterator is not None:
            self._iterator.reset_states()
        logger.debug("vad_reset")


class _VADIterator:
    """
    Internal VAD iterator for streaming detection.

    Tracks speech state and emits start/end events.
    Buffers audio to process in 512-sample chunks as required by Silero VAD.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 1000,
        speech_pad_ms: int = 30,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._min_speech_samples = SAMPLE_RATE * min_speech_duration_ms // 1000
        self._min_silence_samples = SAMPLE_RATE * min_silence_duration_ms // 1000
        self._speech_pad_samples = SAMPLE_RATE * speech_pad_ms // 1000

        # State
        self._triggered = False
        self._speech_start = 0
        self._temp_end = 0
        self._current_sample = 0

        # Audio buffer for chunking
        self._buffer = np.array([], dtype=np.float32)

    def reset_states(self) -> None:
        """Reset internal state."""
        self._model.reset_states()
        self._triggered = False
        self._speech_start = 0
        self._temp_end = 0
        self._current_sample = 0
        self._buffer = np.array([], dtype=np.float32)

    def __call__(self, audio_np: np.ndarray) -> VADResult:
        """
        Process audio chunk and return VAD result.

        Args:
            audio_np: Audio samples as float32 numpy array (any length)

        Returns:
            VADResult with speech detection info
            (aggregated from all 512-sample chunks processed)
        """
        if len(audio_np) == 0:
            return VADResult(is_speech=self._triggered, confidence=0.0)

        # Add to buffer
        self._buffer = np.concatenate([self._buffer, audio_np])

        # Process all complete 512-sample chunks
        last_prob = 0.0
        event: str | None = None

        while len(self._buffer) >= VAD_CHUNK_SAMPLES:
            # Extract chunk
            chunk = self._buffer[:VAD_CHUNK_SAMPLES]
            self._buffer = self._buffer[VAD_CHUNK_SAMPLES:]

            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk).float()

            # Get speech probability
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, SAMPLE_RATE).item()

            last_prob = speech_prob
            self._current_sample += VAD_CHUNK_SAMPLES

            # State machine for speech detection
            if speech_prob >= self._threshold:
                if not self._triggered:
                    if self._speech_start == 0:
                        self._speech_start = self._current_sample
                    if (
                        self._current_sample - self._speech_start
                        >= self._min_speech_samples
                    ):
                        # Speech started only after minimum speech duration
                        self._triggered = True
                        event = "start"
                        self._temp_end = 0
                else:
                    self._temp_end = 0
            else:
                self._speech_start = 0
                if self._triggered:
                    if self._temp_end == 0:
                        self._temp_end = self._current_sample
                    elif (
                        self._current_sample - self._temp_end
                        >= self._min_silence_samples
                    ):
                        # Speech ended (enough silence)
                        self._triggered = False
                        self._temp_end = 0
                        event = "end"

        return VADResult(
            is_speech=self._triggered,
            confidence=last_prob,
            event=event,
        )
