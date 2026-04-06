"""TTS Service using Gwen-TTS-0.6B."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np
import torch

from voice_agent.core import SAMPLE_RATE, SynthesisResult, TTSConfig, TTSError
from voice_agent.utils import Timer, get_logger, numpy_to_pcm, resample

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

logger = get_logger(__name__)

# Gwen-TTS native sample rate
TTS_NATIVE_SAMPLE_RATE = 24000


class TTSService:
    """
    Text-to-Speech using Gwen-TTS-0.6B (Vietnamese optimized).

    Supports voice cloning with reference audio.
    Outputs PCM S16LE audio at 16kHz.

    Attributes:
        config: TTS configuration
        model_name: Name of the TTS model
    """

    def __init__(self, config: TTSConfig | None = None) -> None:
        """
        Initialize TTS service.

        Args:
            config: TTS configuration (uses defaults if None)
        """
        self._config = config or TTSConfig()
        self._model: Qwen3TTSModel | None = None
        self._started = False

        # Generation config for Gwen-TTS
        self._generation_config = {
            "temperature": self._config.temperature,
            "top_k": self._config.top_k,
            "top_p": self._config.top_p,
            "max_new_tokens": 4096,
            "repetition_penalty": 2.0,
            "subtalker_do_sample": True,
            "subtalker_temperature": 0.1,
            "subtalker_top_k": 20,
            "subtalker_top_p": 1.0,
        }

    async def start(self) -> None:
        """Load TTS model."""
        if self._started:
            return

        logger.info("loading_tts_model", model=self._config.model_name)

        try:
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self._config.model_name,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            self._started = True
            logger.info("tts_model_loaded", model=self._config.model_name)

        except ImportError as e:
            logger.error("qwen_tts_not_installed", error=str(e))
            raise TTSError(
                "qwen-tts package not installed. Run: pip install qwen-tts"
            ) from e
        except Exception as e:
            logger.error("tts_load_failed", error=str(e))
            raise TTSError(f"Failed to load TTS model: {e}") from e

    async def stop(self) -> None:
        """Cleanup TTS resources."""
        if not self._started:
            return

        self._model = None
        self._started = False
        logger.info("tts_stopped")

    async def synthesize(self, text: str) -> SynthesisResult:
        """
        Synthesize text to speech.

        Args:
            text: Input text to synthesize

        Returns:
            SynthesisResult with audio bytes, sample rate, duration, and latency

        Raises:
            TTSError: If synthesis fails
        """
        if not self._started or self._model is None:
            raise TTSError("TTS service not started")

        if not text.strip():
            return SynthesisResult(
                audio=b"",
                sample_rate=SAMPLE_RATE,
                duration_ms=0.0,
                latency_ms=0.0,
            )

        timer = Timer()
        try:
            with timer:
                # Generate speech
                if self._config.voice_ref_audio and self._config.voice_ref_text:
                    # Use voice cloning
                    wavs, sr = self._model.generate_voice_clone(
                        text=text,
                        language="Vietnamese",
                        ref_audio=self._config.voice_ref_audio,
                        ref_text=self._config.voice_ref_text,
                        **self._generation_config,
                    )
                else:
                    # Default voice (no cloning)
                    wavs, sr = self._model.generate(
                        text=text,
                        language="Vietnamese",
                        **self._generation_config,
                    )

                # Get first result
                audio_np = wavs[0] if isinstance(wavs, list) else wavs

                # Resample from 24kHz to 16kHz
                if sr != SAMPLE_RATE:
                    audio_np = resample(audio_np, sr, SAMPLE_RATE)

                # Convert to PCM S16LE
                audio_bytes = numpy_to_pcm(audio_np)

            # Calculate duration
            duration_ms = len(audio_np) / SAMPLE_RATE * 1000

            logger.info(
                "tts_synthesize",
                latency_ms=round(timer.elapsed_ms, 2),
                text_length=len(text),
                audio_duration_ms=round(duration_ms, 2),
            )

            return SynthesisResult(
                audio=audio_bytes,
                sample_rate=SAMPLE_RATE,
                duration_ms=duration_ms,
                latency_ms=timer.elapsed_ms,
            )

        except Exception as e:
            logger.error("tts_synthesize_failed", error=str(e), text=text[:50])
            raise TTSError(f"Synthesis failed: {e}") from e

    async def synthesize_stream(
        self,
        text: str,
        chunk_duration_ms: int = 100,
    ) -> AsyncIterator[bytes]:
        """
        Stream audio chunks as they're generated (progressive synthesis).

        For models that don't support true streaming, this simulates streaming
        by generating full audio and yielding it in small chunks.

        Args:
            text: Input text to synthesize
            chunk_duration_ms: Duration of each audio chunk (default 100ms)

        Yields:
            Audio chunks (PCM S16LE bytes)

        Raises:
            TTSError: If synthesis fails
        """
        if not self._started or self._model is None:
            raise TTSError("TTS service not started")

        if not text.strip():
            return

        logger.debug(
            "tts_stream_start",
            text_length=len(text),
            chunk_duration_ms=chunk_duration_ms,
        )

        start_time = asyncio.get_event_loop().time()
        first_chunk = True

        try:
            # Generate full audio (Gwen-TTS doesn't support progressive generation)
            # TODO: Implement true streaming if model supports it
            if self._config.voice_ref_audio and self._config.voice_ref_text:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language="Vietnamese",
                    ref_audio=self._config.voice_ref_audio,
                    ref_text=self._config.voice_ref_text,
                    **self._generation_config,
                )
            else:
                wavs, sr = self._model.generate(
                    text=text,
                    language="Vietnamese",
                    **self._generation_config,
                )

            audio_np = wavs[0] if isinstance(wavs, list) else wavs

            # Resample to 16kHz
            if sr != SAMPLE_RATE:
                audio_np = resample(audio_np, sr, SAMPLE_RATE)

            # Calculate chunk size in samples
            chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)

            # Yield audio in chunks
            for i in range(0, len(audio_np), chunk_samples):
                chunk = audio_np[i : i + chunk_samples]
                chunk_bytes = numpy_to_pcm(chunk)

                if first_chunk:
                    ttfa_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    logger.info(
                        "tts_first_chunk",
                        ttfa_ms=round(ttfa_ms, 2),
                        text_length=len(text),
                    )
                    first_chunk = False

                yield chunk_bytes

                # Small delay to simulate real-time playback rate
                # This prevents overwhelming the client buffer
                await asyncio.sleep(chunk_duration_ms / 1000 * 0.8)

            total_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(
                "tts_stream_complete",
                total_ms=round(total_ms, 2),
                audio_duration_ms=round(len(audio_np) / SAMPLE_RATE * 1000, 2),
            )

        except Exception as e:
            logger.error("tts_stream_failed", error=str(e), text=text[:50])
            raise TTSError(f"Streaming synthesis failed: {e}") from e

    async def synthesize_numpy(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text to numpy array.

        Args:
            text: Input text to synthesize

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            TTSError: If synthesis fails
        """
        result = await self.synthesize(text)
        from voice_agent.utils import pcm_to_numpy

        audio_np = pcm_to_numpy(result.audio)
        return audio_np, result.sample_rate

    def set_voice(
        self,
        ref_audio: str,
        ref_text: str,
    ) -> None:
        """
        Set reference voice for cloning.

        Args:
            ref_audio: Path to reference audio file
            ref_text: Transcript of reference audio
        """
        self._config = TTSConfig(
            model_name=self._config.model_name,
            voice_ref_audio=ref_audio,
            voice_ref_text=ref_text,
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
        )
        logger.info("voice_set", ref_audio=ref_audio)

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._config.model_name
