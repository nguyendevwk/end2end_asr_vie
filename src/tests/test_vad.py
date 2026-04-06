"""Tests for VAD service."""

import pytest
import numpy as np

from voice_agent.core import VADConfig, VADResult
from voice_agent.utils import numpy_to_pcm


class TestVADService:
    """Tests for VAD service."""

    @pytest.fixture
    def vad_config(self) -> VADConfig:
        """Create test VAD config."""
        return VADConfig(
            threshold=0.5,
            min_silence_ms=500,
            min_speech_ms=200,
        )

    def test_vad_result_creation(self) -> None:
        """Should create VADResult correctly."""
        result = VADResult(
            is_speech=True,
            confidence=0.85,
            event="start",
        )

        assert result.is_speech is True
        assert result.confidence == 0.85
        assert result.event == "start"

    def test_vad_result_no_event(self) -> None:
        """Should handle VADResult without event."""
        result = VADResult(
            is_speech=False,
            confidence=0.1,
        )

        assert result.is_speech is False
        assert result.event is None

    def test_generate_silence_audio(self) -> None:
        """Should generate valid silence audio."""
        # Generate 100ms of silence at 16kHz
        samples = 1600  # 16000 * 0.1
        silence = np.zeros(samples, dtype=np.float32)
        audio_bytes = numpy_to_pcm(silence)

        assert len(audio_bytes) == samples * 2  # 2 bytes per sample (S16LE)

    def test_generate_speech_like_audio(self) -> None:
        """Should generate valid speech-like audio."""
        # Generate 100ms of sine wave (simulated speech)
        samples = 1600
        t = np.linspace(0, 0.1, samples, dtype=np.float32)
        speech = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone
        audio_bytes = numpy_to_pcm(speech)

        assert len(audio_bytes) == samples * 2


class TestVADServiceIntegration:
    """Integration tests for VAD service (requires model)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_vad_detect_silence(self) -> None:
        """Should detect silence correctly."""
        from voice_agent.services import VADService

        service = VADService()
        await service.start()

        try:
            # Generate silence
            silence = np.zeros(512, dtype=np.float32)
            audio = numpy_to_pcm(silence)

            result = await service.detect(audio)

            assert result.is_speech is False
            assert result.confidence < 0.5
        finally:
            await service.stop()

    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_vad_reset(self) -> None:
        """Should reset VAD state."""
        from voice_agent.services import VADService

        service = VADService()
        await service.start()

        try:
            service.reset()
            assert service.is_started is True
        finally:
            await service.stop()
