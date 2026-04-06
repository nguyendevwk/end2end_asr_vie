"""Tests for TTS service."""

import pytest

from voice_agent.core import SynthesisResult, TTSConfig


class TestTTSConfig:
    """Tests for TTS configuration."""

    def test_default_config(self) -> None:
        """Should create default config."""
        config = TTSConfig()

        assert config.model_name == "g-group-ai-lab/gwen-tts-0.6B"
        assert config.voice_ref_audio is None
        assert config.voice_ref_text is None
        assert config.temperature == 0.3

    def test_voice_cloning_config(self) -> None:
        """Should create voice cloning config."""
        config = TTSConfig(
            voice_ref_audio="/path/to/audio.wav",
            voice_ref_text="Reference transcript",
        )

        assert config.voice_ref_audio == "/path/to/audio.wav"
        assert config.voice_ref_text == "Reference transcript"


class TestSynthesisResult:
    """Tests for SynthesisResult."""

    def test_result_creation(self) -> None:
        """Should create result correctly."""
        result = SynthesisResult(
            audio=b"\x00\x00" * 1600,
            sample_rate=16000,
            duration_ms=100.0,
            latency_ms=150.5,
        )

        assert len(result.audio) == 3200
        assert result.sample_rate == 16000
        assert result.duration_ms == 100.0
        assert result.latency_ms == 150.5

    def test_empty_result(self) -> None:
        """Should handle empty result."""
        result = SynthesisResult(
            audio=b"",
            sample_rate=16000,
            duration_ms=0.0,
            latency_ms=0.0,
        )

        assert result.audio == b""
        assert result.duration_ms == 0.0


class TestTTSServiceIntegration:
    """Integration tests for TTS service (requires model)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_tts_synthesize_empty(self) -> None:
        """Should handle empty text."""
        from voice_agent.services import TTSService

        service = TTSService()
        await service.start()

        try:
            result = await service.synthesize("")

            assert result.audio == b""
            assert result.duration_ms == 0.0
        finally:
            await service.stop()

    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_tts_synthesize_whitespace(self) -> None:
        """Should handle whitespace text."""
        from voice_agent.services import TTSService

        service = TTSService()
        await service.start()

        try:
            result = await service.synthesize("   ")

            assert result.audio == b""
        finally:
            await service.stop()
