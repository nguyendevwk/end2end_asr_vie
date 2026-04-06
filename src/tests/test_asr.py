"""Tests for ASR service."""

import pytest

from voice_agent.core import ASRConfig, TranscriptionResult


class TestASRConfig:
    """Tests for ASR configuration."""

    def test_default_config(self) -> None:
        """Should create default config."""
        config = ASRConfig()

        assert config.model_name == "Qwen/Qwen3-ASR-0.6B"
        assert config.language is None
        assert config.max_new_tokens == 256
        assert config.gpu_memory_utilization == 0.5

    def test_custom_config(self) -> None:
        """Should create custom config."""
        config = ASRConfig(
            model_name="Qwen/Qwen3-ASR-1.7B",
            language="Vietnamese",
            gpu_memory_utilization=0.7,
        )

        assert config.model_name == "Qwen/Qwen3-ASR-1.7B"
        assert config.language == "Vietnamese"
        assert config.gpu_memory_utilization == 0.7


class TestTranscriptionResult:
    """Tests for TranscriptionResult."""

    def test_result_creation(self) -> None:
        """Should create result correctly."""
        result = TranscriptionResult(
            text="Xin chào",
            language="Vietnamese",
            latency_ms=250.5,
        )

        assert result.text == "Xin chào"
        assert result.language == "Vietnamese"
        assert result.latency_ms == 250.5

    def test_empty_result(self) -> None:
        """Should handle empty result."""
        result = TranscriptionResult(
            text="",
            language="",
            latency_ms=0.0,
        )

        assert result.text == ""
        assert result.language == ""


class TestASRServiceIntegration:
    """Integration tests for ASR service (requires model)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    async def test_asr_transcribe_empty(self) -> None:
        """Should handle empty audio."""
        from voice_agent.services import ASRService

        service = ASRService()
        await service.start()

        try:
            result = await service.transcribe(b"")

            assert result.text == ""
            assert result.latency_ms == 0.0
        finally:
            await service.stop()
