"""Tests for utility modules: AudioBuffer, processor, monitor."""

from __future__ import annotations

import numpy as np
import pytest

from voice_agent.core import AudioFormat
from voice_agent.utils.audio import AudioBuffer, get_audio_duration_ms, pcm_to_numpy, numpy_to_pcm


class TestAudioBuffer:
    def test_add_and_get(self) -> None:
        buf = AudioBuffer(max_duration_ms=1000, sample_rate=16000, chunk_ms=100)
        buf.add(b"\x00\x01" * 100)
        buf.add(b"\x02\x03" * 100)
        assert buf.chunk_count == 2
        assert len(buf.get_all()) == 400

    def test_overflow_drops_oldest(self) -> None:
        buf = AudioBuffer(max_duration_ms=200, sample_rate=16000, chunk_ms=100)
        for i in range(5):
            buf.add(bytes([i, 0]) * 160)
        assert buf.dropped_chunks == 3
        assert buf.chunk_count == 2

    def test_clear(self) -> None:
        buf = AudioBuffer(max_duration_ms=1000)
        buf.add(b"\x00\x01" * 100)
        buf.clear()
        assert buf.chunk_count == 0
        assert buf.dropped_chunks == 0
        assert buf.duration_ms == 0.0

    def test_duration_ms(self) -> None:
        buf = AudioBuffer(max_duration_ms=1000, sample_rate=16000)
        buf.add(b"\x00\x00" * 16000)  # 16000 samples = 1 second at 16kHz
        assert buf.duration_ms == pytest.approx(1000.0, rel=0.01)

    def test_bool(self) -> None:
        buf = AudioBuffer(max_duration_ms=1000)
        assert not buf
        buf.add(b"\x00\x01" * 10)
        assert buf

    def test_get_numpy(self) -> None:
        buf = AudioBuffer(max_duration_ms=1000)
        audio = pcm_to_numpy(b"\x00\x40" * 100)  # ~0.5 amplitude
        buf.add(b"\x00\x40" * 100)
        result = buf.get_numpy()
        assert len(result) == 100
        assert result.dtype == np.float32


class TestAudioConversion:
    def test_pcm_numpy_roundtrip(self) -> None:
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = numpy_to_pcm(original)
        recovered = pcm_to_numpy(pcm)
        np.testing.assert_array_almost_equal(original, recovered, decimal=3)

    def test_empty_audio(self) -> None:
        assert len(pcm_to_numpy(b"")) == 0
        assert len(numpy_to_pcm(np.array([], dtype=np.float32))) == 0

    def test_odd_bytes_raises(self) -> None:
        from voice_agent.core import AudioError
        with pytest.raises(AudioError, match="even number"):
            pcm_to_numpy(b"\x00")


class TestAudioDuration:
    def test_duration_calculation(self) -> None:
        # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        # 1 second = 32000 bytes
        assert get_audio_duration_ms(b"\x00" * 32000) == pytest.approx(1000.0)
        assert get_audio_duration_ms(b"\x00" * 16000) == pytest.approx(500.0)

    def test_empty_duration(self) -> None:
        assert get_audio_duration_ms(b"") == 0.0


class TestAudioConfig:
    def test_chunk_bytes_s16le(self) -> None:
        from voice_agent.core import AudioConfig
        cfg = AudioConfig(sample_rate=16000, chunk_duration_ms=100, format=AudioFormat.PCM_S16LE)
        assert cfg.chunk_bytes == 3200  # 1600 samples * 2 bytes

    def test_chunk_bytes_f32le(self) -> None:
        from voice_agent.core import AudioConfig
        cfg = AudioConfig(sample_rate=16000, chunk_duration_ms=100, format=AudioFormat.PCM_F32LE)
        assert cfg.chunk_bytes == 6400  # 1600 samples * 4 bytes
