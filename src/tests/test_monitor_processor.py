"""Tests for monitor and processor modules."""

from __future__ import annotations

import numpy as np
import pytest

from voice_agent.utils.monitor import LatencyStats, PipelineMonitor
from voice_agent.utils.processor import AudioPostprocessor, AudioPreprocessor


class TestLatencyStats:
    def test_record(self) -> None:
        stats = LatencyStats()
        stats.record(10.0)
        stats.record(20.0)
        assert stats.count == 2
        assert stats.total_ms == 30.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 20.0
        assert stats.avg_ms == 15.0

    def test_empty_stats(self) -> None:
        stats = LatencyStats()
        assert stats.avg_ms == 0.0
        d = stats.to_dict()
        assert d["min_ms"] == 0.0

    def test_to_dict(self) -> None:
        stats = LatencyStats()
        stats.record(100.0)
        d = stats.to_dict()
        assert d["count"] == 1
        assert d["avg_ms"] == 100.0


class TestPipelineMonitor:
    def test_turn_tracking(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        assert mon.turn_id == 1
        mon.start_turn()
        assert mon.turn_id == 2

    def test_record_vad(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        mon.record_vad(5.0, True)
        assert mon.vad_stats.count == 1

    def test_record_asr(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        mon.record_asr(100.0, 1000.0, 50)
        assert mon.asr_stats.count == 1

    def test_record_llm(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        mon.record_llm(50.0, 200.0, 10)
        assert mon.llm_stats.count == 1

    def test_record_tts(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        mon.record_tts(80.0, 20, 500.0)
        assert mon.tts_stats.count == 1

    def test_record_first_audio(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        ttfa = mon.record_first_audio()
        assert ttfa > 0

    def test_end_turn(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        e2e = mon.end_turn()
        assert e2e > 0

    def test_get_summary(self) -> None:
        mon = PipelineMonitor(session_id="test")
        mon.start_turn()
        mon.record_vad(5.0, True)
        mon.record_asr(100.0, 1000.0, 50)
        mon.record_llm(50.0, 200.0, 10)
        mon.record_tts(80.0, 20, 500.0)
        mon.record_first_audio()
        mon.end_turn()
        summary = mon.get_summary()
        assert "vad" in summary
        assert "asr" in summary
        assert "llm" in summary
        assert "tts" in summary
        assert "ttfa" in summary
        assert "e2e" in summary


class TestAudioPreprocessor:
    def test_empty_audio(self) -> None:
        prep = AudioPreprocessor()
        assert len(prep.process_numpy(np.array([], dtype=np.float32))) == 0

    def test_process_numpy_no_filter(self) -> None:
        from voice_agent.utils.processor import PreprocessConfig
        # Disable filters to avoid scipy/torch mock conflict
        cfg = PreprocessConfig(high_pass_hz=0, low_pass_hz=0, noise_gate_enabled=False, normalize=False, remove_dc=False)
        prep = AudioPreprocessor(cfg)
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        result = prep.process_numpy(audio)
        assert len(result) == len(audio)
        assert result.dtype == np.float32

    def test_dc_removal(self) -> None:
        from voice_agent.utils.processor import PreprocessConfig
        cfg = PreprocessConfig(high_pass_hz=0, low_pass_hz=0, noise_gate_enabled=False, normalize=False)
        prep = AudioPreprocessor(cfg)
        audio = np.ones(1000, dtype=np.float32) * 0.5  # DC offset
        result = prep.process_numpy(audio)
        assert abs(np.mean(result)) < 0.01


class TestAudioPostprocessor:
    def test_empty_audio(self) -> None:
        post = AudioPostprocessor()
        assert len(post.process_numpy(np.array([], dtype=np.float32))) == 0

    def test_process_numpy(self) -> None:
        from voice_agent.utils.processor import PostprocessConfig
        cfg = PostprocessConfig(normalize=False, limiter_enabled=False)
        post = AudioPostprocessor(cfg)
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        result = post.process_numpy(audio)
        assert len(result) == len(audio)

    def test_soft_limiter(self) -> None:
        from voice_agent.utils.processor import PostprocessConfig
        post = AudioPostprocessor(PostprocessConfig(limiter_enabled=True, normalize=False))
        loud = np.ones(1000, dtype=np.float32) * 0.9
        result = post.process_numpy(loud)
        assert np.max(np.abs(result)) <= 1.0

    def test_fades(self) -> None:
        from voice_agent.utils.processor import PostprocessConfig
        post = AudioPostprocessor(PostprocessConfig(normalize=False, limiter_enabled=False))
        audio = np.ones(16000, dtype=np.float32)
        result = post.process_numpy(audio)
        # First sample should be faded (less than 1.0)
        assert result[0] < 1.0
        # Last sample should be faded
        assert result[-1] < 1.0
