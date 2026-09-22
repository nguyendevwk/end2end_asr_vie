"""Production hardening tests (no GPU/model required)."""

# ruff: noqa: ARG001, ARG002 - test doubles mirror real service signatures.

import asyncio
import time
from typing import AsyncIterator

import pytest

from voice_agent.core import VADResult, TranscriptionResult, SynthesisResult
from voice_agent.services.text_tasks import (
    LLMTask,
    PassthroughTask,
    RouterTask,
    available_tasks,
    collect,
    create_text_task,
    keyword_rule,
    register_text_task,
)
from voice_agent.utils.metrics import MetricsRegistry
from voice_agent.utils.resilience import AdmissionGate, retry_async
from voice_agent.utils.session_store import SessionSnapshot, SessionStore


class TestTextTasks:
    async def test_passthrough(self) -> None:
        task = PassthroughTask(prefix="Echo: ")
        assert await collect(task, "hello", []) == "Echo: hello"

    async def test_passthrough_empty(self) -> None:
        assert await collect(PassthroughTask(), "   ", []) == ""

    async def test_router_dispatch(self) -> None:
        router = RouterTask(
            rules=[(keyword_rule("stop", "dừng"), PassthroughTask(prefix="STOP:"))],
            default=PassthroughTask(prefix="CHAT:"),
        )
        assert await collect(router, "please stop now", []) == "STOP:please stop now"
        assert await collect(router, "xin chào", []) == "CHAT:xin chào"

    async def test_router_bad_predicate_skipped(self) -> None:
        def boom(text: str) -> bool:
            raise RuntimeError("bad rule")

        router = RouterTask(rules=[(boom, PassthroughTask())], default=PassthroughTask(prefix="D:"))
        assert await collect(router, "hi", []) == "D:hi"

    async def test_llm_task_delegates(self) -> None:
        class FakeLLM:
            async def stream(self, query: str, history: list) -> object:
                yield f"reply:{query}"
                yield "more"

        task = LLMTask(llm=FakeLLM())
        assert await collect(task, "hi", []) == "reply:hi more"

    def test_registry(self) -> None:
        assert "passthrough" in available_tasks()
        assert "llm" in available_tasks()
        assert "router" in available_tasks()
        with pytest.raises(ValueError, match="Unknown text task"):
            create_text_task("nope")

    def test_custom_registration(self) -> None:
        @register_text_task("test-echo")
        def _make(**kwargs: object) -> PassthroughTask:
            return PassthroughTask(prefix="T:")

        try:
            assert isinstance(create_text_task("test-echo"), PassthroughTask)
        finally:
            from voice_agent.services import text_tasks

            del text_tasks._registry["test-echo"]


class TestMetrics:
    def test_counter_and_prometheus(self) -> None:
        reg = MetricsRegistry()
        reg.inc("connections_accepted", 2)
        reg.observe("asr_ms", 120.0)
        out = reg.to_prometheus()
        assert "voice_connections_accepted 2" in out
        assert 'voice_asr_ms_bucket{le="+Inf"} 1' in out
        assert "voice_asr_ms_count 1" in out

    def test_dict_snapshot(self) -> None:
        reg = MetricsRegistry()
        reg.gauge("ccu_current").set(3.0)
        data = reg.to_dict()
        assert data["counters"] == {}
        assert data["gauges"]["ccu_current"] == 3.0  # type: ignore[index]


class TestResilience:
    async def test_retry_recovers(self) -> None:
        calls = 0

        async def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise RuntimeError("transient")
            return "ok"

        assert await retry_async(flaky, attempts=3, base_delay_s=0.001) == "ok"
        assert calls == 2

    async def test_retry_exhausts(self) -> None:
        async def always_fail() -> str:
            raise RuntimeError("down")

        with pytest.raises(RuntimeError, match="down"):
            await retry_async(always_fail, attempts=2, base_delay_s=0.001)

    async def test_retry_timeout(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(5)
            return "late"

        with pytest.raises(asyncio.TimeoutError):
            await retry_async(slow, attempts=1, timeout_s=0.01)

    async def test_gate_sheds(self) -> None:
        gate = AdmissionGate(max_concurrent=1)
        assert await gate.acquire() is True
        assert await gate.acquire() is False
        await gate.release()
        assert await gate.acquire() is True
        await gate.release()
        assert gate.high_watermark == 1


class TestSessionStore:
    def test_save_load(self) -> None:
        store = SessionStore(ttl_s=60.0)
        store.save(SessionSnapshot(session_id="s1", history=[{"role": "user", "content": "hi"}]))
        snap = store.load("s1")
        assert snap is not None and snap.history[0]["content"] == "hi"

    def test_expire(self) -> None:
        store = SessionStore(ttl_s=0.01)
        store.save(SessionSnapshot(session_id="s1"))
        time.sleep(0.02)
        assert store.load("s1") is None

    def test_evict_oldest(self) -> None:
        store = SessionStore(ttl_s=60.0, max_sessions=2)
        store.save(SessionSnapshot(session_id="a"))
        store.save(SessionSnapshot(session_id="b"))
        store.save(SessionSnapshot(session_id="c"))
        assert store.size == 2
        assert store.load("a") is None


class _FakeVAD:
    min_speech_ms: int = 250

    async def detect(self, audio: bytes) -> VADResult:
        return VADResult(is_speech=True, confidence=0.9, event=None)

    def reset(self) -> None:
        pass

    @property
    def is_started(self) -> bool:
        return True


class _FakeASR:
    streaming: bool = False
    preprocess_enabled: bool = True

    async def transcribe(self, audio: bytes, preprocess: bool = True) -> TranscriptionResult:
        return TranscriptionResult(text="xin chào", language="vi", latency_ms=5.0)

    @property
    def is_started(self) -> bool:
        return True


class _FakeTTS:
    stream_chunk_ms: int = 100

    async def synthesize_stream(self, text: str, chunk_duration_ms: int = 100) -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 160

    @property
    def is_started(self) -> bool:
        return True


class TestOrchestratorProduction:
    def _make(self, vad: object | None = None, **kwargs: object) -> object:
        from voice_agent.services.orchestrator import Orchestrator

        return Orchestrator(vad=vad or _FakeVAD(), asr=_FakeASR(), tts=_FakeTTS(), **kwargs)  # type: ignore[arg-type]

    async def test_empty_transcript(self) -> None:
        from voice_agent.services.orchestrator import Orchestrator

        class EmptyASR(_FakeASR):
            async def transcribe(self, audio: bytes, preprocess: bool = True) -> TranscriptionResult:
                return TranscriptionResult(text="", language="", latency_ms=0.0)

        orch = Orchestrator(vad=_FakeVAD(), asr=EmptyASR(), tts=_FakeTTS())
        results = [r async for r in orch._run_pipeline(b"\x00\x00" * 32000)]
        # Should not crash, and monitor should have end_turn called
        assert orch._monitor.turn_id >= 0

    async def test_passthrough_pipeline(self) -> None:
        orch = self._make(text_task=PassthroughTask(prefix="Echo: "))
        results = [r async for r in orch._run_pipeline(b"\x00\x00" * 32000)]
        texts = [r for r in results if isinstance(r, str)]
        assert any("TRANSCRIPT:xin chào" in t for t in texts)
        assert any("RESPONSE_TEXT:Echo: xin chào" in t for t in texts)
        assert any(isinstance(r, bytes) for r in results)
        assert orch._history[0]["role"] == "user"
        assert orch._history[1] == {"role": "assistant", "content": "Echo: xin chào"}

    async def test_asr_failure_degrades(self) -> None:
        class BadASR(_FakeASR):
            async def transcribe(self, audio: bytes, preprocess: bool = True) -> object:
                raise RuntimeError("gpu oom")

        from voice_agent.services.orchestrator import Orchestrator

        orch = Orchestrator(vad=_FakeVAD(), asr=BadASR(), tts=_FakeTTS())
        results = [r async for r in orch._run_pipeline(b"\x00\x00" * 32000)]
        assert any(isinstance(r, str) and r.startswith("ERROR:ASR") for r in results)

    async def test_snapshot_restore(self) -> None:
        orch = self._make(text_task=PassthroughTask())
        orch._history.append({"role": "user", "content": "hi"})
        snap = orch.snapshot()
        assert snap.history == [{"role": "user", "content": "hi"}]
        orch2 = self._make(text_task=PassthroughTask())
        orch2.restore(snap)
        assert orch2._history == snap.history

    async def test_set_task(self) -> None:
        orch = self._make(text_task=PassthroughTask(prefix="A: "))
        orch.set_task(PassthroughTask(prefix="B: "))
        results = [r async for r in orch._run_pipeline(b"\x00\x00" * 32000)]
        assert any("RESPONSE_TEXT:B: xin chào" in r for r in results if isinstance(r, str))

    async def test_process_audio_full_loop(self) -> None:
        class SeqVAD:
            min_speech_ms: int = 250
            _call_count = 0

            async def detect(self, audio: bytes) -> VADResult:
                SeqVAD._call_count += 1
                if SeqVAD._call_count == 1:
                    return VADResult(is_speech=True, confidence=0.9, event="start")
                elif SeqVAD._call_count == 3:
                    return VADResult(is_speech=False, confidence=0.1, event="end")
                return VADResult(is_speech=True, confidence=0.9, event=None)

            def reset(self) -> None:
                SeqVAD._call_count = 0

            @property
            def is_started(self) -> bool:
                return True

        orch = self._make(vad=SeqVAD(), text_task=PassthroughTask(prefix="Echo: "))
        results = []
        for chunk in [b"\x00\x00" * 8000, b"\x00\x00" * 8000, b"\x00\x00" * 8000]:
            async for r in orch.process_audio(chunk):
                results.append(r)
        texts = [r for r in results if isinstance(r, str)]
        assert any("LISTENING" in t for t in texts)
        assert any("TRANSCRIPT:" in t for t in texts)

    async def test_interrupt(self) -> None:
        orch = self._make(text_task=PassthroughTask())
        orch._state = orch._state.__class__.SPEAKING
        orch._interrupted = False
        await orch.interrupt()
        assert orch._interrupted is True

    async def test_clear_history(self) -> None:
        orch = self._make(text_task=PassthroughTask())
        orch._history = [{"role": "user", "content": "hi"}]
        orch.clear_history()
        assert len(orch._history) == 0

    async def test_history_cap_preserves_pairs(self) -> None:
        orch = self._make(text_task=PassthroughTask())
        orch._max_history = 4
        # Run pipeline multiple times to trigger history capping
        for _ in range(4):
            results = [r async for r in orch._run_pipeline(b"\x00\x00" * 32000)]
        # After 4 turns (8 messages), cap should have kicked in
        assert len(orch._history) <= orch._max_history
        # Check no orphaned messages (each user should have assistant reply)
        for i in range(0, len(orch._history), 2):
            if i + 1 < len(orch._history):
                assert orch._history[i]["role"] == "user"
                assert orch._history[i + 1]["role"] == "assistant"

    async def test_snapshot_returns_state(self) -> None:
        orch = self._make(text_task=PassthroughTask())
        orch._history = [{"role": "user", "content": "test"}]
        snap = orch.snapshot()
        assert snap.session_id == orch.session_id
        assert snap.history == [{"role": "user", "content": "test"}]
