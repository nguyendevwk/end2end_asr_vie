"""Tests for pluggable text tasks registry and built-in tasks."""

import pytest

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


class TestRegistry:
    def test_available_tasks_includes_builtins(self) -> None:
        names = available_tasks()
        assert "passthrough" in names
        assert "llm" in names
        assert "router" in names

    def test_create_passthrough(self) -> None:
        task = create_text_task("passthrough")
        assert task.name == "passthrough"
        assert isinstance(task, PassthroughTask)

    def test_create_passthrough_with_prefix(self) -> None:
        task = create_text_task("passthrough", prefix="Echo: ")
        assert isinstance(task, PassthroughTask)
        assert task._prefix == "Echo: "

    def test_create_llm_requires_llm(self) -> None:
        with pytest.raises(ValueError, match="requires an llm="):
            create_text_task("llm")

    def test_create_llm_with_service(self) -> None:
        fake_llm = object()
        task = create_text_task("llm", llm=fake_llm)
        assert task.name == "llm"
        assert isinstance(task, LLMTask)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown text task"):
            create_text_task("nonexistent")

    def test_register_custom_task(self) -> None:
        @register_text_task("test-custom")
        def _make(**kwargs: object) -> PassthroughTask:
            return PassthroughTask(prefix="custom")

        task = create_text_task("test-custom")
        assert task.name == "passthrough"
        assert isinstance(task, PassthroughTask)

    def test_create_router(self) -> None:
        task = create_text_task("router")
        assert task.name == "router"
        assert isinstance(task, RouterTask)


class TestPassthroughTask:
    @pytest.mark.asyncio
    async def test_yields_transcript(self) -> None:
        task = PassthroughTask()
        chunks = [c async for c in task.stream("hello", [])]
        assert chunks == ["hello"]

    @pytest.mark.asyncio
    async def test_with_prefix(self) -> None:
        task = PassthroughTask(prefix="You said: ")
        chunks = [c async for c in task.stream("hello", [])]
        assert chunks == ["You said: hello"]

    @pytest.mark.asyncio
    async def test_empty_transcript_yields_nothing(self) -> None:
        task = PassthroughTask()
        chunks = [c async for c in task.stream("", [])]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_prefix_only_yields_nothing(self) -> None:
        task = PassthroughTask(prefix="  ")
        chunks = [c async for c in task.stream("", [])]
        assert chunks == []


class TestLLMTask:
    @pytest.mark.asyncio
    async def test_delegates_to_llm_stream(self) -> None:
        class FakeLLM:
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield f"response to {transcript}"

        task = LLMTask(llm=FakeLLM())
        chunks = [c async for c in task.stream("hi", [])]
        assert chunks == ["response to hi"]

    @pytest.mark.asyncio
    async def test_forwards_history(self) -> None:
        received_history: list = []

        class FakeLLM:
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                received_history.extend(history)
                yield "ok"

        task = LLMTask(llm=FakeLLM())
        history = [{"role": "user", "content": "prev"}]
        await collect(task, "test", history)
        assert received_history == history


class TestRouterTask:
    @pytest.mark.asyncio
    async def test_no_rules_uses_default(self) -> None:
        router = RouterTask()
        chunks = [c async for c in router.stream("anything", [])]
        assert chunks == ["anything"]

    @pytest.mark.asyncio
    async def test_matching_rule_dispatches(self) -> None:
        class TaskA:
            name = "a"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "from A"

        router = RouterTask()
        router.add_rule(keyword_rule("help"), TaskA())
        chunks = [c async for c in router.stream("I need help", [])]
        assert chunks == ["from A"]

    @pytest.mark.asyncio
    async def test_non_matching_uses_default(self) -> None:
        class TaskA:
            name = "a"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "from A"

        router = RouterTask()
        router.add_rule(keyword_rule("help"), TaskA())
        chunks = [c async for c in router.stream("hello world", [])]
        assert chunks == ["hello world"]

    @pytest.mark.asyncio
    async def test_first_matching_rule_wins(self) -> None:
        class TaskA:
            name = "a"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "from A"

        class TaskB:
            name = "b"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "from B"

        router = RouterTask()
        router.add_rule(keyword_rule("foo"), TaskA())
        router.add_rule(keyword_rule("foo"), TaskB())
        chunks = [c async for c in router.stream("foo bar", [])]
        assert chunks == ["from A"]

    @pytest.mark.asyncio
    async def test_exception_in_predicate_skips_rule(self) -> None:
        class TaskA:
            name = "a"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "from A"

        def bad_predicate(text: str) -> bool:
            raise RuntimeError("boom")

        router = RouterTask()
        router.add_rule(bad_predicate, TaskA())
        chunks = [c async for c in router.stream("anything", [])]
        assert chunks == ["anything"]


class TestKeywordRule:
    def test_matches_single_keyword(self) -> None:
        pred = keyword_rule("hello")
        assert pred("hello world") is True
        assert pred("goodbye") is False

    def test_matches_any_of_multiple(self) -> None:
        pred = keyword_rule("cat", "dog")
        assert pred("i have a cat") is True
        assert pred("i have a dog") is True
        assert pred("i have a bird") is False

    def test_case_insensitive(self) -> None:
        pred = keyword_rule("Hello")
        assert pred("hello world") is True
        assert pred("HELLO world") is False  # predicate receives pre-lowered input from RouterTask


class TestCollect:
    @pytest.mark.asyncio
    async def test_collects_chunks(self) -> None:
        task = PassthroughTask()
        result = await collect(task, "hello", [])
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_joins_multiple_chunks(self) -> None:
        class MultiChunk:
            name = "multi"
            async def stream(self, transcript: str, history: list) -> str:  # type: ignore[override]
                yield "a"
                yield "b"
                yield "c"

        result = await collect(MultiChunk(), "", [])
        assert result == "a b c"
