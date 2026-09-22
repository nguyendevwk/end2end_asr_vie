"""LLM Service using Groq API (OpenAI compatible)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncIterator

from voice_agent.core import LLMConfig, LLMError
from voice_agent.utils import Timer, get_logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = get_logger(__name__)


class LLMService:
    """
    Large Language Model service using Groq API.

    Provides streaming response generation optimized for voice agents.
    Uses OpenAI-compatible API for easy provider switching.

    Attributes:
        config: LLM configuration
        model: Model name
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initialize LLM service.

        Args:
            config: LLM configuration (uses defaults if None)
        """
        self._config = config or LLMConfig()
        self._client: AsyncOpenAI | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize OpenAI client."""
        if self._started:
            return

        if not self._config.api_key:
            raise LLMError("API key is required. Set GROQ_API_KEY environment variable.")

        logger.info(
            "initializing_llm",
            model=self._config.model,
            base_url=self._config.base_url,
        )

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
            )

            self._started = True
            logger.info("llm_initialized", model=self._config.model)

        except ImportError as e:
            logger.error("openai_not_installed", error=str(e))
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            ) from e
        except Exception as e:
            logger.error("llm_init_failed", error=str(e))
            raise LLMError(f"Failed to initialize LLM client: {e}") from e

    async def stop(self) -> None:
        """Cleanup LLM resources."""
        if not self._started:
            return

        if self._client:
            await self._client.close()
        self._client = None
        self._started = False
        logger.info("llm_stopped")

    async def generate(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate complete response for query.

        Args:
            query: User query
            history: Conversation history as list of {"role": ..., "content": ...}

        Returns:
            Complete response text

        Raises:
            LLMError: If generation fails
        """
        if not self._started or self._client is None:
            raise LLMError("LLM service not started")

        timer = Timer()
        try:
            with timer:
                messages = self._build_messages(query, history)

                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )

                text = response.choices[0].message.content or "" if response.choices else ""

            logger.info(
                "llm_generate",
                latency_ms=round(timer.elapsed_ms, 2),
                input_length=len(query),
                output_length=len(text),
            )

            return text

        except Exception as e:
            logger.error("llm_generate_failed", error=str(e))
            raise LLMError(f"Generation failed: {e}") from e

    async def stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens for query.

        Yields complete sentences for optimal TTS synthesis.

        Args:
            query: User query
            history: Conversation history

        Yields:
            Complete sentences as they are generated

        Raises:
            LLMError: If streaming fails
        """
        if not self._started or self._client is None:
            raise LLMError("LLM service not started")

        start_time = time.perf_counter()
        ttft_recorded = False
        total_tokens = 0
        buffer = ""

        # Sentence-ending punctuation (Vietnamese + CJK)
        sentence_endings = (".", "!", "?", "。", "！", "？")
        # Fallback split points for long buffers without terminal punctuation:
        # commas/semicolons/colons/newlines keep TTS fed and cut TTFA.
        clause_breaks = (",", ";", ":", "、", "\n")
        max_buffer_chars = 120

        try:
            messages = self._build_messages(query, history)

            stream = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content

                    # Record time to first token
                    if not ttft_recorded:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                        logger.debug("llm_ttft", ttft_ms=round(ttft_ms, 2))
                        ttft_recorded = True

                    buffer += token

                    # Yield complete sentences immediately (lowest TTFA)
                    stripped = buffer.strip()
                    if stripped and stripped[-1] in sentence_endings:
                        yield stripped
                        buffer = ""
                    elif len(buffer) >= max_buffer_chars:
                        # Long sentence without terminal punctuation:
                        # split at last clause boundary so TTS doesn't starve.
                        cut = -1
                        for br in clause_breaks:
                            idx = buffer.rfind(br)
                            if idx > cut:
                                cut = idx
                        if cut > 0:
                            chunk_text = buffer[: cut + 1].strip()
                            if chunk_text:
                                yield chunk_text
                            buffer = buffer[cut + 1 :]

            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()

            total_ms = (time.perf_counter() - start_time) * 1000
            # Count tokens from API usage if available, else estimate from chars
            if hasattr(stream, "usage") and stream.usage:
                total_tokens = stream.usage.total_tokens
            else:
                total_tokens = sum(len(s.split()) for s in [query] + [buffer])
            logger.info(
                "llm_stream_complete",
                total_ms=round(total_ms, 2),
                token_count=total_tokens,
            )

        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
            raise LLMError(f"Streaming failed: {e}") from e

    def _build_messages(
        self,
        query: str,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Build messages list for API call."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._config.system_prompt},
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})

        return messages

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started

    @property
    def model(self) -> str:
        """Get model name."""
        return self._config.model
