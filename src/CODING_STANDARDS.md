# 📋 CODING STANDARDS - Voice Agent Demo

> **Mục tiêu**: Source clean, đơn giản, dễ đọc, dễ mở rộng, tối ưu latency
> **Triển khai**: UV package manager + Python 3.12+

---

## 1. 📁 CẤU TRÚC DỰ ÁN

```
src/
├── voice_agent/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings (Pydantic)
│   │
│   ├── core/                   # Business logic (framework-agnostic)
│   │   ├── __init__.py
│   │   ├── types.py            # Type definitions, dataclasses
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── constants.py        # Constants, enums
│   │
│   ├── services/               # AI Services
│   │   ├── __init__.py
│   │   ├── vad.py              # Silero VAD
│   │   ├── asr.py              # Qwen3-ASR (vLLM)
│   │   ├── tts.py              # Gwen-TTS-0.6B
│   │   ├── llm.py              # Groq/OpenAI client
│   │   └── orchestrator.py     # Pipeline coordinator
│   │
│   ├── api/                    # API layer
│   │   ├── __init__.py
│   │   ├── websocket.py        # WebSocket handlers
│   │   └── routes.py           # HTTP routes (health, etc.)
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── audio.py            # Audio processing helpers
│       ├── logger.py           # Structured logging
│       └── monitor.py          # Performance monitoring
│
├── tests/
│   ├── __init__.py
│   ├── test_vad.py
│   ├── test_asr.py
│   └── test_tts.py
│
├── assets/                     # Reference audio files for TTS
│   └── voices/
│       └── default.wav
│
├── pyproject.toml              # UV dependencies
├── .env.example
└── README.md
```

---

## 2. 🎯 QUY TẮC CODE

### 2.1 Naming Conventions

```python
# ✅ ĐÚNG
class ASRService:           # PascalCase cho class
    sample_rate: int        # snake_case cho attributes

def process_audio():        # snake_case cho functions
    pass

MAX_BUFFER_SIZE = 1024      # UPPER_SNAKE cho constants
audio_chunk: bytes          # snake_case cho variables

# ❌ SAI
class asrService:           # Không dùng camelCase cho class
def ProcessAudio():         # Không dùng PascalCase cho function
maxBufferSize = 1024        # Không dùng camelCase cho constants
```

### 2.2 Type Hints (BẮT BUỘC)

```python
# ✅ ĐÚNG - Luôn có type hints
from typing import AsyncIterator

async def transcribe(
    audio: bytes,
    language: str | None = None
) -> str:
    """Transcribe audio to text."""
    ...

# ✅ ĐÚNG - Dùng | thay vì Optional (Python 3.10+)
def get_config(key: str) -> str | None:
    ...

# ❌ SAI - Thiếu type hints
async def transcribe(audio, language=None):
    ...
```

### 2.3 Docstrings (Google Style)

```python
async def synthesize(
    text: str,
    voice: str = "default"
) -> tuple[bytes, int]:
    """
    Synthesize text to speech audio.

    Args:
        text: Input text to synthesize
        voice: Voice ID (default: "default")

    Returns:
        Tuple of (audio_bytes, sample_rate)

    Raises:
        TTSError: If synthesis fails

    Example:
        >>> audio, sr = await tts.synthesize("Xin chào")
    """
```

### 2.4 Class Structure

```python
from dataclasses import dataclass
from typing import Protocol

# 1. Protocol/Interface đầu tiên
class IAudioService(Protocol):
    """Interface for audio services."""

    async def process(self, audio: bytes) -> bytes:
        ...

# 2. Dataclass cho data containers
@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 100

# 3. Service class
class ASRService:
    """
    Automatic Speech Recognition using Qwen3-ASR.

    Attributes:
        model: vLLM model instance
        config: ASR configuration
    """

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model: LLM | None = None
        self._logger = get_logger(__name__)

    async def start(self) -> None:
        """Initialize and load model."""
        ...

    async def stop(self) -> None:
        """Cleanup resources."""
        ...

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio to text."""
        ...
```

---

## 3. 🔧 SERVICE PATTERNS

### 3.1 Service Lifecycle

```python
class BaseService:
    """Base class for all services with lifecycle management."""

    def __init__(self) -> None:
        self._started = False
        self._logger = get_logger(self.__class__.__name__)

    async def start(self) -> None:
        """Initialize service. Call before using."""
        if self._started:
            return
        self._logger.info("starting")
        await self._on_start()
        self._started = True
        self._logger.info("started")

    async def stop(self) -> None:
        """Cleanup service. Call when done."""
        if not self._started:
            return
        self._logger.info("stopping")
        await self._on_stop()
        self._started = False
        self._logger.info("stopped")

    async def _on_start(self) -> None:
        """Override in subclass."""
        pass

    async def _on_stop(self) -> None:
        """Override in subclass."""
        pass
```

### 3.2 Error Handling

```python
# ✅ ĐÚNG - Specific exceptions
from voice_agent.core.exceptions import ASRError, TTSError

async def transcribe(self, audio: bytes) -> str:
    try:
        result = await self._model.generate(audio)
        return result.text
    except TimeoutError as e:
        self._logger.error("timeout", audio_size=len(audio))
        raise ASRError("Transcription timeout") from e
    except Exception as e:
        self._logger.error("failed", error=str(e))
        raise ASRError(f"Transcription failed: {e}") from e

# ❌ SAI - Catch-all without re-raise
async def transcribe(self, audio: bytes) -> str:
    try:
        return await self._model.generate(audio)
    except Exception:
        return ""  # Silent failure - BAD!
```

### 3.3 Async Best Practices

```python
import asyncio
from contextlib import asynccontextmanager

# ✅ ĐÚNG - TaskGroup for parallel tasks (Python 3.11+)
async def process_pipeline(audio: bytes) -> PipelineResult:
    async with asyncio.TaskGroup() as tg:
        vad_task = tg.create_task(vad.detect(audio))
        preprocess_task = tg.create_task(preprocess(audio))

    return PipelineResult(
        vad=vad_task.result(),
        audio=preprocess_task.result()
    )

# ✅ ĐÚNG - Context manager for resources
@asynccontextmanager
async def get_model():
    model = await load_model()
    try:
        yield model
    finally:
        await model.close()
```

---

## 4. 📊 LOGGING & MONITORING

### 4.1 Structured Logging

```python
from voice_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ✅ ĐÚNG - Structured với context
logger.info(
    "transcription_complete",
    audio_duration_ms=1500,
    transcript_length=42,
    latency_ms=350,
    language="vi"
)

# ✅ ĐÚNG - Error với context
logger.error(
    "transcription_failed",
    audio_size=len(audio),
    error=str(e),
    error_type=type(e).__name__
)

# ❌ SAI - String formatting
logger.info(f"Transcribed {len(text)} chars in {latency}ms")
```

### 4.2 Performance Monitoring

```python
from voice_agent.utils.monitor import Timer, track_latency

# ✅ ĐÚNG - Track latency với decorator
@track_latency("asr_transcribe")
async def transcribe(self, audio: bytes) -> str:
    ...

# ✅ ĐÚNG - Manual timing
async def process(self, audio: bytes) -> str:
    with Timer() as t:
        result = await self._transcribe(audio)

    logger.info(
        "process_complete",
        latency_ms=t.elapsed_ms,
        input_size=len(audio),
        output_size=len(result)
    )
    return result
```

### 4.3 Required Metrics

Mỗi service PHẢI log các metrics sau:

| Service | Metrics |
|---------|---------|
| **VAD** | `vad_latency_ms`, `speech_detected`, `chunk_count` |
| **ASR** | `asr_latency_ms`, `audio_duration_ms`, `transcript_length` |
| **TTS** | `tts_latency_ms`, `text_length`, `audio_duration_ms` |
| **LLM** | `llm_ttft_ms`, `llm_total_ms`, `token_count` |
| **Pipeline** | `e2e_latency_ms`, `ttfa_ms` (time to first audio) |

---

## 5. ⚡ LATENCY OPTIMIZATION

### 5.1 Streaming First

```python
# ✅ ĐÚNG - Yield results as soon as available
async def generate_response(
    self,
    query: str
) -> AsyncIterator[str]:
    """Stream LLM response sentence by sentence."""
    buffer = ""

    async for token in self._llm.stream(query):
        buffer += token

        # Yield complete sentences immediately
        if any(buffer.endswith(p) for p in ".!?。"):
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()
```

### 5.2 Parallel Processing

```python
# ✅ ĐÚNG - Process TTS while LLM is generating
async def speak_response(self, query: str) -> AsyncIterator[bytes]:
    """Generate and speak response with minimal latency."""

    async for sentence in self._llm.generate(query):
        # Start TTS immediately for each sentence
        audio = await self._tts.synthesize(sentence)
        yield audio
```

### 5.3 Buffer Management

```python
# ✅ ĐÚNG - Efficient audio buffering
from collections import deque

class AudioBuffer:
    """Fixed-size audio buffer with O(1) operations."""

    __slots__ = ("_buffer", "_max_chunks")

    def __init__(self, max_duration_ms: int, chunk_ms: int) -> None:
        self._max_chunks = max_duration_ms // chunk_ms
        self._buffer: deque[bytes] = deque(maxlen=self._max_chunks)

    def add(self, chunk: bytes) -> None:
        self._buffer.append(chunk)

    def get_all(self) -> bytes:
        return b"".join(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
```

---

## 6. 🧪 TESTING

### 6.1 Test Structure

```python
import pytest
from voice_agent.services.asr import ASRService

class TestASRService:
    """Tests for ASR service."""

    @pytest.fixture
    async def service(self):
        """Create service for testing."""
        svc = ASRService(ASRConfig())
        await svc.start()
        yield svc
        await svc.stop()

    async def test_transcribe_vietnamese(self, service: ASRService):
        """Should transcribe Vietnamese audio correctly."""
        audio = load_test_audio("vietnamese_hello.wav")

        result = await service.transcribe(audio)

        assert "xin chào" in result.lower()

    async def test_transcribe_empty_audio(self, service: ASRService):
        """Should handle empty audio gracefully."""
        result = await service.transcribe(b"")

        assert result == ""
```

### 6.2 Performance Tests

```python
import pytest
from voice_agent.utils.monitor import Timer

class TestLatency:
    """Latency benchmark tests."""

    @pytest.mark.benchmark
    async def test_asr_latency(self, asr_service: ASRService):
        """ASR should complete within 500ms for 3s audio."""
        audio = generate_3s_audio()

        with Timer() as t:
            await asr_service.transcribe(audio)

        assert t.elapsed_ms < 500, f"ASR too slow: {t.elapsed_ms}ms"
```

---

## 7. 📦 DEPENDENCIES (pyproject.toml)

```toml
[project]
name = "voice-agent"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    # Web
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "websockets>=15.0",

    # AI/ML
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "qwen-asr[vllm]>=0.1.0",      # Qwen3-ASR with vLLM
    "qwen-tts>=0.1.0",             # Gwen-TTS

    # LLM Client
    "openai>=1.50.0",              # Groq compatible

    # Audio
    "numpy>=2.0.0",
    "soundfile>=0.12.0",

    # Utils
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "structlog>=24.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]
```

---

## 8. ✅ CHECKLIST TRƯỚC KHI COMMIT

- [ ] Type hints đầy đủ cho tất cả functions
- [ ] Docstrings cho public methods
- [ ] Structured logging (không dùng f-string)
- [ ] Error handling với specific exceptions
- [ ] Latency tracking cho mọi operation
- [ ] Tests cho logic mới
- [ ] `ruff check` pass
- [ ] `mypy` pass

---

## 9. 🚀 COMMANDS

```bash
# Setup
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run
uv run python -m voice_agent.main

# Test
uv run pytest tests/ -v

# Lint
uv run ruff check src/
uv run mypy src/

# Format
uv run ruff format src/
```

---

## 10. 📝 GHI CHÚ QUAN TRỌNG

### Về Demo vs Production

| Aspect | Demo (hiện tại) | Production (sau này) |
|--------|-----------------|----------------------|
| Model loading | In-process | Model server (Triton) |
| Session | In-memory | Redis |
| Auth | None | JWT |
| Logging | Console + File | ELK/Loki |

### Về Security (Demo)

- **KHÔNG** commit API keys vào git
- Dùng `.env` cho secrets
- `.env` PHẢI nằm trong `.gitignore`

### Về Performance Targets

| Metric | Target |
|--------|--------|
| VAD latency | < 10ms |
| ASR latency (3s audio) | < 500ms |
| TTS latency (1 sentence) | < 300ms |
| TTFA (Time to First Audio) | < 800ms |
| E2E latency | < 2s |

---

**Last Updated**: 2026-04-06
