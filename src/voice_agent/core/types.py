"""Core type definitions and data structures."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Protocol


# ============== ENUMS ==============


class PipelineState(Enum):
    """State of the conversation pipeline."""

    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()


class AudioFormat(Enum):
    """Supported audio formats."""

    PCM_S16LE = "pcm_s16le"  # 16-bit signed little-endian
    PCM_F32LE = "pcm_f32le"  # 32-bit float little-endian


# ============== DATACLASSES ==============


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    format: AudioFormat = AudioFormat.PCM_S16LE

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return self.sample_rate * self.chunk_duration_ms // 1000

    @property
    def chunk_bytes(self) -> int:
        """Number of bytes per chunk (for PCM_S16LE)."""
        return self.chunk_samples * 2 * self.channels


@dataclass(frozen=True, slots=True)
class VADConfig:
    """VAD configuration."""

    threshold: float = 0.5
    min_speech_ms: int = 1000
    min_silence_ms: int = 1000
    speech_pad_ms: int = 30


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """ASR configuration."""

    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    language: str | None = None  # None = auto-detect
    max_new_tokens: int = 256
    gpu_memory_utilization: float = 0.3  # Reduced for smaller GPUs
    backend: str = "transformers"  # "transformers" (default, less VRAM) or "vllm" (faster)
    streaming: bool = True  # Enable streaming transcription for lower latency
    preprocess: bool = True  # Enable audio preprocessing for better accuracy


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """TTS configuration."""

    model_name: str = "g-group-ai-lab/gwen-tts-0.6B"
    voice_ref_audio: str | None = None  # Path to reference audio for voice cloning
    voice_ref_text: str | None = None  # Transcript of reference audio
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.9
    stream_chunk_ms: int = 100  # Audio chunk duration for streaming (ms)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM configuration."""

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = (
        "Bạn là trợ lý ảo thông minh, thân thiện, hỗ trợ người dùng bằng tiếng Việt. "
        "Trả lời ngắn gọn, tự nhiên như đang nói chuyện."
    )


@dataclass(slots=True)
class VADResult:
    """VAD detection result."""

    is_speech: bool
    confidence: float
    event: str | None = None  # "start", "end", or None


@dataclass(slots=True)
class TranscriptionResult:
    """ASR transcription result."""

    text: str
    language: str
    latency_ms: float

    # Streaming fields
    is_final: bool = True           # True nếu đây là kết quả cuối cùng
    stability: float = 1.0          # 0.0-1.0, độ ổn định của transcript
    confidence: float = 0.0         # 0.0-1.0, độ tin cậy
    result_end_offset_ms: float = 0.0  # Vị trí kết thúc trong audio stream


@dataclass(slots=True)
class SynthesisResult:
    """TTS synthesis result."""

    audio: bytes
    sample_rate: int
    duration_ms: float
    latency_ms: float


@dataclass(slots=True)
class PipelineMetrics:
    """Metrics for a single pipeline execution."""

    session_id: str
    turn_id: int
    audio_duration_ms: float = 0.0
    vad_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_latency_ms: float = 0.0
    ttfa_ms: float = 0.0  # Time to first audio
    e2e_latency_ms: float = 0.0


# ============== PROTOCOLS ==============


class IAudioProcessor(Protocol):
    """Protocol for audio processing services."""

    async def process(self, audio: bytes) -> bytes:
        """Process audio chunk."""
        ...


class IVADService(Protocol):
    """Protocol for VAD service."""

    async def detect(self, audio: bytes) -> VADResult:
        """Detect voice activity in audio chunk."""
        ...

    def reset(self) -> None:
        """Reset VAD state for new utterance."""
        ...


class IASRService(Protocol):
    """Protocol for ASR service."""

    async def transcribe(self, audio: bytes, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio to text."""
        ...


class ITTSService(Protocol):
    """Protocol for TTS service."""

    async def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize text to speech."""
        ...


class ILLMService(Protocol):
    """Protocol for LLM service."""

    async def generate(self, query: str, history: list[dict[str, str]]) -> str:
        """Generate response for query."""
        ...

    async def stream(
        self, query: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...
