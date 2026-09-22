"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_agent.core import (
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TTS_MODEL,
    GROQ_BASE_URL,
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables are prefixed with VOICE_AGENT_ or use standard names.
    Example: VOICE_AGENT_DEBUG=true or GROQ_API_KEY=xxx
    """

    model_config = SettingsConfigDict(
        # Absolute path: works whether CWD is repo root or src/
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === Server ===
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=False, description="JSON log output")

    # === Audio ===
    sample_rate: int = Field(default=SAMPLE_RATE, description="Audio sample rate")
    chunk_duration_ms: int = Field(default=100, description="Audio chunk duration in ms")

    # === VAD ===
    vad_enabled: bool = Field(default=True, description="Enable VAD service")
    vad_threshold: float = Field(default=VAD_THRESHOLD, description="VAD threshold")
    vad_min_speech_ms: int = Field(default=VAD_MIN_SPEECH_MS, description="Min speech duration")
    vad_min_silence_ms: int = Field(default=VAD_MIN_SILENCE_MS, description="Min silence for end")

    # === ASR ===
    asr_enabled: bool = Field(default=True, description="Enable ASR service")
    asr_model: str = Field(default=DEFAULT_ASR_MODEL, description="ASR model name")
    asr_language: str | None = Field(default=None, description="ASR language (None=auto)")
    asr_gpu_memory: float = Field(default=0.3, description="GPU memory utilization for ASR")
    asr_backend: str = Field(
        default="transformers",
        description="ASR backend: 'transformers' (less VRAM) or 'vllm' (faster)",
    )
    asr_streaming: bool = Field(default=False, description="Enable streaming ASR for lower latency")
    asr_preprocess: bool = Field(default=True, description="Enable audio preprocessing for ASR")

    # === TTS ===
    tts_enabled: bool = Field(default=True, description="Enable TTS service")
    tts_model: str = Field(default=DEFAULT_TTS_MODEL, description="TTS model name")
    tts_voice_ref_audio: str | None = Field(default=None, description="Reference audio for voice cloning")
    tts_voice_ref_text: str | None = Field(default=None, description="Reference audio transcript")
    tts_temperature: float = Field(default=0.3, description="TTS temperature")

    # === LLM ===
    llm_enabled: bool = Field(default=True, description="Enable LLM service")
    groq_api_key: str = Field(default="", description="Groq API key", alias="GROQ_API_KEY")
    llm_base_url: str = Field(default=GROQ_BASE_URL, description="LLM API base URL")
    llm_model: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model name")
    llm_temperature: float = Field(default=0.7, description="LLM temperature")
    llm_max_tokens: int = Field(default=512, description="Max response tokens")
    llm_system_prompt: str = Field(
        default=(
            "Bạn là trợ lý ảo thông minh, thân thiện, hỗ trợ người dùng bằng tiếng Việt. "
            "Trả lời ngắn gọn, tự nhiên như đang nói chuyện."
        ),
        description="System prompt",
    )

    # === Text task (ASR -> TTS core, LLM as pluggable middleware) ===
    text_task: str = Field(
        default="llm",
        description="Text task: 'llm', 'passthrough', 'router', or custom registered name",
    )
    text_task_prefix: str = Field(
        default="Bạn nói: ",
        description="Prefix for the passthrough task",
    )

    # === Production: concurrency & backpressure ===
    max_ccu: int = Field(default=50, description="Max concurrent WebSocket connections")
    max_inflight_infer: int = Field(
        default=4, description="Max concurrent GPU inferences (ASR/TTS admission)"
    )

    # === Production: timeouts & retries ===
    asr_timeout_s: float = Field(default=15.0, description="Per-attempt ASR timeout")
    tts_timeout_s: float = Field(default=20.0, description="Per-sentence TTS timeout")
    llm_timeout_s: float = Field(default=30.0, description="LLM task total timeout")
    llm_first_token_timeout_s: float = Field(default=10.0, description="LLM first chunk timeout")
    max_retries: int = Field(default=1, description="Extra attempts after the first try")

    # === Production: heartbeat & sessions ===
    ws_ping_interval_s: float = Field(default=20.0, description="Server heartbeat interval")
    ws_ping_timeout_s: float = Field(default=60.0, description="Close after silence")
    session_ttl_s: float = Field(default=300.0, description="Reconnect resume window")

    @field_validator("asr_gpu_memory")
    @classmethod
    def validate_gpu_memory(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"asr_gpu_memory must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("max_ccu", "max_inflight_infer")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("session_ttl_s")
    @classmethod
    def validate_ttl(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"session_ttl_s must be positive, got {v}")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
