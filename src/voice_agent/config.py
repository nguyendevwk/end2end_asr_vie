"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
        env_file=".env",
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


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
