"""Core module exports."""

from voice_agent.core.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TTS_MODEL,
    GROQ_BASE_URL,
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
)
from voice_agent.core.exceptions import (
    ASRError,
    AudioError,
    LLMError,
    PipelineError,
    ServiceError,
    TTSError,
    VADError,
    VoiceAgentError,
)
from voice_agent.core.types import (
    ASRConfig,
    AudioConfig,
    AudioFormat,
    IASRService,
    IAudioProcessor,
    ILLMService,
    ITTSService,
    IVADService,
    LLMConfig,
    PipelineMetrics,
    PipelineState,
    SynthesisResult,
    TranscriptionResult,
    TTSConfig,
    VADConfig,
    VADResult,
)

__all__ = [
    # Constants
    "SAMPLE_RATE",
    "VAD_THRESHOLD",
    "VAD_MIN_SPEECH_MS",
    "VAD_MIN_SILENCE_MS",
    "DEFAULT_ASR_MODEL",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_LLM_MODEL",
    "GROQ_BASE_URL",
    # Exceptions
    "VoiceAgentError",
    "ServiceError",
    "VADError",
    "ASRError",
    "TTSError",
    "LLMError",
    "PipelineError",
    "AudioError",
    # Types
    "AudioConfig",
    "AudioFormat",
    "VADConfig",
    "ASRConfig",
    "TTSConfig",
    "LLMConfig",
    "VADResult",
    "TranscriptionResult",
    "SynthesisResult",
    "PipelineMetrics",
    "PipelineState",
    # Protocols
    "IAudioProcessor",
    "IVADService",
    "IASRService",
    "ITTSService",
    "ILLMService",
]
