"""Custom exceptions for Voice Agent."""


class VoiceAgentError(Exception):
    """Base exception for all voice agent errors."""


class ServiceError(VoiceAgentError):
    """Error in a service component."""


class VADError(ServiceError):
    """Error in VAD service."""


class ASRError(ServiceError):
    """Error in ASR service."""


class TTSError(ServiceError):
    """Error in TTS service."""


class LLMError(ServiceError):
    """Error in LLM service."""


class PipelineError(VoiceAgentError):
    """Error in pipeline orchestration."""


class AudioError(VoiceAgentError):
    """Error in audio processing."""


class ConfigError(VoiceAgentError):
    """Error in configuration."""
