"""Custom exceptions for Voice Agent."""


class VoiceAgentError(Exception):
    """Base exception for all voice agent errors."""

    pass


class ServiceError(VoiceAgentError):
    """Error in a service component."""

    pass


class VADError(ServiceError):
    """Error in VAD service."""

    pass


class ASRError(ServiceError):
    """Error in ASR service."""

    pass


class TTSError(ServiceError):
    """Error in TTS service."""

    pass


class LLMError(ServiceError):
    """Error in LLM service."""

    pass


class PipelineError(VoiceAgentError):
    """Error in pipeline orchestration."""

    pass


class AudioError(VoiceAgentError):
    """Error in audio processing."""

    pass


class ConfigError(VoiceAgentError):
    """Error in configuration."""

    pass


class ConnectionError(VoiceAgentError):
    """Error in WebSocket connection."""

    pass
