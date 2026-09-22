"""Constants and configuration values.

Defaults are defined in core/types.py dataclasses; these constants re-export
them for backward compatibility and use in non-dataclass contexts.
"""

from voice_agent.core.types import (
    AudioFormat,
    LLMConfig,
    ASRConfig,
    TTSConfig,
    VADConfig,
)

# ============== AUDIO ==============

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK_DURATION_MS = 100

# Re-export for backward compatibility
SAMPLE_RATE = _SAMPLE_RATE
CHANNELS = _CHANNELS
CHUNK_DURATION_MS = _CHUNK_DURATION_MS
BYTES_PER_SAMPLE = 2  # PCM S16LE

# ============== VAD ==============

_vad_defaults = VADConfig()
VAD_THRESHOLD = _vad_defaults.threshold
VAD_MIN_SPEECH_MS = _vad_defaults.min_speech_ms
VAD_MIN_SILENCE_MS = _vad_defaults.min_silence_ms
VAD_SPEECH_PAD_MS = _vad_defaults.speech_pad_ms

# ============== LATENCY TARGETS ==============

TARGET_VAD_LATENCY_MS = 10
TARGET_ASR_LATENCY_MS = 500
TARGET_TTS_LATENCY_MS = 300
TARGET_TTFA_MS = 800
TARGET_E2E_LATENCY_MS = 2000

# ============== BUFFER LIMITS ==============

MAX_AUDIO_BUFFER_MS = 30000  # 30 seconds max
MAX_AUDIO_BUFFER_CHUNKS = MAX_AUDIO_BUFFER_MS // CHUNK_DURATION_MS

# ============== MODEL NAMES ==============

DEFAULT_ASR_MODEL = ASRConfig().model_name
DEFAULT_TTS_MODEL = TTSConfig().model_name
DEFAULT_LLM_MODEL = LLMConfig().model

# ============== API ==============

GROQ_BASE_URL = LLMConfig().base_url
WEBSOCKET_PATH = "/ws/agent"
HEALTH_PATH = "/health"

# ============== LOGGING ==============

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
