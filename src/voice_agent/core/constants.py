"""Constants and configuration values."""

# ============== AUDIO ==============

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 100
BYTES_PER_SAMPLE = 2  # PCM S16LE

# ============== VAD ==============

VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 1000
VAD_SPEECH_PAD_MS = 30

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

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_TTS_MODEL = "g-group-ai-lab/gwen-tts-0.6B"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# ============== API ==============

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
WEBSOCKET_PATH = "/ws/agent"
HEALTH_PATH = "/health"

# ============== LOGGING ==============

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
