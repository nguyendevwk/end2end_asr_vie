"""Constants and configuration defaults.

Inlined as literals to avoid import-time dataclass allocations and
circular coupling with types.py.
"""

# ============== AUDIO ==============

SAMPLE_RATE = 16000

# ============== VAD ==============

VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 700

# ============== MODEL NAMES ==============

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_TTS_MODEL = "g-group-ai-lab/gwen-tts-0.6B"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
