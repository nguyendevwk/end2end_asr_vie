# Voice Agent System Architecture

## Overview

Voice Agent uses a 4-stage pipeline architecture:

```
Audio Input → VAD → ASR → LLM → TTS → Audio Output
```

### Detailed Processing Flow

```
┌─────────────┐
│ Web Client  │ Browser (Microphone)
└──────┬──────┘
       │ WebSocket (PCM 16kHz mono)
       ▼
┌─────────────────────────────────────────┐
│         FastAPI Server                  │
│  ┌─────────────────────────────────┐   │
│  │    Orchestrator (Coordinator)   │   │
│  └─────────────────────────────────┘   │
│         │         │         │           │
│    ┌────▼───┐ ┌──▼───┐ ┌───▼────┐     │
│    │  VAD   │ │ ASR  │ │  TTS   │     │
│    └────┬───┘ └──┬───┘ └───┬────┘     │
│         │        │          │           │
└─────────┼────────┼──────────┼───────────┘
          │        │          │
     ┌────▼───┐ ┌─▼────┐  ┌──▼─────┐
     │ Silero │ │Qwen3 │  │ Gwen   │
     │  VAD   │ │ ASR  │  │  TTS   │
     └────────┘ └──┬───┘  └────────┘
                   │
               ┌───▼────┐
               │  LLM   │ Groq (cloud)
               └────────┘
```

## Main Components

### 1. VAD Service (Voice Activity Detection)

**Technology:** Silero VAD v5 (TorchScript JIT)

**Functions:**

- Detect voice in audio stream
- Segment continuous speech (utterances)
- Auto-buffering for Silero (requires 512 samples/chunk)

**Characteristics:**

- Model size: ~1.5MB (very lightweight)
- Latency: < 5ms
- Accuracy: 95%+
- No GPU required

**Configuration:**

```python
VAD_THRESHOLD = 0.5      # Voice detection threshold
VAD_MIN_SILENCE = 300    # Minimum silence to end utterance (ms)
VAD_SPEECH_PAD = 300     # Padding before/after speech (ms)
```

### 2. ASR Service (Automatic Speech Recognition)

**Technology:** Qwen3-ASR-0.6B

**Backend options:**

- `transformers` (default): 3GB VRAM, RTF ~0.20
- `vllm`: 6GB+ VRAM, RTF ~0.08 (faster)

**Functions:**

- Convert audio to text (Vietnamese)
- Preprocessing: DC removal, silence trim, pre-emphasis
- Streaming support for long audio (2s chunks)

**Processing Pipeline:**

```
Raw Audio (PCM)
  ↓
ASRPreprocessor:
  - Remove DC offset
  - Trim silence
  - Normalize to -20dB
  - Pre-emphasis filter (0.97)
  ↓
Qwen3-ASR Model:
  - Feature extraction
  - Encoder-decoder inference
  ↓
Text Cleaning:
  - Remove special tokens
  - Strip whitespace
  ↓
Transcript
```

**Configuration:**

```python
ASR_BACKEND = "transformers"     # or "vllm"
ASR_GPU_MEMORY = 0.5             # VRAM usage ratio
ASR_PREPROCESS = True            # Enable preprocessing
ASR_STREAMING = True             # Enable streaming for audio >2s
```

**Optimization:**

- Preprocessing: +4-8% accuracy on low quality audio
- Streaming: Reduce TTFA (Time-to-First-Audio) from 800ms to 400ms

### 3. LLM Service

**Technology:** Groq Cloud API (llama-3.3-70b-versatile)

**Functions:**

- Process semantics from transcript
- Generate natural responses
- Streaming text generation

**System prompt:**

```
You are a smart AI assistant, reply briefly,
naturally in Vietnamese, under 50 words.
```

**Configuration:**

```python
GROQ_API_KEY = "..."             # API key
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 150
```

### 4. TTS Service (Text-to-Speech)

**Technology:** Gwen-TTS 0.6B

**Functions:**

- Convert text to audio (Vietnamese)
- Voice cloning support
- Streaming audio generation

**Pipeline:**

```
Text → Tokenizer → Gwen Model → Mel-Spectrogram → Vocoder → Audio
```

**Configuration:**

```python
TTS_SAMPLE_RATE = 24000      # Output sample rate
TTS_SPEED = 1.0              # Speaking speed
```

### 5. Orchestrator (Pipeline Coordinator)

**Functions:**

- Coordinate entire pipeline
- Manage state (IDLE → LISTENING → PROCESSING → SPEAKING)
- Buffer audio stream management
- Service isolation (enable/disable each service for testing)

**States:**

```python
IDLE       # Waiting for input
LISTENING  # Recording
PROCESSING # Processing (ASR + LLM)
SPEAKING   # Playing audio response
```

**Service Isolation:**

```bash
# Test VAD only
VAD_ENABLED=true
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=false

# Test VAD + ASR
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false

# Full pipeline
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=true
TTS_ENABLED=true
```

## Data Flow

### WebSocket Protocol

**Client to Server:**

```json
{
  "type": "audio",
  "data": "base64_encoded_pcm",
  "sample_rate": 16000,
  "channels": 1
}
```

**Server to Client:**

```json
// State updates
{"type": "state", "state": "LISTENING"}
{"type": "state", "state": "PROCESSING"}

// Transcript
{"type": "transcript", "text": "xin chao"}

// Audio response
{"type": "audio", "data": "base64_encoded_pcm"}

// Speaking state
{"type": "state", "state": "SPEAKING"}
```

### Audio Format

**Input (from browser):**

- Format: PCM S16LE
- Sample rate: 16000 Hz
- Channels: Mono
- Bit depth: 16-bit

**Output (to browser):**

- Format: PCM S16LE
- Sample rate: 16000 Hz (resampled from 24kHz)
- Channels: Mono
- Bit depth: 16-bit

## Performance Metrics

### Latency Breakdown (target)

```
User stops speaking
  ↓ VAD detection: 300-500ms
Speech end detected
  ↓ ASR: 200-400ms (streaming) / 600-1000ms (standard)
Transcript ready
  ↓ LLM: 400-800ms (streaming)
Response text ready
  ↓ TTS: 200-500ms (first chunk)
Audio playback starts
────────────────────────
Total: 1.1 - 2.2s
```

### Resource Usage

**GPU (GTX 1650 Ti 4GB):**

- ASR (transformers): ~2.5GB
- TTS: ~1.5GB
- VAD: 0 (CPU only)
- **Total:** ~4GB (just fits)

**CPU:**

- VAD: 1-2% per stream
- Audio processing: 2-5%
- WebSocket: 1-3%

**Throughput:**

- Concurrent users: 1-2 (limited by GPU)
- Audio streaming: 16kHz = 32KB/s

## Error Handling

### Graceful Degradation

1. **ASR fails** → Return error, do not crash server
2. **LLM timeout** → Fallback: echo transcript
3. **TTS fails** → Return response as text
4. **VAD buffer overflow** → Auto-reset buffer

### Monitoring

- Structlog JSON logging
- Performance metrics (latency, RTF)
- Error tracking with stack traces
- Resource monitoring (VRAM, CPU)

## Extensibility

### Adding a New Service

1. Implement interface protocol (`IXXXService`)
2. Add to `services/__init__.py`
3. Register in `Orchestrator`
4. Update config in `.env`

### Changing Models

**ASR:**

```python
# src/voice_agent/config.py
ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"  # Change to a different model
```

**TTS:**

```python
TTS_MODEL = "g-group-ai-lab/gwen-tts-0.6B"  # Change to a different model
```

## References

- [FastAPI](https://fastapi.tiangolo.com/)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Gwen-TTS](https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Groq API](https://console.groq.com/)
