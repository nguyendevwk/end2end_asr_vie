# User Guide

## Quick Start

### 1. Run the server

```bash
cd src
uv run voice-agent
```

Server starts at `http://0.0.0.0:8000`

**Logs on successful startup:**

```
starting_voice_agent    version=0.1.0 host=0.0.0.0 port=8000
loading_vad_model       model='Silero VAD v5'
loading_asr_model      model=Qwen/Qwen3-ASR-0.6B
loading_llm_service    model=llama-3.3-70b-versatile
loading_tts_model       model=g-group-ai-lab/gwen-tts-0.6B
voice_agent_ready       latency_target_ms=2000
```

### 2. Run the web client (new terminal)

```bash
cd src
uv run voice-agent-client
```

Or specify a different port:

```bash
uv run voice-agent-client --port 8080
```

### 3. Access the web interface

Open browser: `http://localhost:8080`

### 4. Test voice conversation

1. Click **"Start Listening"**
2. Speak into the microphone: *"Xin chao"*
3. Wait for response (1-2 seconds)
4. Listen to the audio response

## API Endpoints

### HTTP REST API

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 123.45,
  "services": {
    "vad": "ready",
    "asr": "ready",
    "llm": "ready",
    "tts": "ready"
  }
}
```

#### Metrics

```bash
curl http://localhost:8000/metrics
```

Response:

```json
{
  "sessions": {
    "active": 1,
    "total": 5
  },
  "latency_ms": {
    "vad": 3.2,
    "asr": 450.1,
    "llm": 680.5,
    "tts": 320.8,
    "total": 1454.6
  },
  "performance": {
    "asr_rtf": 0.18,
    "tts_rtf": 0.12
  }
}
```

### WebSocket API

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

#### Sending audio

```javascript
// PCM 16kHz mono, 16-bit
ws.send(JSON.stringify({
  type: 'audio',
  data: base64EncodedPCM,
  sample_rate: 16000,
  channels: 1
}));
```

#### Receiving messages

**State updates:**

```json
{"type": "state", "state": "LISTENING"}
{"type": "state", "state": "PROCESSING"}
{"type": "state", "state": "SPEAKING"}
{"type": "state", "state": "IDLE"}
```

**Transcript:**

```json
{
  "type": "transcript",
  "text": "xin chao",
  "confidence": 0.95
}
```

**Audio response:**

```json
{
  "type": "audio",
  "data": "base64_encoded_pcm",
  "sample_rate": 16000
}
```

**Errors:**

```json
{
  "type": "error",
  "message": "ASR failed: ...",
  "code": "ASR_ERROR"
}
```

## Testing Individual Services

### Test VAD alone

```bash
# .env
VAD_ENABLED=true
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=false
```

```bash
uv run voice-agent
```

VAD will only detect speech and log, without transcribing.

### Test VAD + ASR

```bash
# .env
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false
```

The system will transcribe but will not call LLM/TTS.

### Test full pipeline (default)

```bash
# .env
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=true
TTS_ENABLED=true
```

## Advanced Configuration

### Customize VAD sensitivity

```bash
# .env
VAD_THRESHOLD=0.5          # 0.0-1.0 (default: 0.5)
VAD_MIN_SILENCE_MS=300     # Silence duration to end speech (default: 300)
VAD_SPEECH_PAD_MS=300      # Padding before/after speech (default: 300)
```

**Experiments:**

- `VAD_THRESHOLD=0.3` -> More sensitive (detects more noise)
- `VAD_THRESHOLD=0.7` -> Less sensitive (only detects clear voice)

### Customize ASR

```bash
# .env
ASR_BACKEND=transformers   # or 'vllm'
ASR_GPU_MEMORY=0.5         # 0.0-0.9
ASR_PREPROCESS=true        # Enable preprocessing
ASR_STREAMING=true         # Enable streaming for audio >2s
```

**Backend comparison:**

| Backend        | VRAM | RTF  | Latency | Quality |
|----------------|------|------|---------|---------|
| transformers   | 3GB  | 0.20 | 600ms   | Good    |
| vllm           | 6GB  | 0.08 | 250ms   | Good    |

### Customize LLM

```bash
# .env
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7       # 0.0-2.0 (creativity)
GROQ_MAX_TOKENS=150        # Response length
```

**Model options:**

- `llama-3.3-70b-versatile` - Balanced, fast (default)
- `mixtral-8x7b-32768` - Longer context
- `llama-3.1-8b-instant` - Faster, lower quality

### Customize TTS

```bash
# .env
TTS_SAMPLE_RATE=24000      # Output quality
TTS_SPEED=1.0              # Speaking speed (0.5-2.0)
TTS_GPU_MEMORY=0.4
```

## Monitoring & Logging

### View real-time logs

```bash
# Structured JSON logs
uv run voice-agent 2>&1 | jq .

# Filter by level
uv run voice-agent 2>&1 | jq 'select(.level=="error")'

# Filter by event
uv run voice-agent 2>&1 | jq 'select(.event=="transcript")'
```

### Debug mode

```bash
# .env
DEBUG=true
LOG_LEVEL=DEBUG
```

Debug mode logs:

- Audio chunk sizes
- Model input/output shapes
- Processing timestamps
- Memory usage

### Performance metrics

Metrics are logged after each request:

```json
{
  "event": "request_completed",
  "session_id": "abc123",
  "latency_breakdown_ms": {
    "vad": 3.2,
    "asr": 450.1,
    "llm": 680.5,
    "tts": 320.8,
    "total": 1454.6
  },
  "asr_rtf": 0.18,
  "audio_duration_ms": 2500
}
```

## Debugging

### Test ASR with an audio file

```python
import asyncio
from voice_agent.services import ASRService

async def test_asr():
    asr = ASRService()
    await asr.start()
    
    # Load audio file (PCM 16kHz mono)
    with open('test.pcm', 'rb') as f:
        audio = f.read()
    
    result = await asr.transcribe(audio)
    print(f"Transcript: {result.text}")
    print(f"Confidence: {result.confidence}")

asyncio.run(test_asr())
```

### Test TTS

```python
import asyncio
from voice_agent.services import TTSService

async def test_tts():
    tts = TTSService()
    await tts.start()
    
    result = await tts.synthesize("Xin chao, toi la tro ly AI")
    
    # Save to file
    with open('output.pcm', 'wb') as f:
        f.write(result.audio)
    
    print(f"Generated {len(result.audio)} bytes")

asyncio.run(test_tts())
```

### Test VAD

```python
import asyncio
from voice_agent.services import VADService

async def test_vad():
    vad = VADService()
    await vad.start()
    
    # Test with an audio chunk (512 samples = 32ms @ 16kHz)
    import numpy as np
    audio_chunk = np.random.randn(512).astype(np.float32).tobytes()
    
    is_speech = await vad.detect(audio_chunk)
    print(f"Is speech: {is_speech}")

asyncio.run(test_vad())
```

## Use Cases

### 1. Voice Chatbot

Full pipeline with all services enabled.

### 2. Transcription Service

```bash
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false
```

Only converts speech -> text.

### 3. Text-to-Speech Service

```bash
VAD_ENABLED=false
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=true
```

HTTP endpoint for TTS:

```python
POST /tts
{
  "text": "Xin chao"
}
```

## Performance Tips

### Reduce latency

1. **Enable ASR streaming:**

```bash
ASR_STREAMING=true
```

1. **Use the vLLM backend:**

```bash
ASR_BACKEND=vllm
```

1. **Reduce VAD sensitivity:**

```bash
VAD_MIN_SILENCE_MS=200  # End speech faster
```

### Reduce memory usage

1. **Reduce GPU allocation:**

```bash
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.3
```

1. **Disable preprocessing:**

```bash
ASR_PREPROCESS=false
```

1. **Use CPU for VAD/LLM:**

```bash
VAD_DEVICE=cpu
LLM_ENABLED=false  # Or use a smaller model
```

## Integration Examples

### Using in a Python app

```python
from voice_agent.services import Orchestrator
import asyncio

async def main():
    orch = Orchestrator(session_id="my-session")
    await orch.start()
    
    # Receive audio from some source
    audio_data = ...
    
    # Process
    async for result in orch.process_audio(audio_data):
        if isinstance(result, str):
            print(f"Event: {result}")
        else:
            print(f"Audio: {len(result)} bytes")

asyncio.run(main())
```

### Using with curl

```bash
# Test health
curl http://localhost:8000/health

# WebSocket (requires wscat)
npm install -g wscat
wscat -c ws://localhost:8000/ws
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

**Common errors:**

- **WebSocket disconnect:** Check firewall, CORS settings
- **CUDA OOM:** Reduce GPU_MEMORY, disable services
- **Slow latency:** Enable streaming, check network to Groq
- **Audio garbled:** Check sample rate, format matching

## Next Steps

- Optimize performance: [OPTIMIZATION.md](OPTIMIZATION.md)
- API reference: [API.md](API.md)
- Detailed architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
