# API Reference

## HTTP REST API

Base URL: `http://localhost:8000`

### Health Check

Check server health and service status.

**Endpoint:** `GET /health`

**Response:**

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

**Service states:**

- `ready`: Service loaded and ready
- `disabled`: Service disabled in config
- `error`: Service failed to load

**Example:**

```bash
curl http://localhost:8000/health
```

### Metrics

Get performance metrics.

**Endpoint:** `GET /metrics`

**Response:**

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
  },
  "memory": {
    "gpu_allocated_mb": 2048.5,
    "gpu_cached_mb": 512.3
  }
}
```

**Example:**

```bash
curl http://localhost:8000/metrics | jq .
```

---

## WebSocket API

WebSocket endpoint for real-time voice conversation.

**URL:** `ws://localhost:8000/ws`

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleMessage(message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

### Message Types

#### Client -> Server

##### 1. Audio Data

Send audio chunk for processing.

**Format:**

```json
{
  "type": "audio",
  "data": "base64_encoded_pcm_data",
  "sample_rate": 16000,
  "channels": 1
}
```

**Audio specs:**

- Format: PCM S16LE (16-bit signed little-endian)
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Encoding: Base64

**Example (JavaScript):**

```javascript
// From AudioWorklet or ScriptProcessor
const audioData = new Int16Array(samples);
const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));

ws.send(JSON.stringify({
  type: 'audio',
  data: base64,
  sample_rate: 16000,
  channels: 1
}));
```

##### 2. Control Messages (Future)

```json
{
  "type": "stop",
  "reason": "user_cancelled"
}
```

#### Server -> Client

##### 1. State Updates

Pipeline state changes.

**Format:**

```json
{
  "type": "state",
  "state": "LISTENING"
}
```

**States:**

- `IDLE`: Waiting for input
- `LISTENING`: Recording audio
- `PROCESSING`: Running ASR + LLM
- `SPEAKING`: Playing TTS response

**Example:**

```javascript
if (message.type === 'state') {
  updateUI(message.state);
}
```

##### 2. Transcript

ASR output.

**Format:**

```json
{
  "type": "transcript",
  "text": "xin chao",
  "confidence": 0.95,
  "language": "vi"
}
```

**Fields:**

- `text`: Transcribed text
- `confidence`: 0.0-1.0 (future feature)
- `language`: Language code (always "vi")

**Example:**

```javascript
if (message.type === 'transcript') {
  displayTranscript(message.text);
}
```

##### 3. Audio Response

TTS output audio.

**Format:**

```json
{
  "type": "audio",
  "data": "base64_encoded_pcm_data",
  "sample_rate": 16000,
  "duration_ms": 2500
}
```

**Audio specs:**

- Format: PCM S16LE
- Sample rate: 16000 Hz (resampled from 24kHz)
- Channels: 1 (mono)
- Encoding: Base64

**Example:**

```javascript
if (message.type === 'audio') {
  const audioData = base64ToFloat32(message.data);
  playAudio(audioData);
}

function base64ToFloat32(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);

  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;  // Normalize to [-1, 1]
  }

  return float32;
}
```

##### 4. Error Messages

Error notifications.

**Format:**

```json
{
  "type": "error",
  "message": "ASR failed: CUDA out of memory",
  "code": "ASR_ERROR",
  "severity": "error"
}
```

**Error codes:**

- `VAD_ERROR`: VAD detection failed
- `ASR_ERROR`: Transcription failed
- `LLM_ERROR`: LLM generation failed
- `TTS_ERROR`: Speech synthesis failed
- `UNKNOWN_ERROR`: Unexpected error

**Severity:**

- `warning`: Recoverable, retryable
- `error`: Failed, not retryable

**Example:**

```javascript
if (message.type === 'error') {
  showError(message.message);
  if (message.severity === 'error') {
    resetConversation();
  }
}
```

---

## Service APIs (Python)

Internal service APIs for programmatic use.

### VAD Service

```python
from voice_agent.services import VADService

vad = VADService()
await vad.start()

# Detect speech in audio chunk
is_speech = await vad.detect(audio_chunk)

# Process audio stream (yields complete utterances)
async for utterance in vad.process_stream(audio_stream):
    print(f"Speech detected: {len(utterance)} bytes")
```

**Methods:**

#### `start() -> None`

Initialize VAD model.

#### `detect(audio: bytes) -> bool`

Detect speech in audio chunk.

**Args:**

- `audio`: PCM audio data (any length, will be buffered to 512 samples)

**Returns:** `True` if speech detected

#### `process_stream(audio: AsyncIterator[bytes]) -> AsyncIterator[bytes]`

Process audio stream, yield complete utterances.

**Args:**

- `audio`: Audio stream (chunks of any size)

**Yields:** Complete utterances (VAD-segmented speech)

### ASR Service

```python
from voice_agent.services import ASRService

asr = ASRService()
await asr.start()

# Single transcription
result = await asr.transcribe(audio, preprocess=True)
print(result.text)

# Streaming transcription (for long audio)
async for partial_result in asr.transcribe_stream(audio, chunk_duration_ms=2000):
    print(partial_result.text)
```

**Methods:**

#### `start() -> None`

Load ASR model (transformers or vLLM backend).

#### `transcribe(audio: bytes, preprocess: bool = True) -> ASRResult`

Transcribe audio to text.

**Args:**

- `audio`: PCM 16kHz mono audio
- `preprocess`: Enable preprocessing (DC removal, normalization, etc.)

**Returns:**

```python
@dataclass
class ASRResult:
    text: str              # Transcribed text
    confidence: float      # 0.0-1.0 (future)
    language: str          # "vi"
    duration_ms: float     # Processing time
```

#### `transcribe_stream(audio: bytes, chunk_duration_ms: int = 2000) -> AsyncIterator[ASRResult]`

Streaming transcription for long audio.

**Args:**

- `audio`: Complete audio data
- `chunk_duration_ms`: Chunk size (default 2000ms)

**Yields:** Partial ASR results

### LLM Service

```python
from voice_agent.services import LLMService

llm = LLMService()
await llm.start()

# Streaming generation
async for chunk in llm.generate_stream("Xin chao"):
    print(chunk, end='', flush=True)
```

**Methods:**

#### `start() -> None`

Initialize Groq client.

#### `generate(prompt: str) -> str`

Generate response (non-streaming).

**Args:**

- `prompt`: User message

**Returns:** Complete response text

#### `generate_stream(prompt: str) -> AsyncIterator[str]`

Generate response with streaming.

**Args:**

- `prompt`: User message

**Yields:** Text chunks as they arrive

### TTS Service

```python
from voice_agent.services import TTSService

tts = TTSService()
await tts.start()

# Synthesize speech
result = await tts.synthesize("Xin chao, toi la tro ly AI")

# Save to file
with open('output.pcm', 'wb') as f:
    f.write(result.audio)
```

**Methods:**

#### `start() -> None`

Load TTS model.

#### `synthesize(text: str, voice_id: str = None) -> TTSResult`

Convert text to speech.

**Args:**

- `text`: Text to synthesize
- `voice_id`: Voice ID (future feature)

**Returns:**

```python
@dataclass
class TTSResult:
    audio: bytes           # PCM audio data (24kHz mono)
    sample_rate: int       # 24000
    duration_ms: float     # Audio duration
```

### Orchestrator

```python
from voice_agent.services import Orchestrator

orch = Orchestrator(session_id="user-123")
await orch.start()

# Process audio (full pipeline: VAD -> ASR -> LLM -> TTS)
async for result in orch.process_audio(audio_chunk):
    if isinstance(result, str):
        print(f"Event: {result}")  # State updates, transcripts
    else:
        print(f"Audio: {len(result)} bytes")  # TTS output
```

**Methods:**

#### `__init__(session_id: str)`

Create orchestrator for a session.

#### `start() -> None`

Initialize all enabled services.

#### `process_audio(audio: bytes) -> AsyncIterator[bytes | str]`

Process audio through full pipeline.

**Args:**

- `audio`: PCM audio chunk

**Yields:**

- `str`: Events (state changes, transcripts)
- `bytes`: Audio responses

---

## Configuration API

Environment variables (`.env` file).

### Server

```bash
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
```

### VAD

```bash
VAD_ENABLED=true
VAD_THRESHOLD=0.5
VAD_MIN_SILENCE_MS=300
VAD_SPEECH_PAD_MS=300
```

### ASR

```bash
ASR_ENABLED=true
ASR_MODEL=Qwen/Qwen3-ASR-0.6B
ASR_BACKEND=transformers      # or 'vllm'
ASR_DEVICE=cuda               # or 'cpu'
ASR_GPU_MEMORY=0.5
ASR_PREPROCESS=true
ASR_STREAMING=true
```

### LLM

```bash
LLM_ENABLED=true
GROQ_API_KEY=gsk_xxx...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=150
```

### TTS

```bash
TTS_ENABLED=true
TTS_MODEL=g-group-ai-lab/gwen-tts-0.6B
TTS_DEVICE=cuda
TTS_GPU_MEMORY=0.4
TTS_SAMPLE_RATE=24000
TTS_SPEED=1.0
```

See [.env.example](../src/.env.example) for full reference.

---

## Type Definitions

### Audio Format

```python
# PCM S16LE format
SampleRate = 16000  # Hz
Channels = 1        # Mono
BitDepth = 16       # bits
Encoding = "little-endian"
```

### Pipeline States

```python
from enum import Enum

class PipelineState(Enum):
    IDLE = "IDLE"              # Waiting for input
    LISTENING = "LISTENING"     # Recording
    PROCESSING = "PROCESSING"   # ASR + LLM
    SPEAKING = "SPEAKING"       # TTS playback
```

### Error Codes

```python
class ErrorCode(Enum):
    VAD_ERROR = "VAD_ERROR"
    ASR_ERROR = "ASR_ERROR"
    LLM_ERROR = "LLM_ERROR"
    TTS_ERROR = "TTS_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
```

---

## Examples

### Full client example (JavaScript)

See [web_client/server.py](../src/web_client/server.py) for complete implementation.

### Python client example

```python
import asyncio
import websockets
import json
import base64

async def voice_client():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as ws:
        # Send audio
        with open('input.pcm', 'rb') as f:
            audio_data = f.read()

        message = {
            'type': 'audio',
            'data': base64.b64encode(audio_data).decode(),
            'sample_rate': 16000,
            'channels': 1
        }
        await ws.send(json.dumps(message))

        # Receive responses
        async for msg in ws:
            data = json.loads(msg)

            if data['type'] == 'transcript':
                print(f"Transcript: {data['text']}")

            elif data['type'] == 'audio':
                audio = base64.b64decode(data['data'])
                with open('output.pcm', 'wb') as f:
                    f.write(audio)
                print("Audio saved")

            elif data['type'] == 'error':
                print(f"Error: {data['message']}")
                break

asyncio.run(voice_client())
```
