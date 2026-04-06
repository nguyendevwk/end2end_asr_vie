# Hướng dẫn sử dụng

## 🚀 Quick Start

### 1. Chạy server

```bash
cd src
uv run voice-agent
```

Server khởi động tại `http://0.0.0.0:8000`

**Logs khi khởi động thành công:**

```
🚀 starting_voice_agent    version=0.1.0 host=0.0.0.0 port=8000
📦 loading_vad_model       model='Silero VAD v5'
🎙️ loading_asr_model      model=Qwen/Qwen3-ASR-0.6B
🗣️ loading_llm_service    model=llama-3.3-70b-versatile
🔊 loading_tts_model       model=g-group-ai-lab/gwen-tts-0.6B
✅ voice_agent_ready       latency_target_ms=2000
```

### 2. Chạy web client (terminal mới)

```bash
cd src
uv run voice-agent-client
```

Hoặc chỉ định port khác:

```bash
uv run voice-agent-client --port 8080
```

### 3. Truy cập web interface

Mở browser: `http://localhost:8080`

### 4. Test voice conversation

1. Click **"Start Listening"**
2. Nói vào microphone: *"Xin chào"*
3. Chờ response (1-2 giây)
4. Nghe audio response

## 📡 API Endpoints

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

#### Kết nối

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

#### Gửi audio

```javascript
// PCM 16kHz mono, 16-bit
ws.send(JSON.stringify({
  type: 'audio',
  data: base64EncodedPCM,
  sample_rate: 16000,
  channels: 1
}));
```

#### Nhận messages

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
  "text": "xin chào",
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

## 🧪 Testing từng service

### Test riêng VAD

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

VAD sẽ chỉ detect speech và log, không transcribe.

### Test VAD + ASR

```bash
# .env
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false
```

Hệ thống sẽ transcribe nhưng không gọi LLM/TTS.

### Test full pipeline (mặc định)

```bash
# .env
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=true
TTS_ENABLED=true
```

## 🛠️ Cấu hình nâng cao

### Tùy chỉnh VAD sensitivity

```bash
# .env
VAD_THRESHOLD=0.5          # 0.0-1.0 (default: 0.5)
VAD_MIN_SILENCE_MS=300     # Silence duration to end speech (default: 300)
VAD_SPEECH_PAD_MS=300      # Padding before/after speech (default: 300)
```

**Thử nghiệm:**

- `VAD_THRESHOLD=0.3` → Nhạy hơn (phát hiện nhiều noise)
- `VAD_THRESHOLD=0.7` → Ít nhạy hơn (chỉ detect giọng rõ)

### Tùy chỉnh ASR

```bash
# .env
ASR_BACKEND=transformers   # hoặc 'vllm'
ASR_GPU_MEMORY=0.5         # 0.0-0.9
ASR_PREPROCESS=true        # Bật preprocessing
ASR_STREAMING=true         # Bật streaming cho audio >2s
```

**Backend comparison:**

| Backend        | VRAM | RTF  | Latency | Quality |
|----------------|------|------|---------|---------|
| transformers   | 3GB  | 0.20 | 600ms   | Good    |
| vllm           | 6GB  | 0.08 | 250ms   | Good    |

### Tùy chỉnh LLM

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

### Tùy chỉnh TTS

```bash
# .env
TTS_SAMPLE_RATE=24000      # Output quality
TTS_SPEED=1.0              # Speaking speed (0.5-2.0)
TTS_GPU_MEMORY=0.4
```

## 📊 Monitoring & Logging

### Xem logs real-time

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

Metrics được log sau mỗi request:

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

## 🔍 Debugging

### Test ASR với file audio

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
    
    result = await tts.synthesize("Xin chào, tôi là trợ lý AI")
    
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
    
    # Test với audio chunk (512 samples = 32ms @ 16kHz)
    import numpy as np
    audio_chunk = np.random.randn(512).astype(np.float32).tobytes()
    
    is_speech = await vad.detect(audio_chunk)
    print(f"Is speech: {is_speech}")

asyncio.run(test_vad())
```

## 🎯 Use Cases

### 1. Voice Chatbot

Full pipeline với tất cả services enabled.

### 2. Transcription Service

```bash
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false
```

Chỉ convert speech → text.

### 3. Text-to-Speech Service

```bash
VAD_ENABLED=false
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=true
```

HTTP endpoint cho TTS:

```python
POST /tts
{
  "text": "Xin chào"
}
```

## ⚡ Performance Tips

### Giảm latency

1. **Enable ASR streaming:**

```bash
ASR_STREAMING=true
```

1. **Dùng vLLM backend:**

```bash
ASR_BACKEND=vllm
```

1. **Giảm VAD sensitivity:**

```bash
VAD_MIN_SILENCE_MS=200  # Kết thúc speech nhanh hơn
```

### Giảm memory usage

1. **Giảm GPU allocation:**

```bash
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.3
```

1. **Disable preprocessing:**

```bash
ASR_PREPROCESS=false
```

1. **Dùng CPU cho VAD/LLM:**

```bash
VAD_DEVICE=cpu
LLM_ENABLED=false  # Hoặc dùng smaller model
```

## 📚 Ví dụ Integration

### Dùng trong Python app

```python
from voice_agent.services import Orchestrator
import asyncio

async def main():
    orch = Orchestrator(session_id="my-session")
    await orch.start()
    
    # Nhận audio từ nguồn nào đó
    audio_data = ...
    
    # Xử lý
    async for result in orch.process_audio(audio_data):
        if isinstance(result, str):
            print(f"Event: {result}")
        else:
            print(f"Audio: {len(result)} bytes")

asyncio.run(main())
```

### Dùng với curl

```bash
# Test health
curl http://localhost:8000/health

# WebSocket (cần wscat)
npm install -g wscat
wscat -c ws://localhost:8000/ws
```

## 🔧 Troubleshooting

Xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md) cho chi tiết.

**Lỗi thường gặp:**

- **WebSocket disconnect:** Check firewall, CORS settings
- **CUDA OOM:** Giảm GPU_MEMORY, disable services
- **Slow latency:** Enable streaming, check network to Groq
- **Audio garbled:** Check sample rate, format matching

## 📖 Next Steps

- Tối ưu performance: [OPTIMIZATION.md](OPTIMIZATION.md)
- API reference: [API.md](API.md)
- Kiến trúc chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md)
