# Kiến trúc hệ thống Voice Agent

## 🏗️ Tổng quan

Voice Agent sử dụng kiến trúc pipeline 4 giai đoạn:

```
Audio Input → VAD → ASR → LLM → TTS → Audio Output
```

### Luồng xử lý chi tiết

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

## 🔧 Các thành phần chính

### 1. VAD Service (Voice Activity Detection)

**Công nghệ:** Silero VAD v5 (TorchScript JIT)

**Chức năng:**

- Phát hiện giọng nói trong audio stream
- Tách các đoạn nói liên tục (utterances)
- Buffer tự động cho Silero (yêu cầu 512 samples/chunk)

**Đặc điểm:**

- Model size: ~1.5MB (cực nhẹ)
- Latency: < 5ms
- Accuracy: 95%+
- Không cần GPU

**Cấu hình:**

```python
VAD_THRESHOLD = 0.5      # Ngưỡng phát hiện giọng nói
VAD_MIN_SILENCE = 300    # Im lặng tối thiểu để kết thúc utterance (ms)
VAD_SPEECH_PAD = 300     # Padding trước/sau speech (ms)
```

### 2. ASR Service (Automatic Speech Recognition)

**Công nghệ:** Qwen3-ASR-0.6B

**Backend options:**

- `transformers` (default): 3GB VRAM, RTF ~0.20
- `vllm`: 6GB+ VRAM, RTF ~0.08 (nhanh hơn)

**Chức năng:**

- Chuyển audio sang text (tiếng Việt)
- Preprocessing: DC removal, silence trim, pre-emphasis
- Streaming support cho audio dài (chunks 2s)

**Pipeline xử lý:**

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

**Cấu hình:**

```python
ASR_BACKEND = "transformers"     # hoặc "vllm"
ASR_GPU_MEMORY = 0.5             # Tỷ lệ VRAM sử dụng
ASR_PREPROCESS = True            # Bật preprocessing
ASR_STREAMING = True             # Bật streaming cho audio >2s
```

**Optimization:**

- Preprocessing: +4-8% accuracy trên audio chất lượng thấp
- Streaming: Giảm TTFA (Time-to-First-Audio) từ 800ms → 400ms

### 3. LLM Service

**Công nghệ:** Groq Cloud API (llama-3.3-70b-versatile)

**Chức năng:**

- Xử lý ngữ nghĩa từ transcript
- Tạo response tự nhiên
- Streaming text generation

**System prompt:**

```
Bạn là trợ lý AI thông minh, trả lời ngắn gọn, 
tự nhiên bằng tiếng Việt, dưới 50 từ.
```

**Cấu hình:**

```python
GROQ_API_KEY = "..."             # API key
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 150
```

### 4. TTS Service (Text-to-Speech)

**Công nghệ:** Gwen-TTS 0.6B

**Chức năng:**

- Chuyển text sang audio (tiếng Việt)
- Voice cloning support
- Streaming audio generation

**Pipeline:**

```
Text → Tokenizer → Gwen Model → Mel-Spectrogram → Vocoder → Audio
```

**Cấu hình:**

```python
TTS_SAMPLE_RATE = 24000      # Output sample rate
TTS_SPEED = 1.0              # Speaking speed
```

### 5. Orchestrator (Pipeline Coordinator)

**Chức năng:**

- Điều phối toàn bộ pipeline
- Quản lý state (IDLE → LISTENING → PROCESSING → SPEAKING)
- Buffer quản lý audio stream
- Service isolation (bật/tắt từng service để test)

**States:**

```python
IDLE       # Chờ input
LISTENING  # Đang thu âm
PROCESSING # Đang xử lý (ASR + LLM)
SPEAKING   # Đang phát audio response
```

**Service Isolation:**

```bash
# Test riêng VAD
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

## 📊 Luồng dữ liệu

### WebSocket Protocol

**Client → Server:**

```json
{
  "type": "audio",
  "data": "base64_encoded_pcm",
  "sample_rate": 16000,
  "channels": 1
}
```

**Server → Client:**

```json
// State updates
{"type": "state", "state": "LISTENING"}
{"type": "state", "state": "PROCESSING"}

// Transcript
{"type": "transcript", "text": "xin chào"}

// Audio response
{"type": "audio", "data": "base64_encoded_pcm"}

// Speaking state
{"type": "state", "state": "SPEAKING"}
```

### Audio Format

**Input (từ browser):**

- Format: PCM S16LE
- Sample rate: 16000 Hz
- Channels: Mono
- Bit depth: 16-bit

**Output (tới browser):**

- Format: PCM S16LE
- Sample rate: 16000 Hz (resampled từ 24kHz)
- Channels: Mono
- Bit depth: 16-bit

## 🎯 Performance Metrics

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
- **Total:** ~4GB (vừa khít)

**CPU:**

- VAD: 1-2% per stream
- Audio processing: 2-5%
- WebSocket: 1-3%

**Throughput:**

- Concurrent users: 1-2 (giới hạn bởi GPU)
- Audio streaming: 16kHz = 32KB/s

## 🔒 Error Handling

### Graceful Degradation

1. **ASR fails** → Trả về lỗi, không crash server
2. **LLM timeout** → Fallback: echo transcript
3. **TTS fails** → Trả response dạng text
4. **VAD buffer overflow** → Auto-reset buffer

### Monitoring

- Structlog JSON logging
- Performance metrics (latency, RTF)
- Error tracking với stack traces
- Resource monitoring (VRAM, CPU)

## 🔧 Extensibility

### Thêm service mới

1. Implement interface protocol (`IXXXService`)
2. Add to `services/__init__.py`
3. Register trong `Orchestrator`
4. Update config trong `.env`

### Thay đổi model

**ASR:**

```python
# src/voice_agent/config.py
ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"  # Đổi model khác
```

**TTS:**

```python
TTS_MODEL = "g-group-ai-lab/gwen-tts-0.6B"  # Đổi model khác
```

## 📚 Tham khảo

- [FastAPI](https://fastapi.tiangolo.com/)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Gwen-TTS](https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Groq API](https://console.groq.com/)
