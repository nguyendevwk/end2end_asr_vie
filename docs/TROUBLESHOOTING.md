# Xử lý lỗi thường gặp

## 🔧 Installation Issues

### Lỗi: Dependencies không resolve được

**Triệu chứng:**

```
× No solution found when resolving dependencies
```

**Nguyên nhân:** Conflict giữa qwen-asr và qwen-tts về transformers version

**Giải pháp:**

```bash
# 1. Xóa lock file và cache
rm -f uv.lock
rm -rf .venv

# 2. Reinstall
uv sync

# 3. Nếu vẫn lỗi, force cài transformers trước
uv pip install transformers==4.57.6
uv sync
```

### Lỗi: CUDA not available

**Triệu chứng:**

```python
torch.cuda.is_available() == False
```

**Nguyên nhân:** PyTorch không detect GPU

**Giải pháp:**

```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall PyTorch với CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi: `qwen-asr` không tìm thấy

**Triệu chứng:**

```
ModuleNotFoundError: No module named 'qwen_asr'
```

**Nguyên nhân:** Package name là `qwen-asr` không phải `qwen3-asr`

**Giải pháp:**

```bash
# Cài đúng package
uv pip install qwen-asr[vllm]
```

---

## 🚀 Runtime Issues

### Lỗi: CUDA Out of Memory

**Triệu chứng:**

```
CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 3.63 GiB...
```

**Nguyên nhân:** Models chiếm quá nhiều VRAM

**Giải pháp:**

**Option 1: Giảm GPU allocation**

```bash
# .env
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.3
```

**Option 2: Tắt services không cần**

```bash
# .env
TTS_ENABLED=false  # Chỉ test ASR
```

**Option 3: Dùng CPU backend**

```bash
# .env
ASR_DEVICE=cpu
TTS_DEVICE=cpu
```

**Option 4: Clear CUDA cache**

```python
import torch
torch.cuda.empty_cache()
```

**Option 5: Dùng gradient checkpointing**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Lỗi: VAD detection failed - wrong sample count

**Triệu chứng:**

```
ValueError: Provided number of samples is 2048 (Supported values: 256 for 8000 sample rate, 512 for 16000)
```

**Nguyên nhân:** Audio chunk size không đúng 512 samples

**Giải pháp:** ✅ **Đã fix** - VAD service tự động buffer:

```python
# voice_agent/services/vad.py
class _VADIterator:
    def __init__(self):
        self._buffer = b""  # Auto-buffer to 512 samples
```

Nếu vẫn lỗi, check version:

```bash
git pull  # Get latest fix
```

### Lỗi: Web client buffer size invalid

**Triệu chứng:**

```
BaseAudioContext.createScriptProcessor: 1600 is not a valid bufferSize
```

**Nguyên nhân:** Buffer size phải là power of 2

**Giải pháp:** ✅ **Đã fix** - Đổi sang 2048:

```javascript
// web_client/server.py line 504
const bufferSize = 2048;  // Must be 1024, 2048, 4096, etc.
```

### Lỗi: IndentationError

**Triệu chứng:**

```python
IndentationError: unexpected indent at line 263
```

**Nguyên nhân:** Duplicate code hoặc tab/space mixing

**Giải pháp:** ✅ **Đã fix**

```bash
# Get latest version
git pull
```

Hoặc tự fix:

```bash
# Format code
uv run black src/
```

### Lỗi: `check_model_inputs()` missing argument

**Triệu chứng:**

```
TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

**Nguyên nhân:** Transformers version incompatible với qwen-asr

**Giải pháp:**

```bash
# Pin transformers version
uv pip install transformers==4.57.6 --force-reinstall
```

### Lỗi: WebSocket disconnect ngay lập tức

**Triệu chứng:**

```
WebSocket connection failed
Client disconnected
```

**Nguyên nhân:**

- Firewall block port
- CORS policy
- Server chưa sẵn sàng

**Giải pháp:**

1. **Check server running:**

```bash
curl http://localhost:8000/health
```

1. **Check firewall:**

```bash
sudo ufw allow 8000
sudo ufw allow 8080
```

1. **Enable CORS (if needed):**

```python
# voice_agent/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🎙️ Audio Quality Issues

### Transcript rỗng hoặc sai

**Triệu chứng:**

```json
{"type": "transcript", "text": ""}
```

**Nguyên nhân:**

- Audio quá ngắn
- Noise quá nhiều
- Sample rate sai

**Giải pháp:**

1. **Enable preprocessing:**

```bash
ASR_PREPROCESS=true
```

1. **Check audio format:**

```javascript
// Must be PCM 16kHz mono
const sampleRate = 16000;
const channels = 1;
```

1. **Increase speech duration:**

```bash
VAD_MIN_SILENCE_MS=500  # Wait longer for complete speech
```

1. **Test với file audio:**

```python
# Test offline
import asyncio
from voice_agent.services import ASRService

async def test():
    asr = ASRService()
    await asr.start()
    
    with open('test.wav', 'rb') as f:
        # Convert to PCM if needed
        audio = f.read()
    
    result = await asr.transcribe(audio, preprocess=True)
    print(result.text)

asyncio.run(test())
```

### Audio output garbled/distorted

**Triệu chứng:** TTS output không nghe được

**Nguyên nhân:**

- Sample rate mismatch
- Format sai
- Normalization issue

**Giải pháp:**

1. **Check sample rates match:**

```bash
# Server outputs 16kHz (resampled)
# Client expects 16kHz
```

1. **Enable postprocessing:**

```python
# Already enabled in Orchestrator
output_audio = self._postprocessor.process(tts_result.audio)
```

1. **Test TTS riêng:**

```python
import asyncio
from voice_agent.services import TTSService

async def test():
    tts = TTSService()
    await tts.start()
    
    result = await tts.synthesize("Xin chào")
    
    # Save and play
    with open('test_tts.pcm', 'wb') as f:
        f.write(result.audio)
    
    # Play with ffplay
    # ffplay -f s16le -ar 24000 -ac 1 test_tts.pcm

asyncio.run(test())
```

---

## ⚡ Performance Issues

### Latency quá cao (> 3s)

**Triệu chứng:** Response chậm

**Nguyên nhân:**

- Model chậm
- Network to Groq chậm
- Audio processing overhead

**Giải pháp:**

1. **Enable ASR streaming:**

```bash
ASR_STREAMING=true
```

1. **Use vLLM backend (if GPU ≥ 6GB):**

```bash
ASR_BACKEND=vllm
```

1. **Reduce LLM max tokens:**

```bash
GROQ_MAX_TOKENS=100
```

1. **Check network to Groq:**

```bash
curl -w "@-" -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" <<'EOF'
    time_total: %{time_total}s
EOF
```

1. **Profile latency:**

```bash
# Enable debug logs
DEBUG=true

# Check logs for bottleneck
uv run voice-agent 2>&1 | grep latency
```

### RTF > 1.0 (slower than real-time)

**Triệu chứng:** ASR/TTS quá chậm

**Nguyên nhân:**

- CPU backend
- GPU overloaded
- Large batch size

**Giải pháp:**

1. **Ensure GPU is used:**

```python
import torch
print(f"Using GPU: {torch.cuda.is_available()}")
```

1. **Reduce GPU memory for other processes:**

```bash
# Kill other GPU processes
nvidia-smi
kill <PID>
```

1. **Use vLLM:**

```bash
ASR_BACKEND=vllm
```

### Server bị crash/hang

**Triệu chứng:** Server không phản hồi

**Nguyên nhân:**

- Deadlock
- Memory leak
- Uncaught exception

**Giải pháp:**

1. **Check logs:**

```bash
uv run voice-agent 2>&1 | tee server.log
```

1. **Enable debug mode:**

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

1. **Restart với clean state:**

```bash
# Kill all processes
pkill -f voice-agent

# Clear CUDA cache
uv run python -c "import torch; torch.cuda.empty_cache()"

# Restart
uv run voice-agent
```

1. **Check resource limits:**

```bash
# Increase file descriptors
ulimit -n 4096

# Check memory
free -h
```

---

## 🌐 Network Issues

### Groq API timeout

**Triệu chứng:**

```
TimeoutError: Request to Groq API timed out
```

**Nguyên nhân:** Network to Groq cloud chậm/down

**Giải pháp:**

1. **Check Groq status:**

```bash
curl https://status.groq.com/
```

1. **Increase timeout:**

```python
# voice_agent/services/llm.py
timeout = httpx.Timeout(30.0)  # Increase from 10s
```

1. **Retry logic (already implemented):**

```python
# Auto-retry on timeout
```

1. **Test API key:**

```bash
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

### Model download fails

**Triệu chứng:**

```
ConnectionError: Failed to download model from HuggingFace
```

**Nguyên nhân:** Network issue or HF down

**Giải pháp:**

1. **Check HuggingFace status:**

```bash
curl https://status.huggingface.co/
```

1. **Download manually:**

```bash
# Use git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B models/asr/
```

1. **Use mirror (China users):**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 🔍 Debugging Tips

### Enable structured logging

```bash
# .env
DEBUG=true
LOG_LEVEL=DEBUG

# Run with jq for pretty logs
uv run voice-agent 2>&1 | jq .
```

### Check metrics

```bash
# Get real-time metrics
watch -n 1 'curl -s http://localhost:8000/metrics | jq .'
```

### Test services individually

```bash
# Test VAD only
VAD_ENABLED=true
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=false
```

### Memory debugging

```python
import torch
import gc

# Force garbage collection
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()

# Print memory stats
print(torch.cuda.memory_summary())
```

---

## 📞 Support

**Nếu vẫn gặp lỗi:**

1. Check logs với `DEBUG=true`
2. Search issues trong repo
3. Create GitHub issue với:
   - Full error stack trace
   - Environment info (GPU, OS, Python version)
   - Config (.env)
   - Steps to reproduce

**Thông tin hữu ích:**

```bash
# System info
nvidia-smi
python --version
uv --version

# Package versions
uv pip list | grep -E "(torch|transformers|qwen)"

# Config
cat .env
```
