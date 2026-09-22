# Common Troubleshooting

## Installation Issues

### Error: Dependencies cannot be resolved

**Symptoms:**

```
x No solution found when resolving dependencies
```

**Cause:** Conflict between qwen-asr and qwen-tts regarding transformers version

**Solution:**

```bash
# 1. Delete lock file and cache
rm -f uv.lock
rm -rf .venv

# 2. Reinstall
uv sync

# 3. If still failing, force install transformers first
uv pip install transformers==4.57.6
uv sync
```

### Error: CUDA not available

**Symptoms:**

```python
torch.cuda.is_available() == False
```

**Cause:** PyTorch cannot detect GPU

**Solution:**

```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Error: `qwen-asr` not found

**Symptoms:**

```
ModuleNotFoundError: No module named 'qwen_asr'
```

**Cause:** Package name is `qwen-asr` not `qwen3-asr`

**Solution:**

```bash
# Install correct package
uv pip install qwen-asr[vllm]
```

---

## Runtime Issues

### Error: CUDA Out of Memory

**Symptoms:**

```
CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 3.63 GiB...
```

**Cause:** Models consume too much VRAM

**Solution:**

**Option 1: Reduce GPU allocation**

```bash
# .env
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.3
```

**Option 2: Disable unnecessary services**

```bash
# .env
TTS_ENABLED=false  # Only test ASR
```

**Option 3: Use CPU backend**

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

**Option 5: Use gradient checkpointing**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Error: VAD detection failed - wrong sample count

**Symptoms:**

```
ValueError: Provided number of samples is 2048 (Supported values: 256 for 8000 sample rate, 512 for 16000)
```

**Cause:** Audio chunk size is not 512 samples

**Solution:** **Fixed** - VAD service automatically buffers:

```python
# voice_agent/services/vad.py
class _VADIterator:
    def __init__(self):
        self._buffer = b""  # Auto-buffer to 512 samples
```

If still failing, check version:

```bash
git pull  # Get latest fix
```

### Error: Web client buffer size invalid

**Symptoms:**

```
BaseAudioContext.createScriptProcessor: 1600 is not a valid bufferSize
```

**Cause:** Buffer size must be a power of 2

**Solution:** **Fixed** - Changed to 2048:

```javascript
// web_client/server.py line 504
const bufferSize = 2048;  // Must be 1024, 2048, 4096, etc.
```

### Error: IndentationError

**Symptoms:**

```python
IndentationError: unexpected indent at line 263
```

**Cause:** Duplicate code or tab/space mixing

**Solution:** **Fixed**

```bash
# Get latest version
git pull
```

Or fix manually:

```bash
# Format code
uv run black src/
```

### Error: `check_model_inputs()` missing argument

**Symptoms:**

```
TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

**Cause:** Transformers version incompatible with qwen-asr

**Solution:**

```bash
# Pin transformers version
uv pip install transformers==4.57.6 --force-reinstall
```

### Error: WebSocket disconnects immediately

**Symptoms:**

```
WebSocket connection failed
Client disconnected
```

**Cause:**

- Firewall blocks port
- CORS policy
- Server not ready

**Solution:**

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

## Audio Quality Issues

### Transcript empty or incorrect

**Symptoms:**

```json
{"type": "transcript", "text": ""}
```

**Cause:**

- Audio too short
- Too much noise
- Wrong sample rate

**Solution:**

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

1. **Test with audio file:**

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

**Symptoms:** TTS output is inaudible

**Cause:**

- Sample rate mismatch
- Wrong format
- Normalization issue

**Solution:**

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

1. **Test TTS separately:**

```python
import asyncio
from voice_agent.services import TTSService

async def test():
    tts = TTSService()
    await tts.start()
    
    result = await tts.synthesize("Hello")
    
    # Save and play
    with open('test_tts.pcm', 'wb') as f:
        f.write(result.audio)
    
    # Play with ffplay
    # ffplay -f s16le -ar 24000 -ac 1 test_tts.pcm

asyncio.run(test())
```

---

## Performance Issues

### Latency too high (> 3s)

**Symptoms:** Slow response

**Cause:**

- Slow model
- Slow network to Groq
- Audio processing overhead

**Solution:**

1. **Enable ASR streaming:**

```bash
ASR_STREAMING=true
```

1. **Use vLLM backend (if GPU >= 6GB):**

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

**Symptoms:** ASR/TTS too slow

**Cause:**

- CPU backend
- GPU overloaded
- Large batch size

**Solution:**

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

### Server crashes/hangs

**Symptoms:** Server not responding

**Cause:**

- Deadlock
- Memory leak
- Uncaught exception

**Solution:**

1. **Check logs:**

```bash
uv run voice-agent 2>&1 | tee server.log
```

1. **Enable debug mode:**

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

1. **Restart with clean state:**

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

## Network Issues

### Groq API timeout

**Symptoms:**

```
TimeoutError: Request to Groq API timed out
```

**Cause:** Network to Groq cloud slow/down

**Solution:**

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

**Symptoms:**

```
ConnectionError: Failed to download model from HuggingFace
```

**Cause:** Network issue or HF down

**Solution:**

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

## Debugging Tips

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

## Support

**If still encountering errors:**

1. Check logs with `DEBUG=true`
2. Search issues in the repo
3. Create GitHub issue with:
   - Full error stack trace
   - Environment info (GPU, OS, Python version)
   - Config (.env)
   - Steps to reproduce

**Useful information:**

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
