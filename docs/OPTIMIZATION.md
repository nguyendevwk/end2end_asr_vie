# Performance Optimization

## Performance Goals

| Metric | Target | Actual (GTX 1650 Ti 4GB) |
|--------|--------|---------------------------|
| E2E Latency | < 2s | 1.2-1.8s |
| ASR RTF | < 0.3 | 0.15-0.25 |
| TTS RTF | < 0.2 | 0.10-0.15 |
| VRAM Usage | ≤ 4GB | 3.5-4.0GB |
| Concurrent Users | 1-2 | 1-2 |

## Optimization Techniques

### 1. ASR Optimization

#### A. Backend Selection

**Transformers (default):**

- VRAM: 2.5-3GB
- Stable, compatible
- RTF: 0.20-0.25
- **Best for:** GPU 4GB, stable deployment

**vLLM:**

- RTF: 0.08-0.12 (2-3x faster)
- Better batching
- VRAM: 5-6GB
- **Best for:** GPU 6GB+, high throughput

**Configuration:**

```bash
# .env
ASR_BACKEND=transformers  # For small GPUs
ASR_GPU_MEMORY=0.4        # Limit VRAM usage
```

#### B. Preprocessing

Enable preprocessing to improve accuracy +4-8%:

```bash
ASR_PREPROCESS=true
```

**Pipeline:**

1. DC offset removal
2. Silence trimming  
3. Normalization to -20dB
4. Pre-emphasis filter (0.97)

**Trade-off:**

- Accuracy: +4-8% (especially with low-quality audio)
- Latency: +10-20ms
- **Recommendation:** Enable for production

#### C. Streaming ASR

Reduce Time-to-First-Audio (TTFA) for long audio:

```bash
ASR_STREAMING=true
```

**How it works:**

- Auto-enable for audio > 2s
- Split into 2s chunks
- Transcribe in parallel
- Join results

**Performance:**

- Non-streaming: 800-1000ms TTFA
- Streaming: 400-600ms TTFA
- **Recommendation:** Enable for conversations

### 2. VAD Optimization

#### A. Threshold Tuning

```bash
VAD_THRESHOLD=0.5  # Default: balanced
```

**Tuning guide:**

- `0.3`: More sensitive → more noise → more false positives
- `0.5`: Balanced (recommended)
- `0.7`: Less sensitive → only clear speech → may miss soft speech

#### B. Silence Detection

```bash
VAD_MIN_SILENCE_MS=300  # Default
```

**Trade-off:**

- `200ms`: Faster, but may cut mid-sentence
- `300ms`: Balanced (recommended)
- `500ms`: Wait longer, ensures complete sentences

#### C. Buffer Management

VAD automatically buffers to process 512-sample chunks:

```python
# voice_agent/services/vad.py
class _VADIterator:
    def __init__(self):
        self._buffer = b""  # Internal buffer
```

**No user config needed** - automatically optimized.

### 3. LLM Optimization

#### A. Model Selection

```bash
GROQ_MODEL=llama-3.3-70b-versatile  # Default: balanced
```

**Options:**

| Model | Speed | Quality | Context |
|-------|-------|---------|---------|
| llama-3.1-8b-instant | Fast | Medium | 8K |
| llama-3.3-70b-versatile | Medium | High | 32K |
| mixtral-8x7b-32768 | Medium | High | 32K |

**Recommendation:** llama-3.3-70b-versatile (best balance)

#### B. Streaming

```bash
# Enabled by default
```

Groq API streaming reduces latency:

- Non-streaming: 800-1200ms
- Streaming: 400-800ms (first token)

#### C. Temperature & Length

```bash
GROQ_TEMPERATURE=0.7    # 0.0 = deterministic, 2.0 = creative
GROQ_MAX_TOKENS=150     # Shorter = faster
```

**For low-latency:**

```bash
GROQ_TEMPERATURE=0.5
GROQ_MAX_TOKENS=100
```

### 4. TTS Optimization

#### A. GPU Memory

```bash
TTS_GPU_MEMORY=0.4  # 40% of available VRAM
```

**Tuning:**

- GPU 4GB: `0.3-0.4`
- GPU 6GB: `0.5-0.6`
- GPU 8GB+: `0.6-0.8`

#### B. Sample Rate

```bash
TTS_SAMPLE_RATE=24000  # Default
```

**Options:**

- 16000: Faster, lower quality
- 24000: Balanced (recommended)
- 48000: Slower, higher quality

#### C. Speed

```bash
TTS_SPEED=1.0  # Normal speed
```

**Tuning:**

- `0.8`: Slower, clearer (for elderly users)
- `1.0`: Normal (recommended)
- `1.2`: Faster (for power users)

### 5. Pipeline Optimization

#### A. Service Isolation

Test each service individually to debug bottlenecks:

```bash
# Test ASR latency only
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false
```

#### B. Concurrent Processing

Pipeline is automatic:

- VAD → ASR: sequential
- ASR → LLM: sequential  
- LLM → TTS: streaming (parallel text chunks)

**No manual tuning needed.**

#### C. Audio Buffer Size

```javascript
// web_client/server.py
const bufferSize = 2048;  // Must be power of 2
```

**Options:**

- 1024: Lower latency, more overhead
- 2048: Balanced (recommended)
- 4096: Higher latency, less overhead

### 6. System-Level Optimization

#### A. CUDA Settings

```bash
# .env or shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reduces fragmentation, helps avoid OOM.

#### B. Python GC

```python
# voice_agent/main.py (already configured)
import gc
gc.set_threshold(700, 10, 10)  # More aggressive GC
```

#### C. Uvicorn Workers

```bash
# Single worker (default) - for GPU
uv run voice-agent

# Multiple workers - ONLY for CPU-only
uv run voice-agent --workers 4  # Multiple GPUs needed
```

**Warning:** Multiple workers require multiple GPUs (1 GPU/worker).

## Benchmarking

### Latency Breakdown Script

```python
import asyncio
import time
from voice_agent.services import Orchestrator

async def benchmark():
    orch = Orchestrator(session_id="bench")
    await orch.start()
    
    # Test audio (2s @ 16kHz)
    import numpy as np
    audio = np.random.randn(32000).astype(np.float32).tobytes()
    
    start = time.perf_counter()
    results = []
    
    async for result in orch.process_audio(audio):
        results.append(result)
    
    total_ms = (time.perf_counter() - start) * 1000
    print(f"Total latency: {total_ms:.1f}ms")
    
    # Check metrics
    metrics = orch._monitor.get_metrics()
    print(f"ASR RTF: {metrics['performance']['asr_rtf']:.3f}")

asyncio.run(benchmark())
```

### Expected Results

**Good configuration (transformers backend):**

```
Total latency: 1456.2ms
  VAD: 3.1ms
  ASR: 453.2ms (RTF: 0.18)
  LLM: 687.5ms
  TTS: 312.4ms (RTF: 0.12)
```

**Excellent configuration (vLLM backend):**

```
Total latency: 1124.8ms
  VAD: 2.9ms
  ASR: 189.7ms (RTF: 0.08)
  LLM: 645.3ms
  TTS: 286.9ms (RTF: 0.11)
```

## Configuration Presets

### Preset 1: Low-Latency (GPU 6GB+)

```bash
# .env
ASR_BACKEND=vllm
ASR_GPU_MEMORY=0.6
ASR_STREAMING=true
ASR_PREPROCESS=false       # Skip for speed

VAD_MIN_SILENCE_MS=200     # Quick cutoff

GROQ_TEMPERATURE=0.5
GROQ_MAX_TOKENS=100

TTS_SPEED=1.1
TTS_GPU_MEMORY=0.5
```

**Target:** < 1s E2E latency

### Preset 2: Balanced (GPU 4GB)

```bash
# .env (defaults)
ASR_BACKEND=transformers
ASR_GPU_MEMORY=0.4
ASR_STREAMING=true
ASR_PREPROCESS=true

VAD_THRESHOLD=0.5
VAD_MIN_SILENCE_MS=300

GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=150

TTS_GPU_MEMORY=0.4
TTS_SPEED=1.0
```

**Target:** 1.2-1.8s E2E latency

### Preset 3: High-Quality (GPU 8GB)

```bash
# .env
ASR_BACKEND=vllm
ASR_GPU_MEMORY=0.7
ASR_STREAMING=false        # Full-context for quality
ASR_PREPROCESS=true

VAD_MIN_SILENCE_MS=500     # Wait for complete sentences

GROQ_MODEL=mixtral-8x7b-32768
GROQ_TEMPERATURE=0.8
GROQ_MAX_TOKENS=200

TTS_SAMPLE_RATE=24000
TTS_SPEED=0.9
TTS_GPU_MEMORY=0.6
```

**Target:** Best quality, latency < 2.5s

### Preset 4: CPU-Only (No GPU)

```bash
# .env
ASR_DEVICE=cpu
ASR_BACKEND=transformers
ASR_PREPROCESS=false

VAD_DEVICE=cpu

LLM_ENABLED=true  # Groq is cloud-based (OK)

TTS_DEVICE=cpu
```

**Warning:** CPU ASR/TTS is very slow (RTF > 1.0), not real-time.

## Profiling & Debugging

### Enable Profiling

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Memory Profiling

```python
import torch

# Check VRAM usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

# Clear cache
torch.cuda.empty_cache()
```

### Latency Profiling

Use the `Timer` utility:

```python
from voice_agent.utils.monitor import Timer

with Timer() as t:
    result = await service.process(data)

print(f"Elapsed: {t.elapsed_ms:.1f}ms")
```

## Scaling

### Horizontal Scaling (Multiple Servers)

```bash
# Server 1 (ASR only)
ASR_ENABLED=true
LLM_ENABLED=false
TTS_ENABLED=false

# Server 2 (TTS only)
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=true

# Load balancer distributes requests
```

### Vertical Scaling (Bigger GPU)

```bash
# RTX 3090 (24GB)
ASR_BACKEND=vllm
ASR_GPU_MEMORY=0.8
TTS_GPU_MEMORY=0.7

# Can run multiple concurrent sessions
```

## References

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [VLLM Documentation](https://docs.vllm.ai/)
- [Groq Cloud Speed](https://groq.com/)
- [Audio Preprocessing Best Practices](https://www.audiokinetic.com/)

## Optimization Checklist

- [ ] Choose ASR backend (transformers vs vLLM) based on GPU
- [ ] Enable ASR streaming for conversations
- [ ] Tune VAD threshold for your environment
- [ ] Enable preprocessing for noisy audio
- [ ] Select appropriate LLM model for speed/quality balance
- [ ] Configure GPU memory limits to avoid OOM
- [ ] Set up monitoring & logging
- [ ] Benchmark on target hardware
- [ ] Test service isolation for debugging
- [ ] Document your production config

## Next Steps

- Implement caching for repeated queries
- Add batch processing for multiple users
- Optimize model quantization (INT8/INT4)
- Implement GPU memory pooling
- Add CDN for TTS audio caching