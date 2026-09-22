# Installation Guide

## System Requirements

### Hardware

**Minimum:**

- CPU: 4 cores
- RAM: 8GB
- GPU: 4GB VRAM (NVIDIA)
- Disk: 10GB free

**Recommended:**

- CPU: 6+ cores
- RAM: 16GB
- GPU: 6GB+ VRAM (NVIDIA RTX)
- Disk: 20GB SSD

### Software

- **OS:** Linux (Ubuntu 20.04+) / macOS (CPU only)
- **Python:** 3.12.x
- **CUDA:** 11.8+ (for GPU)
- **Git:** 2.0+

## Installation

### 1. Install UV (Python package manager)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 2. Clone repository

```bash
git clone <repository-url>
cd end2end_asr_vie/src
```

### 3. Install dependencies

```bash
# Automatically create venv and install packages
uv sync

# Or with extras (if vLLM backend is needed)
uv sync --extra vllm
```

**Notes on dependencies:**

- `transformers==4.57.6` - Pinned for compatibility with qwen-asr/qwen-tts
- `qwen-asr[vllm]<=0.0.6` - ASR model
- `qwen-tts>=0.1.1` - TTS model
- `torch>=2.0.0` - PyTorch with CUDA support

### 4. Download models

Models will be automatically downloaded on first run, but you can download them beforehand:

```bash
# Activate venv
source .venv/bin/activate

# Download models
python -c "
from transformers import AutoModel
from qwen_asr import Qwen3ASRModel

# ASR model (~1.5GB)
Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-0.6B')

# TTS model (~1.2GB)
AutoModel.from_pretrained('g-group-ai-lab/gwen-tts-0.6B')
"

# VAD model (auto-downloaded when running, ~1.5MB)
```

**Model storage locations:**

```
~/.cache/huggingface/hub/          # Transformers models
~/.cache/torch/hub/                # Silero VAD
```

Or specify a custom directory:

```bash
# Download to a local directory
mkdir -p models/{asr,tts,vad}

# See detailed instructions in models/README.md
```

### 5. Configuration

```bash
# Copy example file
cp .env.example .env

# Edit configuration
nano .env
```

**Minimum configuration:**

```bash
# .env
GROQ_API_KEY=gsk_xxx...  # Get from https://console.groq.com/
```

**Small GPU configuration (4GB VRAM):**

```bash
# .env
ASR_BACKEND=transformers
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.4

# Or disable TTS if needed
TTS_ENABLED=false
```

## Verify Installation

### Test import

```bash
uv run python -c "
from voice_agent import __version__
from voice_agent.services import VADService, ASRService, TTSService
print(f'Voice Agent v{__version__} installed successfully')
"
```

### Test GPU

```bash
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

Expected output:

```
CUDA available: True
Device: NVIDIA GeForce GTX 1650 Ti
Memory: 4.0GB
```

### Test services

```bash
# Test VAD only
uv run python -c "
import asyncio
from voice_agent.services import VADService

async def test():
    vad = VADService()
    await vad.start()
    print('VAD loaded')

asyncio.run(test())
"
```

## Installation Troubleshooting

### Error: `No solution found when resolving dependencies`

**Cause:** Conflict between qwen-asr and qwen-tts regarding transformers version

**Solution:**

```bash
# Delete lock file
rm uv.lock

# Reinstall
uv sync
```

### Error: `CUDA out of memory`

**Cause:** GPU does not have enough VRAM

**Solution:**

1. Reduce GPU memory allocation:

```bash
# .env
ASR_GPU_MEMORY=0.2
TTS_GPU_MEMORY=0.3
```

2. Use CPU backend:

```bash
# .env
ASR_DEVICE=cpu
TTS_DEVICE=cpu
```

3. Disable unnecessary services:

```bash
# .env
TTS_ENABLED=false  # Only test ASR
```

### Error: `flash-attn not installed`

**Cause:** Flash Attention is not installed (optional optimization)

**Solution:**

You can ignore it (the system will automatically fall back to the PyTorch implementation):

```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```

Or install flash-attn (requires GPU compute capability >= 8.0):

```bash
uv pip install flash-attn --no-build-isolation
```

### Error: `transformers compatibility`

**Cause:** Wrong transformers version

**Solution:**

```bash
# Force install the correct version
uv pip install transformers==4.57.6 --force-reinstall
```

## Advanced Installation

### Using vLLM backend (faster ASR)

**Requirements:** GPU 6GB+ VRAM

```bash
# Install vLLM
uv sync --extra vllm

# Or
uv pip install vllm

# Configure
echo "ASR_BACKEND=vllm" >> .env
```

### Custom model paths

```bash
# .env
ASR_MODEL=/path/to/qwen3-asr
TTS_MODEL=/path/to/gwen-tts
```

### Development setup

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Or
uv sync --extra dev

# Includes: pytest, black, ruff, mypy
```

## Directory Structure After Installation

```
src/
├── .venv/                    # Virtual environment (auto-created)
├── .env                      # Config (user-created)
├── voice_agent/              # Source code
├── tests/                    # Tests
├── web_client/               # Web test client
└── models/                   # Optional: local models
    ├── asr/
    ├── tts/
    └── vad/
```

## Run Server

After installation is complete:

```bash
# Run voice agent server
uv run voice-agent

# Open another terminal and run web client
uv run voice-agent-client

# Access http://localhost:8080
```

## Update

```bash
# Pull new code
git pull

# Update dependencies
uv sync

# Restart server
```

## Uninstall

```bash
# Delete virtual environment
rm -rf .venv

# Delete model cache (optional)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Delete source code
cd ..
rm -rf end2end_asr_vie
```

## Next Steps

- Read [USAGE.md](USAGE.md) to learn how to use
- Read [OPTIMIZATION.md](OPTIMIZATION.md) to optimize performance
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter errors