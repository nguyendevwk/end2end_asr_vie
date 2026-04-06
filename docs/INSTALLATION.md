# Hướng dẫn cài đặt

## 📋 Yêu cầu hệ thống

### Phần cứng

**Tối thiểu:**

- CPU: 4 cores
- RAM: 8GB
- GPU: 4GB VRAM (NVIDIA)
- Disk: 10GB free

**Khuyến nghị:**

- CPU: 6+ cores
- RAM: 16GB
- GPU: 6GB+ VRAM (NVIDIA RTX)
- Disk: 20GB SSD

### Phần mềm

- **OS:** Linux (Ubuntu 20.04+) / macOS (CPU only)
- **Python:** 3.12.x
- **CUDA:** 11.8+ (cho GPU)
- **Git:** 2.0+

## 🚀 Cài đặt

### 1. Cài đặt UV (Python package manager)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Hoặc dùng pip
pip install uv
```

### 2. Clone repository

```bash
git clone <repository-url>
cd end2end_asr_vie/src
```

### 3. Cài đặt dependencies

```bash
# Tự động tạo venv và cài packages
uv sync

# Hoặc với extras (nếu cần vLLM backend)
uv sync --extra vllm
```

**Lưu ý về dependencies:**

- `transformers==4.57.6` - Đã pin để tương thích qwen-asr/qwen-tts
- `qwen-asr[vllm]<=0.0.6` - ASR model
- `qwen-tts>=0.1.1` - TTS model
- `torch>=2.0.0` - PyTorch với CUDA support

### 4. Tải models

Models sẽ tự động download lần đầu chạy, nhưng bạn có thể tải trước:

```bash
# Kích hoạt venv
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

# VAD model (tự động download khi chạy, ~1.5MB)
```

**Vị trí lưu models:**

```
~/.cache/huggingface/hub/          # Transformers models
~/.cache/torch/hub/                # Silero VAD
```

Hoặc tự chỉ định thư mục:

```bash
# Download vào thư mục cục bộ
mkdir -p models/{asr,tts,vad}

# Xem hướng dẫn chi tiết trong models/README.md
```

### 5. Cấu hình

```bash
# Copy file mẫu
cp .env.example .env

# Chỉnh sửa cấu hình
nano .env
```

**Cấu hình tối thiểu:**

```bash
# .env
GROQ_API_KEY=gsk_xxx...  # Lấy từ https://console.groq.com/
```

**Cấu hình GPU nhỏ (4GB VRAM):**

```bash
# .env
ASR_BACKEND=transformers
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.4

# Hoặc tắt TTS nếu cần
TTS_ENABLED=false
```

## ✅ Kiểm tra cài đặt

### Test import

```bash
uv run python -c "
from voice_agent import __version__
from voice_agent.services import VADService, ASRService, TTSService
print(f'✅ Voice Agent v{__version__} installed successfully')
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
    print('✅ VAD loaded')

asyncio.run(test())
"
```

## 🐛 Xử lý lỗi cài đặt

### Lỗi: `No solution found when resolving dependencies`

**Nguyên nhân:** Conflict giữa qwen-asr và qwen-tts về transformers version

**Giải pháp:**

```bash
# Xóa lock file
rm uv.lock

# Cài lại
uv sync
```

### Lỗi: `CUDA out of memory`

**Nguyên nhân:** GPU không đủ VRAM

**Giải pháp:**

1. Giảm GPU memory allocation:

```bash
# .env
ASR_GPU_MEMORY=0.2
TTS_GPU_MEMORY=0.3
```

1. Dùng CPU backend:

```bash
# .env
ASR_DEVICE=cpu
TTS_DEVICE=cpu
```

1. Tắt services không cần:

```bash
# .env
TTS_ENABLED=false  # Chỉ test ASR
```

### Lỗi: `flash-attn not installed`

**Nguyên nhân:** Flash Attention chưa cài (optional optimization)

**Giải pháp:**

Có thể ignore (hệ thống tự fallback sang PyTorch implementation):

```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```

Hoặc cài flash-attn (cần GPU compute capability >= 8.0):

```bash
uv pip install flash-attn --no-build-isolation
```

### Lỗi: `transformers compatibility`

**Nguyên nhân:** Sai version transformers

**Giải pháp:**

```bash
# Force cài đúng version
uv pip install transformers==4.57.6 --force-reinstall
```

## 🔧 Cài đặt nâng cao

### Dùng vLLM backend (faster ASR)

**Yêu cầu:** GPU 6GB+ VRAM

```bash
# Cài vLLM
uv sync --extra vllm

# Hoặc
uv pip install vllm

# Cấu hình
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
# Cài dev dependencies
uv pip install -e ".[dev]"

# Hoặc
uv sync --extra dev

# Bao gồm: pytest, black, ruff, mypy
```

## 📦 Cấu trúc thư mục sau cài đặt

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

## ▶️ Chạy server

Sau khi cài đặt xong:

```bash
# Chạy voice agent server
uv run voice-agent

# Mở terminal khác, chạy web client
uv run voice-agent-client

# Truy cập http://localhost:8080
```

## 🔄 Cập nhật

```bash
# Pull code mới
git pull

# Cập nhật dependencies
uv sync

# Restart server
```

## 🗑️ Gỡ cài đặt

```bash
# Xóa virtual environment
rm -rf .venv

# Xóa cache models (optional)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Xóa source code
cd ..
rm -rf end2end_asr_vie
```

## 📚 Tiếp theo

- Đọc [USAGE.md](USAGE.md) để học cách sử dụng
- Đọc [OPTIMIZATION.md](OPTIMIZATION.md) để tối ưu hiệu năng
- Xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md) nếu gặp lỗi
