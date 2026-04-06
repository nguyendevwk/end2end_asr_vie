# 🎙️ Vietnamese Voice Agent

> Real-time AI Voice Agent cho tiếng Việt với kiến trúc tối ưu low-latency

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-orange.svg)](https://github.com/astral-sh/uv)

## ✨ Features

- 🚀 **Real-time** - Latency < 2s end-to-end
- 🎯 **Vietnamese optimized** - Qwen3-ASR + Gwen-TTS cho tiếng Việt
- 💡 **Lightweight** - Chạy trên GPU 4GB (GTX 1650 Ti)
- 🔧 **Production-ready** - Monitoring, logging, error handling đầy đủ
- 🧩 **Service isolation** - Bật/tắt từng service để test/debug
- 📊 **Streaming support** - ASR streaming cho audio dài
- 🎨 **Clean codebase** - Type hints, protocols, well-documented

## 🏗️ Architecture

```
Web Browser → WebSocket → FastAPI
                            ↓
                    [Orchestrator]
                     /    |    \
                    /     |     \
              [VAD]  [ASR]  [LLM]  [TTS]
                |      |      |      |
            Silero  Qwen3   Groq   Gwen
```

**Pipeline:** Audio → VAD → ASR → LLM → TTS → Audio

**Tech stack:**

- **VAD:** Silero VAD v5 (1.5MB, CPU)
- **ASR:** Qwen3-ASR-0.6B (transformers/vLLM)
- **LLM:** Groq Cloud (llama-3.3-70b)
- **TTS:** Gwen-TTS-0.6B
- **Framework:** FastAPI + WebSocket
- **Package manager:** UV (Astral)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (cho GPU)
- GPU 4GB+ VRAM (recommended)
- Ubuntu 20.04+ / macOS

### Installation

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repo
git clone <repo-url>
cd end2end_asr_vie/src

# 3. Install dependencies
uv sync

# 4. Configure
cp .env.example .env
nano .env  # Add your GROQ_API_KEY

# 5. Run server
uv run voice-agent

# 6. Run web client (new terminal)
uv run voice-agent-client

# 7. Open browser
# http://localhost:8080
```

### First conversation

1. Click **"Start Listening"**
2. Nói: *"Xin chào"*
3. Đợi 1-2 giây
4. Nghe response từ AI

## 📖 Documentation

Tài liệu đầy đủ trong thư mục [docs/](docs/):

- 📘 [Kiến trúc hệ thống](docs/ARCHITECTURE.md) - Thiết kế và luồng xử lý
- 📗 [Hướng dẫn cài đặt](docs/INSTALLATION.md) - Setup chi tiết
- 📙 [Hướng dẫn sử dụng](docs/USAGE.md) - API và examples
- 📕 [Tối ưu hóa](docs/OPTIMIZATION.md) - Performance tuning
- 📔 [Troubleshooting](docs/TROUBLESHOOTING.md) - Xử lý lỗi
- 📓 [API Reference](docs/API.md) - REST & WebSocket API

## ⚙️ Configuration

Cấu hình quan trọng trong `.env`:

```bash
# Groq API (required)
GROQ_API_KEY=gsk_xxx...

# Service toggles
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=true
TTS_ENABLED=true

# ASR backend
ASR_BACKEND=transformers  # hoặc 'vllm' (GPU 6GB+)
ASR_GPU_MEMORY=0.4        # Limit VRAM
ASR_STREAMING=true        # Streaming cho audio >2s
ASR_PREPROCESS=true       # +4-8% accuracy

# Performance
VAD_THRESHOLD=0.5         # Speech detection sensitivity
GROQ_TEMPERATURE=0.7      # LLM creativity
TTS_SPEED=1.0             # Speaking speed
```

Xem [.env.example](src/.env.example) cho tất cả options.

## 📊 Performance

**Latency (GTX 1650 Ti 4GB):**

| Component | Latency | RTF |
|-----------|---------|-----|
| VAD | 3-5ms | - |
| ASR | 400-600ms | 0.18 |
| LLM | 600-800ms | - |
| TTS | 300-500ms | 0.12 |
| **Total** | **1.2-1.8s** ✅ | - |

**Resource usage:**

- GPU: 3.5-4.0GB VRAM
- CPU: 5-10%
- Network: ~50KB/s (audio streaming)

## 🧪 Testing

### Test riêng từng service

```bash
# Test VAD only
VAD_ENABLED=true
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=false

uv run voice-agent
```

### Run tests

```bash
uv run pytest tests/
```

## 🎯 Use Cases

### 1. Voice Chatbot (Full pipeline)

```bash
# All services enabled
uv run voice-agent
```

### 2. Transcription service (VAD + ASR)

```bash
LLM_ENABLED=false
TTS_ENABLED=false
```

### 3. TTS service (Text-to-Speech only)

```bash
VAD_ENABLED=false
ASR_ENABLED=false
LLM_ENABLED=false
TTS_ENABLED=true
```

## 🔧 Development

### Project structure

```
end2end_asr_vie/
├── docs/                    # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── OPTIMIZATION.md
│   ├── TROUBLESHOOTING.md
│   └── API.md
├── src/                     # Source code
│   ├── voice_agent/         # Main package
│   │   ├── api/             # FastAPI routes & WebSocket
│   │   ├── core/            # Types, exceptions, constants
│   │   ├── services/        # VAD, ASR, LLM, TTS, Orchestrator
│   │   ├── utils/           # Logger, monitor, audio utils
│   │   ├── config.py        # Pydantic Settings
│   │   └── main.py          # Application entry
│   ├── web_client/          # Web test client
│   ├── tests/               # Unit tests
│   ├── models/              # Downloaded models (gitignored)
│   ├── pyproject.toml       # UV dependencies
│   ├── .env.example         # Config template
│   └── README.md            # Quick reference
└── README.md                # This file
```

### Coding standards

- **Type hints:** Bắt buộc cho tất cả functions
- **Protocols:** Interface-based design
- **Async/await:** Non-blocking I/O
- **Structured logging:** JSON logs với structlog
- **Error handling:** Custom exceptions hierarchy
- **Monitoring:** Latency tracking, RTF metrics

Xem [src/CODING_STANDARDS.md](src/CODING_STANDARDS.md) cho chi tiết.

### Add new service

1. Implement protocol interface (`IXXXService`)
2. Add service class trong `services/`
3. Register trong `Orchestrator`
4. Add config trong `.env`
5. Update tests

## 🐛 Common Issues

### CUDA Out of Memory

```bash
# Giảm GPU allocation
ASR_GPU_MEMORY=0.3
TTS_GPU_MEMORY=0.3

# Hoặc tắt TTS
TTS_ENABLED=false
```

### Slow latency

```bash
# Enable streaming
ASR_STREAMING=true

# Use vLLM (if GPU ≥ 6GB)
ASR_BACKEND=vllm
```

### Dependencies conflict

```bash
rm -rf .venv uv.lock
uv sync
```

Xem [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) cho tất cả lỗi.

## 📈 Roadmap

- [ ] Multi-user support (concurrent sessions)
- [ ] Voice cloning (speaker adaptation)
- [ ] Emotion detection
- [ ] Model quantization (INT8/INT4)
- [ ] Docker deployment
- [ ] Kubernetes helm charts
- [ ] Metrics dashboard (Grafana)
- [ ] Multi-language support

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repo
2. Create feature branch
3. Follow coding standards
4. Add tests
5. Submit PR

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM) - Qwen3-ASR model
- [G-Group AI Lab](https://huggingface.co/g-group-ai-lab) - Gwen-TTS model
- [Silero Team](https://github.com/snakers4/silero-vad) - Silero VAD
- [Groq](https://groq.com/) - Ultra-fast LLM inference
- [Astral](https://astral.sh/) - UV package manager

## 📧 Contact

- **Demo purpose:** Phỏng vấn xin việc
- **Tech stack:** Python 3.12, FastAPI, PyTorch, CUDA
- **Target:** Real-time voice agent với low latency

---

<p align="center">
  <b>Made with ❤️ for Vietnamese AI Voice Applications</b>
</p>
