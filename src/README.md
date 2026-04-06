# 🎤 Voice Agent

<p align="center">
  <strong>Real-time Vietnamese Voice Agent với AI Pipeline hiện đại</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/PyTorch-2.5+-red.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## 🎯 Overview

Voice Agent là một demo **end-to-end voice pipeline** cho tiếng Việt, sử dụng các model AI state-of-the-art:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  🔇 VAD │ ──▶ │ 🎙️ ASR  │ ──▶ │  🤖 LLM │ ──▶ │ 🔊 TTS  │
│ Silero  │     │Qwen3-ASR│     │  Groq   │     │Qwen3-TTS│
│  <10ms  │     │ <500ms  │     │ <300ms  │     │  <300ms │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                         
         Target: End-to-end latency < 2 seconds
```

## ✨ Features

| Component | Technology | Highlight |
|-----------|------------|-----------|
| **VAD** | Silero VAD v5 | Real-time voice detection, <10ms latency |
| **ASR** | Qwen3-ASR-0.6B (vLLM) | 52 languages, 2000x throughput |
| **LLM** | Groq API (Llama 3.3 70B) | 330 tokens/second |
| **TTS** | Qwen3-TTS-0.6B | 97ms first packet, voice cloning |

### Key Capabilities

- 🎯 **Real-time Processing** - Streaming audio với WebSocket
- 🔄 **Interrupt Handling** - Phát hiện người dùng ngắt bot
- 📊 **Full Observability** - Structured logging + metrics monitoring
- 🎨 **Audio Pipeline** - Preprocessor (noise gate, filters) + Postprocessor (limiter, fade)
- 🌐 **Web Test Client** - Flask app với audio visualizer

## 📋 Requirements

- **Python** 3.12+
- **GPU** CUDA-capable (khuyến nghị 8GB+ VRAM)
- **[UV](https://docs.astral.sh/uv/)** - Fast Python package manager

## 🚀 Quick Start

### 1. Setup

```bash
cd src

# Tạo virtual environment với UV
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
uv pip install -e ".[dev]"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Chỉnh sửa .env - Thêm GROQ_API_KEY (bắt buộc)
# Lấy free key tại: https://console.groq.com/
```

### 3. Run Server

```bash
# Development mode (hot reload)
uv run voice-agent

# Hoặc trực tiếp
uv run python -m voice_agent.main
```

### 4. Test Client

```bash
# Terminal khác
uv run voice-agent-client --port 8080

# Mở http://localhost:8080
```

## 📡 API Reference

### WebSocket `/ws/agent`

Real-time bidirectional voice communication.

```
┌────────────┐                           ┌────────────┐
│   Client   │                           │   Server   │
└─────┬──────┘                           └─────┬──────┘
      │                                        │
      │──── Binary: PCM S16LE @ 16kHz ────────▶│
      │                                        │
      │◀─── Text: "LISTENING" ─────────────────│
      │◀─── Text: "PROCESSING" ────────────────│
      │◀─── Text: "TRANSCRIPT:xin chào" ───────│
      │◀─── Binary: TTS Audio ─────────────────│
      │◀─── Text: "SPEAKING" ──────────────────│
      │                                        │
```

**Events:**

| Event | Description |
|-------|-------------|
| `LISTENING` | Ready for voice input |
| `PROCESSING` | Transcribing & generating response |
| `SPEAKING` | Playing TTS response |
| `IDLE` | No activity |
| `TRANSCRIPT:<text>` | Recognized speech |
| `ERROR:<message>` | Error occurred |

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |
| `/info` | GET | Service info + versions |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

## 📁 Architecture

```
src/
├── voice_agent/              # Main package
│   ├── main.py              # FastAPI + Uvicorn entry point
│   ├── config.py            # Pydantic Settings
│   ├── core/                # Domain types & exceptions
│   │   ├── types.py         # Protocol-based interfaces
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── constants.py     # Audio constants (16kHz, etc.)
│   ├── services/            # AI service implementations
│   │   ├── vad.py           # Silero VAD wrapper
│   │   ├── asr.py           # Qwen3-ASR (vLLM)
│   │   ├── tts.py           # Qwen3-TTS
│   │   ├── llm.py           # Groq API client
│   │   └── orchestrator.py  # Pipeline coordinator
│   ├── api/                 # Web layer
│   │   ├── websocket.py     # WebSocket handler
│   │   └── routes.py        # HTTP endpoints
│   └── utils/               # Utilities
│       ├── logger.py        # Structured logging (structlog)
│       ├── monitor.py       # Metrics & latency tracking
│       ├── audio.py         # Audio conversion helpers
│       └── processor.py     # Pre/Post audio processing
├── web_client/              # Standalone test client
│   ├── server.py            # Flask + HTML/JS
│   └── README.md
├── models/                  # Downloaded model weights
│   ├── asr/
│   ├── tts/
│   └── vad/
└── tests/                   # Unit tests
```

## ⚡ Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| VAD Latency | < 10ms | Voice detection |
| ASR Latency | < 500ms | For 3 seconds audio |
| LLM TTFT | < 300ms | Time to first token |
| TTS Latency | < 300ms | First audio packet |
| **E2E Latency** | **< 2s** | User speaks → Bot responds |

## 🧪 Development

### Testing

```bash
# Unit tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=voice_agent

# Single test
uv run pytest tests/test_vad.py -v
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .
```

### Coding Standards

Xem [CODING_STANDARDS.md](./CODING_STANDARDS.md) để biết quy chuẩn code chi tiết.

## 🔧 Configuration

Tất cả config qua environment variables hoặc `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `GROQ_API_KEY` | - | **Required** - Groq API key |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | ASR model name |
| `TTS_MODEL` | `Qwen/Qwen3-TTS-0.6B` | TTS model name |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `VAD_THRESHOLD` | `0.5` | VAD confidence threshold |

## 📄 License

MIT License - Feel free to use for your projects!

---

<p align="center">
  Built with ❤️ using FastAPI, PyTorch, and Qwen models
</p>
