# Vietnamese Voice Agent

Real-time AI voice agent for Vietnamese with optimized low-latency architecture.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-orange.svg)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

## Features

- **Real-time** — End-to-end latency < 2s on consumer GPU
- **Vietnamese optimized** — Qwen3-ASR + Gwen-TTS tuned for Vietnamese
- **Lightweight** — Runs on 4GB VRAM (GTX 1650 Ti)
- **Production-ready** — Monitoring, logging, error handling, Prometheus metrics
- **Service isolation** — Enable/disable individual services for testing
- **Streaming** — Chunked ASR for long audio, progressive TTS delivery
- **Pluggable LLM** — Swappable text tasks: passthrough, LLM, router, or custom
- **Reconnect resume** — Session snapshots survive transient disconnects
- **Clean codebase** — Full type hints, Protocol-based design, structured logging

## Architecture

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

| Component | Model | Size | Backend |
|-----------|-------|------|---------|
| VAD | Silero VAD v5 | 1.5MB | CPU |
| ASR | Qwen3-ASR-0.6B | ~1.3GB | transformers / vLLM |
| LLM | llama-3.3-70b (Groq Cloud) | — | Groq API |
| TTS | Gwen-TTS-0.6B | ~1.3GB | CPU/GPU |

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (for GPU inference)
- 4GB+ VRAM (recommended)
- [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nguyendevwk/end2end_asr_vie.git
cd end2end_asr_vie/src

# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run server
uv run voice-agent

# Run web client (separate terminal)
uv run voice-agent-client
```

Open http://localhost:8080 in your browser.

### Docker

```bash
docker compose up
```

## Configuration

Key settings in `.env`:

```bash
# Required
GROQ_API_KEY=gsk_xxx...

# Service toggles
VAD_ENABLED=true
ASR_ENABLED=true
LLM_ENABLED=true
TTS_ENABLED=true

# ASR backend
ASR_BACKEND=transformers  # or 'vllm' (GPU 6GB+)
ASR_GPU_MEMORY=0.3        # VRAM limit
ASR_STREAMING=true        # Chunked ASR for audio >2s

# Performance
MAX_CCU=50                # Max concurrent connections
MAX_INFLIGHT_INFER=4      # Max parallel GPU inferences
```

See [`.env.example`](src/.env.example) for all options.

## Performance

Measured on GTX 1650 Ti 4GB:

| Component | Latency | RTF |
|-----------|---------|-----|
| VAD | 3–5ms | — |
| ASR | 400–600ms | 0.18 |
| LLM | 600–800ms | — |
| TTS | 300–500ms | 0.12 |
| **Total** | **1.2–1.8s** | — |

Resource usage: 3.5–4.0GB VRAM, 5–10% CPU, ~50KB/s network.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run without GPU
uv run pytest tests/ -m "not slow and not gpu"

# Type check
uv run mypy src/voice_agent/

# Lint
uv run ruff check src/
```

## Use Cases

**Full voice chatbot** (all services enabled):
```bash
uv run voice-agent
```

**Transcription only** (VAD + ASR):
```bash
ASR_ENABLED=true LLM_ENABLED=false TTS_ENABLED=false uv run voice-agent
```

**TTS only**:
```bash
TTS_ENABLED=true VAD_ENABLED=false ASR_ENABLED=false LLM_ENABLED=false uv run voice-agent
```

## Project Structure

```
end2end_asr_vie/
├── docs/                    # Documentation
├── src/
│   ├── voice_agent/         # Main package
│   │   ├── api/             # FastAPI routes & WebSocket
│   │   ├── core/            # Types, exceptions, constants
│   │   ├── services/        # VAD, ASR, LLM, TTS, Orchestrator
│   │   ├── utils/           # Logger, monitor, audio utilities
│   │   ├── config.py        # Pydantic Settings
│   │   └── main.py          # Application entry point
│   ├── web_client/          # Browser-based test client
│   ├── tests/               # Test suite
│   └── pyproject.toml       # Dependencies & tooling config
├── .github/                 # CI/CD workflows
├── LICENSE                  # MIT License
└── README.md
```

## Roadmap

- [ ] Multi-user concurrent sessions
- [ ] Voice cloning (speaker adaptation)
- [ ] Emotion detection
- [ ] Model quantization (INT8/INT4)
- [ ] Kubernetes helm charts
- [ ] Grafana metrics dashboard
- [ ] Multi-language support

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM) — Qwen3-ASR model
- [G-Group AI Lab](https://huggingface.co/g-group-ai-lab) — Gwen-TTS model
- [Silero Team](https://github.com/snakers4/silero-vad) — Silero VAD
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [Astral](https://astral.sh/) — UV package manager
