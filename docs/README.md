# Vietnamese Voice Agent Documentation

## Documentation Guide

Next-generation Voice Agent system for Vietnamese with real-time optimized architecture.

### Main Documentation

1. **[System Architecture](ARCHITECTURE.md)** - Overview design and processing pipeline
2. **[Installation Guide](INSTALLATION.md)** - Installation and environment configuration
3. **[Usage Guide](USAGE.md)** - Running and testing the system
4. **[Performance Optimization](OPTIMIZATION.md)** - Optimization techniques
5. **[API Reference](API.md)** - API endpoints and WebSocket details

### Technical Documentation

- **[qwenasr/README.md](qwenasr/README.md)** - Qwen3-ASR model guide
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common error handling

## Project Goals

An open-source, end-to-end Vietnamese voice agent for real-time speech-to-text and conversational AI.

- **Clean code** - Easy to read, easy to extend
- **Real-time** - Latency < 2s end-to-end
- **Lightweight** - Runs on 4GB GPU
- **Production-ready** - Monitoring, logging, error handling included

## Quick Start

```bash
# Clone and install
cd src
uv sync

# Configure
cp .env.example .env
# Set GROQ_API_KEY in .env

# Run server
uv run voice-agent

# Run web client (different terminal)
uv run voice-agent-client
```

## Further Reading

- See [../src/README.md](../src/README.md) for quick reference
- See [ARCHITECTURE.md](ARCHITECTURE.md) to understand the architecture
- See [OPTIMIZATION.md](OPTIMIZATION.md) to optimize performance
