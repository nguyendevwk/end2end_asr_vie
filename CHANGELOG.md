# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-22

### Added
- Real-time Vietnamese voice pipeline (VAD → ASR → LLM → TTS)
- Qwen3-ASR with transformers/vLLM backends
- Gwen-TTS with streaming synthesis
- Groq LLM integration (llama-3.3-70b)
- Silero VAD v5 with per-session state isolation
- WebSocket protocol with heartbeat, reconnect resume, and CCU shedding
- Prometheus metrics endpoint (`/metrics`)
- Health probes (`/health/live`, `/health/ready`)
- Pluggable text task framework (passthrough, LLM, router)
- Session snapshots for reconnect recovery
- Audio preprocessing (high-pass, low-pass, noise gate, normalization)
- Audio postprocessing (soft limiter, fade in/out)
- Pydantic settings with environment variable support
- Structured logging via structlog
- Browser-based test client
- Docker support

### Performance
- Causal SOS filters (2x faster than filtfilt)
- Vectorized noise gate
- Polyphase resampling with byte-accurate buffering
- ASR/TTS/LLM inference in worker threads
- Progressive TTS streaming without pacing sleep
- LLM clause-boundary splitting for lower TTFA
- O(n) VAD buffer using list-of-chunks

### Fixed
- VAD speech_start offset by one chunk
- TTS duration calculation from dummy buffer
- ASR merged text not emitted to caller
- LLM IndexError on empty API response
- Orchestrator race conditions on interrupt/reset
- History cap preserving user/assistant pair coherence
- Session store expired accumulation
- Startup failure GPU memory leak
- CORS credentials with wildcard origin
- Health endpoint defaulting missing is_started to True
- Deprecated `asyncio.get_event_loop()` usage

### Testing
- 69 tests covering text tasks, metrics, resilience, session store, orchestrator, audio utilities, processor, and monitor
- Protocol-conforming test doubles
- CI/CD with lint, type check, and test jobs
