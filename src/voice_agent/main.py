"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         🎤 VOICE AGENT - ENTRY POINT                          ║
║                                                                               ║
║  Real-time Vietnamese Voice Agent with:                                       ║
║  • Qwen3-ASR (vLLM) - 52 languages @ 2000x throughput                        ║
║  • Qwen3-TTS - 97ms first packet latency                                     ║
║  • Groq LLM - Llama 3.3 70B @ 330 tokens/s                                   ║
║  • Silero VAD - Real-time voice detection                                    ║
║                                                                               ║
║  Target: End-to-end latency < 2 seconds                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from voice_agent import __version__
from voice_agent.api import router, websocket_endpoint
from voice_agent.config import get_settings
from voice_agent.core import (
    ASRConfig,
    LLMConfig,
    TTSConfig,
    VADConfig,
)
from voice_agent.services import (
    ASRService,
    LLMService,
    Orchestrator,
    PipelineRuntime,
    TTSService,
    VADService,
    create_text_task,
)
from voice_agent.utils import (
    AdmissionGate,
    get_logger,
    setup_logging,
)
from voice_agent.utils.session_store import SessionStore, get_session_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from voice_agent.services.text_tasks import TextTask

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

settings = get_settings()
setup_logging(level=settings.log_level, json_output=settings.log_json)
logger = get_logger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                           SERVICE INSTANCES                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# Global service instances - initialized in lifespan, thread-safe singleton pattern

vad_service: VADService | None = None
asr_service: ASRService | None = None
tts_service: TTSService | None = None
llm_service: LLMService | None = None

# Production: shared admission gate, GPU inference semaphore, session store
_connection_gate = AdmissionGate(max_concurrent=50)
_infer_semaphore = asyncio.Semaphore(4)
_session_store: SessionStore = get_session_store()


def _sync_runtime_knobs() -> None:
    """Apply settings to shared production primitives (called on startup)."""
    global _connection_gate, _infer_semaphore
    _connection_gate = AdmissionGate(max_concurrent=settings.max_ccu)
    _infer_semaphore = asyncio.Semaphore(settings.max_inflight_infer)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                          LIFESPAN MANAGEMENT                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


async def _init_vad() -> VADService:
    """Initialize VAD service with Silero model."""
    config = VADConfig(
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
    )
    service = VADService(config)
    await service.start()
    return service


async def _init_asr() -> ASRService:
    """Initialize ASR service with Qwen3-ASR."""
    config = ASRConfig(
        model_name=settings.asr_model,
        language=settings.asr_language,
        gpu_memory_utilization=settings.asr_gpu_memory,
        backend=settings.asr_backend,
        streaming=settings.asr_streaming,
        preprocess=settings.asr_preprocess,
    )
    service = ASRService(config)
    await service.start()
    return service


async def _init_tts() -> TTSService:
    """Initialize TTS service with Qwen3-TTS."""
    config = TTSConfig(
        model_name=settings.tts_model,
        voice_ref_audio=settings.tts_voice_ref_audio,
        voice_ref_text=settings.tts_voice_ref_text,
        temperature=settings.tts_temperature,
    )
    service = TTSService(config)
    await service.start()
    return service


async def _init_llm() -> LLMService:
    """Initialize LLM service with Groq API."""
    config = LLMConfig(
        api_key=settings.groq_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        system_prompt=settings.llm_system_prompt,
    )
    service = LLMService(config)
    await service.start()
    return service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles graceful startup and shutdown of all AI services.
    Services can be individually enabled/disabled via config.
    """
    global vad_service, asr_service, tts_service, llm_service

    logger.info(
        "🚀 starting_voice_agent",
        version=__version__,
        debug=settings.debug,
        host=settings.host,
        port=settings.port,
        vad_enabled=settings.vad_enabled,
        asr_enabled=settings.asr_enabled,
        tts_enabled=settings.tts_enabled,
        llm_enabled=settings.llm_enabled,
    )

    # ═══════════════ STARTUP (parallel: ASR+TTS loads take ~30s each) ═══════════════
    try:
        # Initialize enabled services concurrently to cut startup time
        tasks: dict[str, asyncio.Task[Any]] = {}
        if settings.vad_enabled:
            logger.info("📦 loading_vad_model", model="Silero VAD v5")
            tasks["vad"] = asyncio.create_task(_init_vad())
        else:
            logger.info("⏭️ vad_disabled")

        if settings.asr_enabled:
            logger.info(
                "🎙️ loading_asr_model",
                model=settings.asr_model,
                backend=settings.asr_backend,
            )
            tasks["asr"] = asyncio.create_task(_init_asr())
        else:
            logger.info("⏭️ asr_disabled")

        if settings.tts_enabled:
            logger.info("🔊 loading_tts_model", model=settings.tts_model)
            tasks["tts"] = asyncio.create_task(_init_tts())
        else:
            logger.info("⏭️ tts_disabled")

        if settings.llm_enabled:
            logger.info("🤖 connecting_llm", model=settings.llm_model)
            tasks["llm"] = asyncio.create_task(_init_llm())
        else:
            logger.info("⏭️ llm_disabled")

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, result in zip(tasks.keys(), results, strict=True):
                if isinstance(result, Exception):
                    logger.error(f"❌ {name}_init_failed", error=str(result))
                    raise result
                if name == "vad":
                    vad_service = result
                elif name == "asr":
                    asr_service = result
                elif name == "tts":
                    tts_service = result
                elif name == "llm":
                    llm_service = result

        logger.info(
            "✅ all_services_ready",
            vad="OK" if vad_service else "DISABLED",
            asr="OK" if asr_service else "DISABLED",
            tts="OK" if tts_service else "DISABLED",
            llm="OK" if llm_service else "DISABLED",
        )

        _sync_runtime_knobs()
        _print_banner()

    except Exception as e:
        logger.error("❌ startup_failed", error=str(e), exc_info=True)
        raise

    yield

    # ═══════════════ SHUTDOWN ═══════════════
    logger.info("🛑 shutting_down")

    cleanup_tasks = []
    if llm_service:
        cleanup_tasks.append(llm_service.stop())
    if tts_service:
        cleanup_tasks.append(tts_service.stop())
    if asr_service:
        cleanup_tasks.append(asr_service.stop())
    if vad_service:
        cleanup_tasks.append(vad_service.stop())

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    logger.info("👋 shutdown_complete")


def _print_banner() -> None:
    """Print startup banner."""
    print(
        f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    🎤 VOICE AGENT v{__version__:<10}                              ║
║                                                                               ║
║  📡 Server: http://{settings.host}:{settings.port:<5}                                        ║
║  🔌 WebSocket: ws://{settings.host}:{settings.port}/ws/agent                              ║
║  📚 API Docs: http://{settings.host}:{settings.port}/docs                                 ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Pipeline: 🔇VAD → 🎙️ASR → 🤖LLM → 🔊TTS                                 │  ║
║  │  Target Latency: < 2000ms end-to-end                                   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  Press Ctrl+C to stop                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    )


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                            FASTAPI APPLICATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

app = FastAPI(
    title="🎤 Voice Agent",
    description="""
## Real-time Vietnamese Voice Agent

End-to-end voice pipeline with state-of-the-art AI models:

- **ASR**: Qwen3-ASR (vLLM) - 52 languages, 2000x throughput
- **TTS**: Qwen3-TTS - 97ms first packet latency
- **LLM**: Groq (Llama 3.3 70B) - 330 tokens/second
- **VAD**: Silero v5 - Real-time voice detection

### WebSocket Protocol

Connect to `/ws/agent` for real-time voice interaction:

```
Client → Server: Binary (PCM S16LE @ 16kHz mono)
Server → Client: Binary (TTS audio) or Text (events)
```

### Events

- `LISTENING` - Ready for voice input
- `PROCESSING` - Transcribing and generating response
- `SPEAKING` - Playing TTS response
- `TRANSCRIPT:<text>` - Recognized speech
- `ERROR:<message>` - Error occurred
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include HTTP routes
app.include_router(router)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                           WEBSOCKET ENDPOINT                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time voice agent.

    Production protocol:
        Client → Server: Binary PCM S16LE @16kHz, or text commands
            (HELLO[:session_id], PONG, INTERRUPT, CLEAR_HISTORY,
            SET_TASK:<name>)
        Server → Client: Binary TTS audio or text events
            (READY, LISTENING, PROCESSING, SPEAKING, IDLE, TRANSCRIPT:,
            RESPONSE_TEXT:, ERROR:, BUSY:, PING)
    """
    # At minimum need VAD to detect speech
    if not vad_service:
        await websocket.close(code=1011, reason="VAD service not ready")
        return

    def task_factory(name: str) -> TextTask:
        if name == "llm":
            if llm_service is None:
                raise ValueError("LLM service is disabled")
            return create_text_task("llm", llm=llm_service)
        if name == "passthrough":
            return create_text_task("passthrough", prefix=settings.text_task_prefix)
        return create_text_task(name)

    def create_orchestrator(session_id: str | None = None) -> Orchestrator:
        runtime = PipelineRuntime(
            asr_timeout_s=settings.asr_timeout_s,
            tts_timeout_s=settings.tts_timeout_s,
            llm_timeout_s=settings.llm_timeout_s,
            llm_first_token_timeout_s=settings.llm_first_token_timeout_s,
            max_retries=settings.max_retries,
            infer_semaphore=_infer_semaphore,
        )
        try:
            task = task_factory(settings.text_task)
        except ValueError:
            task = task_factory("passthrough")
        # Pass None for disabled services - orchestrator degrades gracefully
        return Orchestrator(
            vad=vad_service,
            asr=asr_service,
            llm=llm_service,
            tts=tts_service,
            session_id=session_id,
            text_task=task,
            runtime=runtime,
        )

    await websocket_endpoint(
        websocket,
        create_orchestrator,
        gate=_connection_gate,
        sessions=_session_store,
        task_factory=task_factory,
        ping_interval_s=settings.ws_ping_interval_s,
        ping_timeout_s=settings.ws_ping_timeout_s,
    )


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                                  MAIN                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    """
    Run the Voice Agent server.

    Uses uvicorn with optimized settings for real-time audio processing.
    """
    import uvicorn

    # Uvicorn configuration optimized for real-time audio
    config = uvicorn.Config(
        app="voice_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        # Performance optimizations
        workers=1,  # Single worker for shared GPU memory
        loop="uvloop" if sys.platform != "win32" else "asyncio",
        http="httptools" if sys.platform != "win32" else "h11",
        ws="websockets",
        # Timeout settings for WebSocket
        timeout_keep_alive=30,
        ws_max_size=16 * 1024 * 1024,  # 16MB max WebSocket message
        # Access logging
        access_log=settings.debug,
    )

    server = uvicorn.Server(config)

    # Handle graceful shutdown
    def signal_handler(signum: int, frame: object) -> None:
        logger.info("🛑 received_shutdown_signal", signal=signum)
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("👋 interrupted_by_user")


if __name__ == "__main__":
    main()
