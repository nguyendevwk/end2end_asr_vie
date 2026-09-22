"""HTTP API routes: health probes and observability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from voice_agent import __version__
from voice_agent.core import DEFAULT_ASR_MODEL, DEFAULT_LLM_MODEL, DEFAULT_TTS_MODEL
from voice_agent.services import available_tasks
from voice_agent.utils import get_registry
from voice_agent.utils.session_store import get_session_store

if TYPE_CHECKING:
    from voice_agent.config import Settings

router = APIRouter()

# Module-level state container for dependency injection (set by main.py at startup)
_app_state: dict[str, Any] = {}


def set_app_state(settings: "Settings", services: dict[str, Any]) -> None:
    """Called by main.py lifespan to inject state into routes."""
    _app_state["settings"] = settings
    _app_state["services"] = services


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe (backward compatible)."""
    return {"status": "healthy", "version": __version__}


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Kubernetes-style liveness: process is running."""
    return {"status": "alive", "version": __version__}


@router.get("/health/ready")
async def ready() -> JSONResponse:
    """Readiness: all enabled services initialized."""
    settings = _app_state.get("settings")
    services = _app_state.get("services", {})

    if settings is None:
        return JSONResponse({"ready": False, "error": "app not initialized"}, status_code=503)

    checks: dict[str, str] = {}
    ok = True

    def check(name: str, enabled: bool, svc: object) -> None:
        nonlocal ok
        if not enabled:
            checks[name] = "disabled"
        elif svc is not None and getattr(svc, "is_started", False):
            checks[name] = "ready"
        else:
            checks[name] = "not_ready"
            ok = False

    check("vad", settings.vad_enabled, services.get("vad"))
    check("asr", settings.asr_enabled, services.get("asr"))
    check("tts", settings.tts_enabled, services.get("tts"))
    check("llm", settings.llm_enabled, services.get("llm"))

    status = 200 if ok else 503
    return JSONResponse({"ready": ok, "services": checks}, status_code=status)


@router.get("/info")
async def info() -> dict[str, Any]:
    """Service information including version and models."""
    return {
        "name": "voice-agent",
        "version": __version__,
        "models": {
            "asr": DEFAULT_ASR_MODEL,
            "tts": DEFAULT_TTS_MODEL,
            "llm": DEFAULT_LLM_MODEL,
        },
        "text_tasks": available_tasks(),
    }


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus text exposition."""
    registry = get_registry()
    registry.gauge("sessions_stored").set(get_session_store().size)
    return PlainTextResponse(registry.to_prometheus(), media_type="text/plain")


@router.get("/metrics.json")
async def metrics_json() -> dict[str, Any]:
    """Machine-readable metrics snapshot."""
    data = get_registry().to_dict()
    data["sessions_stored"] = get_session_store().size
    return data
