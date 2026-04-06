"""HTTP API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from voice_agent import __version__

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy", "version": __version__}


@router.get("/health/ready")
async def ready() -> dict[str, bool]:
    """
    Readiness check endpoint.

    Returns:
        Readiness status
    """
    # TODO: Check if all services are loaded
    return {"ready": True}


@router.get("/info")
async def info() -> dict[str, Any]:
    """
    Service information endpoint.

    Returns:
        Service info including version and models
    """
    from voice_agent.core import DEFAULT_ASR_MODEL, DEFAULT_LLM_MODEL, DEFAULT_TTS_MODEL

    return {
        "name": "voice-agent",
        "version": __version__,
        "models": {
            "asr": DEFAULT_ASR_MODEL,
            "tts": DEFAULT_TTS_MODEL,
            "llm": DEFAULT_LLM_MODEL,
        },
    }
