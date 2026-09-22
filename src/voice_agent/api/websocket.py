"""WebSocket API handlers with heartbeat, resume, and admission control."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from voice_agent.utils import get_logger, get_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from voice_agent.services import Orchestrator
    from voice_agent.services.text_tasks import TextTask
    from voice_agent.utils.resilience import AdmissionGate
    from voice_agent.utils.session_store import SessionStore

logger = get_logger(__name__)

# Client -> server text commands
CMD_HELLO = "HELLO"
CMD_PONG = "PONG"
CMD_INTERRUPT = "INTERRUPT"
CMD_CLEAR_HISTORY = "CLEAR_HISTORY"
CMD_SET_TASK = "SET_TASK"


class WebSocketHandler:
    """
    WebSocket handler for voice agent communication.

    Binary frames (client -> server) carry PCM S16LE audio.
    Text frames carry events and commands:

    Server -> client:
        READY:<session_id>, LISTENING, PROCESSING, SPEAKING, IDLE,
        TRANSCRIPT:<text>, RESPONSE_TEXT:<sentence>, ERROR:<msg>,
        BUSY:<reason>, PING
    Client -> server:
        HELLO[:<session_id>], PONG, INTERRUPT, CLEAR_HISTORY,
        SET_TASK:<name>
    """

    def __init__(
        self,
        create_orchestrator: Callable[[str | None], Orchestrator],
        gate: AdmissionGate | None = None,
        sessions: SessionStore | None = None,
        task_factory: Callable[[str], TextTask] | None = None,
        ping_interval_s: float = 20.0,
        ping_timeout_s: float = 60.0,
    ) -> None:
        self._create_orchestrator = create_orchestrator
        self._gate = gate
        self._sessions = sessions
        self._task_factory = task_factory
        self._ping_interval_s = ping_interval_s
        self._ping_timeout_s = ping_timeout_s
        self._metrics = get_registry()

    async def handle(self, websocket: WebSocket) -> None:
        # --- admission control (CCU shedding) ---
        if self._gate is not None and not await self._gate.acquire():
            self._metrics.inc("connections_rejected")
            await websocket.close(code=1013, reason="Server busy, try again later")
            logger.warning("connection_rejected_busy")
            return

        orchestrator: Orchestrator | None = None
        session_id: str | None = None
        last_activity = time.monotonic()
        heartbeat: asyncio.Task[None] | None = None
        admitted = self._gate is not None

        async def send_results(results) -> None:  # type: ignore[no-untyped-def]
            async for result in results:
                if isinstance(result, bytes):
                    await websocket.send_bytes(result)
                else:
                    await websocket.send_text(result)

        try:
            await websocket.accept()
            self._metrics.inc("connections_accepted")
            self._metrics.gauge("ccu_current").inc()
            logger.info("websocket_connected")

            async def heartbeat_loop() -> None:
                nonlocal last_activity
                while True:
                    await asyncio.sleep(self._ping_interval_s)
                    if time.monotonic() - last_activity > self._ping_timeout_s:
                        logger.warning("heartbeat_timeout", session_id=session_id)
                        self._metrics.inc("heartbeat_timeouts")
                        with contextlib.suppress(Exception):
                            await websocket.close(code=1011, reason="Heartbeat timeout")
                        return
                    try:
                        await websocket.send_text("PING")
                    except Exception:
                        return

            heartbeat = asyncio.create_task(heartbeat_loop())

            while True:
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    raise
                except RuntimeError:
                    # Starlette raises after disconnect races; exit cleanly.
                    break

                last_activity = time.monotonic()
                msg_type = message.get("type")

                if msg_type == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"] is not None:
                    if orchestrator is None:
                        orchestrator = self._create_orchestrator(session_id)
                        session_id = orchestrator.session_id
                        await websocket.send_text(f"READY:{session_id}")
                    try:
                        await send_results(orchestrator.process_audio(message["bytes"]))
                    except Exception as e:
                        logger.error("process_audio_error", error=str(e))
                        self._metrics.inc("errors_process_audio")
                        with contextlib.suppress(Exception):
                            await websocket.send_text(f"ERROR:{e}")
                        # Recover orchestrator state after error
                        with contextlib.suppress(Exception):
                            await orchestrator.interrupt()
                        with contextlib.suppress(Exception):
                            orchestrator.reset()
                    continue

                text = message.get("text")
                if text is None:
                    continue

                # --- text commands ---
                if text == CMD_PONG:
                    continue
                if text.startswith(CMD_HELLO):
                    requested = text.split(":", 1)[1] if ":" in text else None
                    resumed = False
                    if requested and self._sessions is not None:
                        snap = self._sessions.load(requested)
                        if snap is not None:
                            orchestrator = self._create_orchestrator(requested)
                            orchestrator.restore(snap)
                            session_id = requested
                            resumed = True
                            self._metrics.inc("sessions_resumed")
                    if not resumed:
                        orchestrator = self._create_orchestrator(None)
                        session_id = orchestrator.session_id
                    await websocket.send_text(f"READY:{session_id}")
                    if resumed and orchestrator is not None:
                        last = orchestrator.snapshot()
                        if last.last_transcript:
                            await websocket.send_text(f"TRANSCRIPT:{last.last_transcript}")
                    logger.info("session_ready", session_id=session_id, resumed=resumed)
                    continue
                if orchestrator is None:
                    await websocket.send_text("ERROR:Send HELLO first")
                    continue
                if text == CMD_INTERRUPT:
                    await orchestrator.interrupt()
                elif text == CMD_CLEAR_HISTORY:
                    orchestrator.clear_history()
                    await websocket.send_text("IDLE")
                elif text.startswith(CMD_SET_TASK + ":"):
                    name = text.split(":", 1)[1].strip()
                    if self._task_factory is None:
                        await websocket.send_text("ERROR:Task switching disabled")
                    else:
                        try:
                            orchestrator.set_task(self._task_factory(name))
                            await websocket.send_text(f"TASK:{name}")
                        except ValueError as e:
                            await websocket.send_text(f"ERROR:{e}")
                else:
                    await websocket.send_text(f"ERROR:Unknown command: {text[:32]}")

        except WebSocketDisconnect:
            logger.info("websocket_disconnected", session_id=session_id)
            self._metrics.inc("connections_closed")
        except Exception as e:
            logger.error("websocket_error", session_id=session_id, error=str(e))
            self._metrics.inc("errors_websocket")
            with contextlib.suppress(Exception):
                await websocket.send_text(f"ERROR:{e}")
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            if orchestrator is not None and self._sessions is not None and session_id:
                try:
                    self._sessions.save(orchestrator.snapshot())
                except Exception as e:
                    logger.error("snapshot_failed", error=str(e))
                orchestrator.log_metrics()
            if admitted and self._gate is not None:
                await self._gate.release()
                self._metrics.gauge("ccu_current").dec()
            logger.info("websocket_closed", session_id=session_id)


async def websocket_endpoint(
    websocket: WebSocket,
    orchestrator_or_factory: Orchestrator | Callable[[str | None], Orchestrator],
    gate: AdmissionGate | None = None,
    sessions: SessionStore | None = None,
    task_factory: Callable[[str], TextTask] | None = None,
    ping_interval_s: float = 20.0,
    ping_timeout_s: float = 60.0,
) -> None:
    """
    WebSocket endpoint handler.

    Accepts either a ready Orchestrator (backward compat: single-session
    mode) or a factory creating one per connection (production mode).
    """
    if callable(orchestrator_or_factory) and not hasattr(orchestrator_or_factory, "process_audio"):
        factory = orchestrator_or_factory
    else:
        ready = orchestrator_or_factory

        def factory(_sid: str | None = None) -> Orchestrator:
            return ready  # type: ignore[return-value]

    handler = WebSocketHandler(
        factory,
        gate=gate,
        sessions=sessions,
        task_factory=task_factory,
        ping_interval_s=ping_interval_s,
        ping_timeout_s=ping_timeout_s,
    )
    await handler.handle(websocket)
