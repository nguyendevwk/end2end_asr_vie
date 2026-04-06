"""WebSocket API handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from voice_agent.utils import get_logger

if TYPE_CHECKING:
    from voice_agent.services import Orchestrator

logger = get_logger(__name__)


class WebSocketHandler:
    """
    WebSocket handler for voice agent communication.

    Handles bidirectional audio streaming with:
    - Incoming: PCM S16LE audio chunks from client
    - Outgoing: Audio responses + text events

    Protocol:
        Client -> Server: Binary (PCM audio chunks)
        Server -> Client: Binary (TTS audio) or Text (events)

    Text events:
        - "LISTENING": VAD detected speech start
        - "PROCESSING": Processing user utterance
        - "SPEAKING": Bot is responding
        - "IDLE": Ready for next input
        - "TRANSCRIPT:text": User transcript
        - "ERROR:message": Error occurred
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Initialize handler.

        Args:
            orchestrator: Pipeline orchestrator instance
        """
        self._orchestrator = orchestrator
        self._connected = False

    async def handle(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()
        self._connected = True
        session_id = self._orchestrator.session_id

        logger.info("websocket_connected", session_id=session_id)

        try:
            await self._receive_loop(websocket)

        except WebSocketDisconnect:
            logger.info("websocket_disconnected", session_id=session_id)

        except Exception as e:
            logger.error("websocket_error", session_id=session_id, error=str(e))
            try:
                await websocket.send_text(f"ERROR:{e}")
            except Exception:
                pass

        finally:
            self._connected = False
            self._orchestrator.reset()
            self._orchestrator.log_metrics()
            logger.info("websocket_closed", session_id=session_id)

    async def _receive_loop(self, websocket: WebSocket) -> None:
        """
        Main receive loop for audio chunks.

        Args:
            websocket: WebSocket connection
        """
        while self._connected:
            try:
                # Receive audio chunk (binary)
                audio_chunk = await websocket.receive_bytes()

                # Process and send responses
                async for result in self._orchestrator.process_audio(audio_chunk):
                    if isinstance(result, bytes):
                        # Audio response
                        await websocket.send_bytes(result)
                    else:
                        # Text event
                        await websocket.send_text(result)

            except WebSocketDisconnect:
                raise

            except asyncio.CancelledError:
                logger.info("receive_loop_cancelled")
                break

            except Exception as e:
                logger.error("receive_loop_error", error=str(e))
                await websocket.send_text(f"ERROR:{e}")


async def websocket_endpoint(
    websocket: WebSocket,
    orchestrator: Orchestrator,
) -> None:
    """
    WebSocket endpoint handler.

    Args:
        websocket: FastAPI WebSocket
        orchestrator: Pipeline orchestrator
    """
    handler = WebSocketHandler(orchestrator)
    await handler.handle(websocket)
