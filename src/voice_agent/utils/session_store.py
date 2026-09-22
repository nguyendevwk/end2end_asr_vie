"""Reconnectable session snapshots with TTL (survives transient disconnects)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class SessionSnapshot:
    """Minimal state needed to resume a voice session."""

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    turn_id: int = 0
    last_transcript: str = ""
    last_state: str = "IDLE"
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """Thread-safe TTL store for session snapshots."""

    def __init__(self, ttl_s: float = 300.0, max_sessions: int = 1000) -> None:
        self._ttl_s = ttl_s
        self._max = max_sessions
        self._sessions: dict[str, SessionSnapshot] = {}
        self._lock = threading.Lock()

    def save(self, snapshot: SessionSnapshot) -> None:
        snapshot.updated_at = time.time()
        with self._lock:
            if len(self._sessions) >= self._max and snapshot.session_id not in self._sessions:
                # Evict oldest to bound memory
                oldest = min(self._sessions, key=lambda k: self._sessions[k].updated_at)
                del self._sessions[oldest]
            self._sessions[snapshot.session_id] = snapshot

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            snap = self._sessions.get(session_id)
            if snap is None:
                return None
            if time.time() - snap.updated_at > self._ttl_s:
                del self._sessions[session_id]
                return None
            return snap

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def sweep(self) -> int:
        """Remove expired snapshots. Returns number evicted."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._sessions.items() if now - v.updated_at > self._ttl_s]
            for k in expired:
                del self._sessions[k]
            return len(expired)

    @property
    def size(self) -> int:
        return len(self._sessions)


_store = SessionStore()


def get_session_store() -> SessionStore:
    """Return the process-wide session store."""
    return _store
