"""
SQLite-backed persistence for workflow sessions and state snapshots.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from .runtime import get_runtime_settings
from .state import AgentState


class SessionStore:
    """Persist workflow sessions, state snapshots, and events."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_runtime_settings()
        self.db_path = db_path or settings.session_db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    repo_url TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    exit_code INTEGER,
                    final_state_json TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    build_status TEXT,
                    current_phase TEXT,
                    created_at TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    agent TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )

    def create_session(
        self,
        repo_url: str,
        max_attempts: int,
        session_id: Optional[str] = None,
    ) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    repo_url,
                    max_attempts,
                    status,
                    started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, repo_url, max_attempts, "running", now),
            )

        return session_id

    def save_snapshot(
        self,
        session_id: str,
        step: int,
        node_name: str,
        state: AgentState,
    ) -> None:
        state_json = json.dumps(state.to_dict(), default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_snapshots (
                    session_id,
                    step,
                    node_name,
                    build_status,
                    current_phase,
                    created_at,
                    state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    step,
                    node_name,
                    getattr(state.build_status, "value", str(state.build_status)),
                    state.current_phase,
                    datetime.now().isoformat(),
                    state_json,
                ),
            )

    def save_events(
        self,
        session_id: str,
        step: int,
        events: Iterable[Dict[str, Any]],
    ) -> None:
        rows = []
        for event in events:
            rows.append(
                (
                    session_id,
                    step,
                    event.get("event", "unknown"),
                    event.get("agent"),
                    json.dumps(event.get("data", {}), default=str),
                    event.get("timestamp", datetime.now().isoformat()),
                )
            )

        if not rows:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO session_events (
                    session_id,
                    step,
                    event_type,
                    agent,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def finish_session(
        self,
        session_id: str,
        final_state: Optional[AgentState],
        exit_code: int,
    ) -> None:
        final_state_json = None
        if final_state is not None:
            final_state_json = json.dumps(final_state.to_dict(), default=str)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = ?, ended_at = ?, exit_code = ?, final_state_json = ?
                WHERE session_id = ?
                """,
                (
                    "completed" if exit_code == 0 else "failed",
                    datetime.now().isoformat(),
                    exit_code,
                    final_state_json,
                    session_id,
                ),
            )

    def get_latest_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT step, node_name, state_json
                FROM state_snapshots
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        return {
            "step": row[0],
            "node_name": row[1],
            "state": json.loads(row[2]),
        }


__all__ = ["SessionStore"]
