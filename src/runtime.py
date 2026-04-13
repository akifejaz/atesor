"""
Runtime settings for the agent workflow.
Centralizes operational thresholds and environment-driven behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from .config import CACHE_DIR


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    """Environment-driven runtime behavior settings."""

    routing_api_call_threshold: int
    routing_cost_threshold: float
    max_total_cost_usd: float
    max_audit_events: int
    max_feedback_issues: int
    knowledge_snapshot_label: str
    session_db_path: str
    persist_sessions: bool


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    """Load and cache runtime settings from environment variables."""
    return RuntimeSettings(
        routing_api_call_threshold=_env_int("ATESOR_ROUTING_API_CALL_THRESHOLD", 8),
        routing_cost_threshold=_env_float("ATESOR_ROUTING_COST_THRESHOLD", 0.08),
        max_total_cost_usd=_env_float("ATESOR_MAX_TOTAL_COST_USD", 1.0),
        max_audit_events=_env_int("ATESOR_MAX_AUDIT_EVENTS", 5000),
        max_feedback_issues=_env_int("ATESOR_MAX_FEEDBACK_ISSUES", 5),
        knowledge_snapshot_label=os.getenv(
            "ATESOR_KNOWLEDGE_SNAPSHOT",
            datetime.now().strftime("%B %Y"),
        ),
        session_db_path=os.getenv(
            "ATESOR_SESSION_DB_PATH",
            os.path.join(CACHE_DIR, "agent_sessions.db"),
        ),
        persist_sessions=_env_bool("ATESOR_PERSIST_SESSIONS", True),
    )


__all__ = ["RuntimeSettings", "get_runtime_settings"]
