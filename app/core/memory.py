"""
In-memory session memory backed by cachetools TTLCache.
Stores the last N conversation turns per (tenant_id, session_id) pair.

Design
------
- Pure in-process, zero dependencies beyond cachetools.
- Thread-safe via threading.Lock (FastAPI runs in a thread pool for sync ops).
- TTL matches settings.session_ttl_seconds (default 24 h).
- Keys: "{tenant_id}:{session_id}" to prevent cross-tenant leakage.
- In production, swap this module for a Redis-backed implementation with
  the same public interface — no other code changes needed.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from cachetools import TTLCache
from loguru import logger

from app.config import settings

# ── Singleton cache ───────────────────────────────────────────────────────────

_cache: TTLCache = TTLCache(
    maxsize=50_000,                    # max concurrent sessions
    ttl=settings.session_ttl_seconds,  # entries expire after 24 h of inactivity
)
_lock = threading.Lock()


# ── Cache key ─────────────────────────────────────────────────────────────────

def _key(tenant_id: str, session_id: str) -> str:
    return f"{tenant_id}:{session_id}"


# ── Public interface ──────────────────────────────────────────────────────────

def load_history(tenant_id: str, session_id: str) -> list[dict[str, Any]]:
    """
    Return the stored conversation history for a session.
    Returns an empty list if the session is new or has expired.
    """
    with _lock:
        return list(_cache.get(_key(tenant_id, session_id), []))


def save_turn(
    tenant_id: str,
    session_id: str,
    *,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Append a single turn to the session history.

    Args:
        role:    "user" | "assistant"
        content: Message text
        metadata: Optional extra info (intent, latency, etc.)
    """
    k = _key(tenant_id, session_id)
    with _lock:
        history: list[dict[str, Any]] = list(_cache.get(k, []))
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        })
        # Keep only the most recent turns to stay within context budget
        max_turns = settings.max_conversation_turns * 2  # × 2 for user + assistant
        if len(history) > max_turns:
            history = history[-max_turns:]
        _cache[k] = history
    logger.debug("Memory saved: tenant='{}' session='{}' turns={}", tenant_id, session_id, len(history))


def get_active_sessions(tenant_id: str) -> list[str]:
    """Return all active session IDs for a tenant (for monitoring)."""
    prefix = f"{tenant_id}:"
    with _lock:
        return [k.split(":", 1)[1] for k in _cache.keys() if k.startswith(prefix)]


def clear_session(tenant_id: str, session_id: str) -> None:
    """Explicitly evict a session (e.g. on logout or tenant deletion)."""
    with _lock:
        _cache.pop(_key(tenant_id, session_id), None)


def format_history_for_prompt(history: list[dict[str, Any]], max_turns: int = 6) -> str:
    """
    Convert stored history into a compact string for LLM prompts.
    Only includes the last ``max_turns`` exchanges.
    """
    if not history:
        return "(No previous conversation.)"

    recent = history[-(max_turns * 2):]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")[:400]  # truncate long turns
        lines.append(f"{role}: {content}")

    return "\n".join(lines)
