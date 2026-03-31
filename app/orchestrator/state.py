"""
AgentState — the single source of truth flowing through the LangGraph.
Every node reads from and writes to this TypedDict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """
    Immutable-style state that flows through every node in the graph.
    Nodes return partial dicts; LangGraph merges them automatically.
    """

    # ── Request identity ──────────────────────────────────────────────────
    tenant_id: str
    session_id: str                 # Chat thread UUID or test session ID
    tenant_name: str                # Display name used in prompts
    tenant_config: dict[str, Any]   # Per-tenant settings (custom_prompt, etc.)

    # ── Input ─────────────────────────────────────────────────────────────
    user_message: str       # Text of the user's message (post-multimodal conversion)
    raw_input_type: str     # "text" | "image" | "audio" | "document"

    # ── Routing ───────────────────────────────────────────────────────────
    intent: str             # "rag" | "tool_call" | "chitchat" | "escalate"
    intent_confidence: float
    intent_reasoning: str

    # ── Context ───────────────────────────────────────────────────────────
    conversation_history: list[dict[str, Any]]  # last N turns [{role, content}]
    retrieved_context: str                       # Formatted RAG context string
    tool_results: list[dict[str, Any]]           # [{tool, args, result}]

    # ── Output ────────────────────────────────────────────────────────────
    response: str             # Final generated chat response
    safety_flags: list[str]   # Issues detected by guardrail
    escalate: bool            # True → route to human agent
    error: str | None         # Non-None if a node encountered a fatal error
