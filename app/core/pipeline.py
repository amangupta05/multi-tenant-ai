"""
Pipeline — main entry point for processing any incoming message.

Usage (from webhook / tests):
    from app.core.pipeline import run_pipeline

    result = await run_pipeline(
        tenant=tenant_orm_obj,
        session_id="+919876543210",
        user_message="What is your return policy?",
        input_type="text",
    )
    print(result["response"])
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from app.orchestrator.graph import get_compiled_graph
from app.orchestrator.state import AgentState


async def run_pipeline(
    *,
    tenant: Any,            # Tenant ORM object from DB (has .id, .name, .config)
    session_id: str,        # Chat thread ID / Web session UUID
    user_message: str,
    input_type: str = "text",
) -> dict[str, Any]:
    """
    Run the full LangGraph pipeline for one message turn.

    Returns a dict with at minimum:
        response      — the final chat reply
        intent        — classified intent
        intent_confidence
        escalate      — True if human handoff is needed
        safety_flags  — list of triggered safety flags
        latency_ms    — end-to-end processing time
        error         — None on success, error string on failure
    """
    graph = get_compiled_graph()
    t0 = time.perf_counter()

    # Build initial state
    initial: AgentState = {
        # Identity
        "tenant_id":     tenant.id,
        "session_id":    session_id,
        "tenant_name":   tenant.name,
        "tenant_config": tenant.config or {},
        # Input
        "user_message":  user_message.strip(),
        "raw_input_type": input_type,
        # Placeholders — nodes will populate these
        "intent":              "",
        "intent_confidence":   0.0,
        "intent_reasoning":    "",
        "conversation_history": [],
        "retrieved_context":   "",
        "tool_results":        [],
        "response":            "",
        "safety_flags":        [],
        "escalate":            False,
        "error":               None,
    }

    logger.info(
        "Pipeline start | tenant='{}' session='{}' type='{}' msg='{}'",
        tenant.id, session_id, input_type, user_message[:80],
    )

    try:
        final_state: AgentState = await graph.ainvoke(initial)
    except Exception as exc:
        logger.error("Pipeline error | tenant='{}': {}", tenant.id, exc)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "response": (
                "I'm sorry, I'm experiencing a technical issue right now. "
                "Please try again in a moment or contact our support team."
            ),
            "intent": "unknown",
            "intent_confidence": 0.0,
            "escalate": False,
            "safety_flags": [],
            "latency_ms": latency_ms,
            "error": str(exc),
        }

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.success(
        "Pipeline done | intent='{}' escalate={} latency={}ms",
        final_state.get("intent"), final_state.get("escalate"), latency_ms,
    )

    return {
        "response":           final_state.get("response", ""),
        "intent":             final_state.get("intent", ""),
        "intent_confidence":  final_state.get("intent_confidence", 0.0),
        "intent_reasoning":   final_state.get("intent_reasoning", ""),
        "retrieved_context":  final_state.get("retrieved_context", ""),
        "tool_results":       final_state.get("tool_results", []),
        "escalate":           final_state.get("escalate", False),
        "safety_flags":       final_state.get("safety_flags", []),
        "latency_ms":         latency_ms,
        "error":              final_state.get("error"),
    }
