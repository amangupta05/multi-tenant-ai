"""
Preprocess Node — validates and normalises the incoming message.
For text input: no-op pass-through.
For image/audio: converts to text description (Gemini Vision / Whisper).
"""

from __future__ import annotations

from loguru import logger

from app.orchestrator.state import AgentState


async def preprocess_node(state: AgentState) -> dict:
    """
    Entry node — ensures user_message is clean text before routing.

    For MVP the webhook already converts image/audio to text before calling
    the pipeline, so this node mainly sanitises and logs the input type.
    """
    message = (state.get("user_message") or "").strip()
    input_type = state.get("raw_input_type", "text")

    if not message:
        # Provide a fallback so the pipeline doesn't fail on empty input
        message = "[Empty message received]"
        logger.warning(
            "Empty message received | tenant='{}' session='{}'",
            state.get("tenant_id"), state.get("session_id"),
        )

    logger.info(
        "▶ preprocess | tenant='{}' session='{}' type='{}' msg='{}'",
        state.get("tenant_id"),
        state.get("session_id"),
        input_type,
        message[:80],
    )

    return {
        "user_message": message,
        "raw_input_type": input_type,
        "retrieved_context": "",
        "tool_results": [],
        "safety_flags": [],
        "escalate": False,
        "error": None,
    }
