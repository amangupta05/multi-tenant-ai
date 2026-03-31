"""
Memory Nodes — load_memory and save_memory.
These two nodes bookend every graph run to maintain conversation continuity.
"""

from __future__ import annotations

from loguru import logger

from app.core import memory as mem
from app.orchestrator.state import AgentState


async def memory_load_node(state: AgentState) -> dict:
    """
    Load the last N conversation turns from TTLCache into the state.
    Runs before the supervisor so the LLM has conversation context.
    """
    history = mem.load_history(state["tenant_id"], state["session_id"])
    logger.debug(
        "Memory load | tenant='{}' session='{}' turns={}",
        state["tenant_id"], state["session_id"], len(history),
    )
    return {"conversation_history": history}


async def memory_save_node(state: AgentState) -> dict:
    """
    Persist the current turn (user + assistant) to TTLCache,
    and optionally archive to the database (async, best-effort).
    Must run AFTER the generate and guardrail nodes so the final
    response is available.
    """
    response = state.get("response", "")
    if not response:
        return {}  # Nothing to save if no response was generated

    tenant_id = state["tenant_id"]
    session_id = state["session_id"]

    mem.save_turn(tenant_id, session_id, role="user", content=state["user_message"])
    mem.save_turn(
        tenant_id,
        session_id,
        role="assistant",
        content=response,
        metadata={
            "intent": state.get("intent"),
            "escalate": state.get("escalate", False),
        },
    )

    # ── Best-effort DB archive (non-blocking) ─────────────────────────────
    try:
        from app.db.database import _get_session_factory  # noqa: PLC0415
        from app.db import crud  # noqa: PLC0415

        factory = _get_session_factory()
        async with factory() as session:
            await crud.append_message(
                session,
                tenant_id=tenant_id,
                session_id=session_id,
                role="user",
                content=state["user_message"],
            )
            await crud.append_message(
                session,
                tenant_id=tenant_id,
                session_id=session_id,
                role="assistant",
                content=response,
                metadata={"intent": state.get("intent"), "escalate": state.get("escalate", False)},
            )
            await session.commit()
    except Exception as exc:
        # DB archive failure must not break the response flow
        logger.warning("DB memory archive failed (non-fatal): {}", exc)

    logger.debug("Memory saved | tenant='{}' session='{}'", tenant_id, session_id)
    return {}
