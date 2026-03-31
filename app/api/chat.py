"""
Chat API Router — Dedicated backend for the Web Dashboard Chat UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.db import crud, get_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

# ── Security Scheme ────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _get_tenant_from_api_key(
    x_api_key: str = Depends(_api_key_header),
    session: AsyncSession = Depends(get_session),
) -> Any:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    tenant = await crud.get_tenant_by_api_key(session, x_api_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key.")
    if not tenant.is_active:
        raise HTTPException(status_code=403, detail="Tenant account is deactivated.")
    return tenant


# ── Chat Endpoint ──────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """
    Standard chat message payload for the Web Dashboard.
    session_id identifies the thread for memory continuity.
    """
    session_id:   str = "web-session-001"
    message:      str
    message_type: str = "text"    # text | image | audio
    image_url:    str | None = None
    audio_url:    str | None = None


class ChatMessageResponse(BaseModel):
    session_id:        str
    response:          str
    intent:            str
    intent_confidence: float
    escalate:          bool
    safety_flags:      list[str]
    latency_ms:        float
    error:             str | None = None
    # Debug fields (only populated when DEBUG=true in .env)
    debug:             dict | None = None


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Send a message to the AI pipeline",
    description=(
        "Send a message from the Web Dashboard and get the full AI response. "
        "Requires a valid `X-API-Key` header."
    ),
)
async def chat_message(
    body: ChatMessageRequest,
    tenant: Any = Depends(_get_tenant_from_api_key),
) -> ChatMessageResponse:
    """Full pipeline: classify intent → retrieve/act → generate → safety check."""
    from app.core.pipeline import run_pipeline  # lazy import for startup speed

    user_message = body.message
    if body.message_type == "image" and body.image_url:
        user_message = f"[Image received: {body.image_url}] {body.message or 'Please describe this image.'}"
    elif body.message_type == "audio" and body.audio_url:
        user_message = f"[Audio received: {body.audio_url}] {body.message or 'Please transcribe this audio.'}"

    result = await run_pipeline(
        tenant=tenant,
        session_id=body.session_id,
        user_message=user_message,
        input_type=body.message_type,
    )

    debug_info = None
    if settings.debug:
        debug_info = {
            "intent_reasoning": result.get("intent_reasoning"),
            "retrieved_context_preview": (result.get("retrieved_context") or "")[:300],
            "tool_results": result.get("tool_results", []),
        }

    return ChatMessageResponse(
        session_id=body.session_id,
        response=result["response"],
        intent=result["intent"],
        intent_confidence=result["intent_confidence"],
        escalate=result["escalate"],
        safety_flags=result["safety_flags"],
        latency_ms=result["latency_ms"],
        error=result.get("error"),
        debug=debug_info,
    )
