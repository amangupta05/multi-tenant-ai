"""
Pydantic schemas (request/response models) for the API layer.
Keeps the ORM models out of HTTP serialisation concerns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Tenant schemas
# ─────────────────────────────────────────────────────────────────────────────

class TenantCreate(BaseModel):
    """Body for POST /admin/tenants."""

    name: str = Field(..., min_length=2, max_length=255, examples=["Acme Corp"])
    plan_tier: str = Field("free", pattern="^(free|standard|pro|enterprise)$")
    config: dict[str, Any] | None = None
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-tenant settings overrides",
        examples=[{"custom_system_prompt": "You are a helpful assistant for Acme Corp."}],
    )


class TenantUpdate(BaseModel):
    """Body for PATCH /admin/tenants/{id}."""

    name: str | None = Field(None, min_length=2, max_length=255)
    plan_tier: str | None = Field(None, pattern="^(free|standard|pro)$")
    is_active: bool | None = None
    config: dict[str, Any] | None = None


class TenantResponse(BaseModel):
    """Returned by tenant endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    api_key: str
    plan_tier: str
    is_active: bool
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TenantSummary(BaseModel):
    """Lightweight version used in list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    api_key: str           # Included so dashboards can copy the key
    plan_tier: str
    is_active: bool
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
#  Conversation schemas
# ─────────────────────────────────────────────────────────────────────────────

class MessageTurn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Returned by conversation endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    session_id: str
    messages: list[dict[str, Any]]
    session_meta: dict[str, Any]
    created_at: datetime
    updated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
#  Document schemas
# ─────────────────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Returned by document / ingestion endpoints."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    tenant_id: str
    filename: str
    doc_type: str
    status: str
    chunk_count: int | None
    # ORM attribute is doc_metadata (renamed to avoid SQLAlchemy reserved name clash)
    # We read from doc_metadata but expose as 'metadata' in the JSON response.
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="doc_metadata",
    )
    error_message: str | None
    created_at: datetime
    updated_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
#  Generic response envelopes
# ─────────────────────────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):
    """Generic success acknowledgement."""

    success: bool = True
    message: str = "OK"


class ErrorResponse(BaseModel):
    """Generic error response body."""

    success: bool = False
    error: str
    detail: Any | None = None
