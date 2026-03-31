"""
Async CRUD operations for Tenant, Conversation, and Document models.
All functions accept an AsyncSession and are fully type-annotated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Document, Tenant


# ─────────────────────────────────────────────────────────────────────────────
#  Tenant CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_tenant(
    session: AsyncSession,
    *,
    name: str,
    plan_tier: str = "free",
    config: dict[str, Any] | None = None,
) -> Tenant:
    """Create and persist a new tenant. Returns the saved instance (with api_key)."""
    tenant = Tenant(
        name=name,
        plan_tier=plan_tier,
        config=config or {},
    )
    session.add(tenant)
    await session.flush()  # populate id without committing
    await session.refresh(tenant)
    return tenant


async def get_tenant_by_id(
    session: AsyncSession, tenant_id: str
) -> Tenant | None:
    """Fetch a tenant by primary key."""
    return await session.get(Tenant, tenant_id)


async def get_tenant_by_api_key(
    session: AsyncSession, api_key: str
) -> Tenant | None:
    """Fetch an active tenant by their API key — used in auth middleware."""
    result = await session.execute(
        select(Tenant).where(Tenant.api_key == api_key, Tenant.is_active == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


    return result.scalar_one_or_none()


async def list_tenants(
    session: AsyncSession,
    *,
    active_only: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> list[Tenant]:
    """Return a paginated list of tenants."""
    query = select(Tenant).offset(offset).limit(limit).order_by(Tenant.created_at.desc())
    if active_only:
        query = query.where(Tenant.is_active == True)  # noqa: E712
    result = await session.execute(query)
    return list(result.scalars().all())


async def update_tenant(
    session: AsyncSession,
    tenant_id: str,
    *,
    name: str | None = None,
    plan_tier: str | None = None,
    is_active: bool | None = None,
    config: dict[str, Any] | None = None,
) -> Tenant | None:
    """Partially update a tenant. Returns the updated instance or None if not found."""
    tenant = await get_tenant_by_id(session, tenant_id)
    if tenant is None:
        return None

    if name is not None:
        tenant.name = name
    if plan_tier is not None:
        tenant.plan_tier = plan_tier
    if is_active is not None:
        tenant.is_active = is_active
    if config is not None:
        # Merge rather than replace, so partial updates are non-destructive
        tenant.config = {**tenant.config, **config}

    tenant.updated_at = datetime.now(timezone.utc)
    await session.flush()
    await session.refresh(tenant)
    return tenant


async def delete_tenant(session: AsyncSession, tenant_id: str) -> bool:
    """Soft-delete a tenant (sets is_active=False). Returns True if found."""
    tenant = await get_tenant_by_id(session, tenant_id)
    if tenant is None:
        return False
    tenant.is_active = False
    tenant.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Conversation CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def get_or_create_conversation(
    session: AsyncSession,
    *,
    tenant_id: str,
    session_id: str,
    session_meta: dict[str, Any] | None = None,
) -> Conversation:
    """
    Return an existing conversation for (tenant_id, session_id),
    or create a new one if it doesn't exist yet.
    """
    result = await session.execute(
        select(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.session_id == session_id,
        )
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        conversation = Conversation(
            tenant_id=tenant_id,
            session_id=session_id,
            messages=[],
            session_meta=session_meta or {},
        )
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)

    return conversation


async def append_message(
    session: AsyncSession,
    *,
    tenant_id: str,
    session_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Conversation:
    """
    Append a message turn to the conversation archive.
    Creates the conversation record if it doesn't already exist.
    """
    conversation = await get_or_create_conversation(
        session, tenant_id=tenant_id, session_id=session_id
    )
    turn = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    # SQLAlchemy JSON columns require reassignment to detect mutations
    conversation.messages = [*conversation.messages, turn]
    conversation.updated_at = datetime.now(timezone.utc)
    await session.flush()
    await session.refresh(conversation)
    return conversation


async def get_conversation(
    session: AsyncSession, *, tenant_id: str, session_id: str
) -> Conversation | None:
    """Fetch a conversation by (tenant_id, session_id)."""
    result = await session.execute(
        select(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.session_id == session_id,
        )
    )
    return result.scalar_one_or_none()


async def list_conversations(
    session: AsyncSession,
    *,
    tenant_id: str,
    offset: int = 0,
    limit: int = 20,
) -> list[Conversation]:
    """List recent conversations for a tenant, newest first."""
    result = await session.execute(
        select(Conversation)
        .where(Conversation.tenant_id == tenant_id)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
#  Document CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_document(
    session: AsyncSession,
    *,
    tenant_id: str,
    filename: str,
    file_path: str,
    doc_type: str,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Register a new document as 'pending' ingestion."""
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        file_path=file_path,
        doc_type=doc_type,
        status="pending",
        doc_metadata=metadata or {},
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


async def get_document(
    session: AsyncSession, doc_id: str
) -> Document | None:
    """Fetch a document by primary key."""
    return await session.get(Document, doc_id)


async def update_document_status(
    session: AsyncSession,
    doc_id: str,
    *,
    status: str,
    chunk_count: int | None = None,
    error_message: str | None = None,
    metadata_update: dict[str, Any] | None = None,
) -> Document | None:
    """
    Update status (and optionally chunk_count / error_message) after processing.

    status must be one of: pending | processing | completed | failed
    """
    doc = await get_document(session, doc_id)
    if doc is None:
        return None

    doc.status = status
    doc.updated_at = datetime.now(timezone.utc)
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    if error_message is not None:
        doc.error_message = error_message
    if metadata_update:
        doc.doc_metadata = {**doc.doc_metadata, **metadata_update}

    await session.flush()
    await session.refresh(doc)
    return doc


async def list_documents(
    session: AsyncSession,
    *,
    tenant_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> list[Document]:
    """List documents for a tenant, optionally filtered by status."""
    query = (
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if status:
        query = query.where(Document.status == status)
    result = await session.execute(query)
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, doc_id: str) -> bool:
    """Hard-delete a document record. Returns True if found and deleted."""
    doc = await get_document(session, doc_id)
    if doc is None:
        return False
    await session.delete(doc)
    await session.flush()
    return True
