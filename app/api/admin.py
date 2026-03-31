"""
Admin API — Tenant management endpoints.
All routes are prefixed with /api/v1/admin.
Authentication: X-Admin-Key header (set ADMIN_API_KEY in .env — defaults to "admin-secret" for MVP).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.db import crud
from app.db.schemas import (
    SuccessResponse,
    TenantCreate,
    TenantResponse,
    TenantSummary,
    TenantUpdate,
)

router = APIRouter(prefix="/admin", tags=["Admin — Tenant Management"])

# ── Security Scheme (Shows 'Authorize' button in Swagger) ─────────────────────

admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def verify_admin_key(x_admin_key: str = Depends(admin_key_header)) -> None:
    """
    Dependency that validates the X-Admin-Key header.
    Automatically shows the 'Authorize' lock icon in Swagger UI.
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Admin-Key header.",
        )

    expected_key = getattr(settings, "admin_api_key", "admin-secret")
    if x_admin_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-Admin-Key.",
        )


# ── Tenant endpoints ──────────────────────────────────────────────────────────

@router.post(
    "/tenants",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new tenant",
    description=(
        "Creates a new tenant and returns their auto-generated API key. "
        "Store the api_key securely — it is only shown once at creation time."
    ),
)
async def create_tenant(
    body: TenantCreate,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_admin_key),
) -> TenantResponse:
    tenant = await crud.create_tenant(
        session,
        name=body.name,
        plan_tier=body.plan_tier,
        config=body.config,
    )
    return TenantResponse.model_validate(tenant)


@router.get(
    "/tenants",
    response_model=list[TenantSummary],
    summary="List all tenants",
)
async def list_tenants(
    active_only: bool = Query(True, description="Return only active tenants"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_admin_key),
) -> list[TenantSummary]:
    tenants = await crud.list_tenants(
        session, active_only=active_only, offset=offset, limit=limit
    )
    return [TenantSummary.model_validate(t) for t in tenants]


@router.get(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    summary="Get a tenant by ID",
)
async def get_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_admin_key),
) -> TenantResponse:
    tenant = await crud.get_tenant_by_id(session, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
    return TenantResponse.model_validate(tenant)


@router.patch(
    "/tenants/{tenant_id}",
    response_model=TenantResponse,
    summary="Update a tenant",
)
async def update_tenant(
    tenant_id: str,
    body: TenantUpdate,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_admin_key),
) -> TenantResponse:
    tenant = await crud.update_tenant(
        session,
        tenant_id,
        name=body.name,
        plan_tier=body.plan_tier,
        is_active=body.is_active,
        config=body.config,
    )
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
    return TenantResponse.model_validate(tenant)


@router.delete(
    "/tenants/{tenant_id}",
    response_model=SuccessResponse,
    summary="Soft-delete a tenant",
)
async def delete_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_admin_key),
) -> SuccessResponse:
    success = await crud.delete_tenant(session, tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
    return SuccessResponse(message=f"Tenant '{tenant_id}' deactivated.")



