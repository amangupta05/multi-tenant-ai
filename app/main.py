"""
Multi-Tenant AI System — FastAPI Application Entry Point
=========================================================

Start locally (Phase 1 + 2):
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.db import close_db, init_db, ping_db


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """Replace the default Loguru handler with one that matches our log level."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )
    # Also write to a rotating file log
    settings.ensure_dirs()
    logger.add(
        str(settings.data_path / "logs" / "app.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    )


# ── Lifespan: startup & shutdown hooks ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """
    Everything before the yield runs on startup.
    Everything after yields runs on shutdown.
    """
    _configure_logging()
    _print_banner()

    # ── Startup ───────────────────────────────────────────────────
    logger.info("🚀  Starting up {}  v{}", settings.app_name, settings.app_version)

    # 1. Create data directories
    settings.ensure_dirs()
    logger.info("📁  Data directory ready: {}", settings.data_dir)

    # 2. Initialise SQLite database + create tables
    logger.info("🗄️   Initialising database …")
    await init_db()
    if await ping_db():
        logger.success("✅  Database connected ({})", settings.database_url)
    else:
        logger.error("❌  Database connection failed!")

    # 3. Initialise Qdrant local vector store
    logger.info("📦  Initialising Qdrant local vector store …")
    try:
        from app.retrieval.qdrant_client import qdrant_service
        qdrant_service.init()
        logger.success("✅  Qdrant ready ({} vectors)", qdrant_service.collection_info().get("vectors_count", 0))
    except Exception as exc:
        logger.warning("⚠️   Qdrant init failed (install qdrant-client): {}", exc)

    logger.success("🎉  Application ready")
    logger.success("   Dashboard → http://localhost:8000/")
    logger.success("   API Docs  → http://localhost:8000/docs")

    yield  # ── Application runs here ──────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("🛑  Shutting down …")
    await close_db()
    logger.info("🗄️   Database connection closed.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Zero-cost multi-tenant AI backend — Web Chat + RAG + LangGraph + MCP.\n\n"
        "**Authentication:** Include `X-Admin-Key: admin-secret` for admin endpoints.\n\n"
        "For tenant endpoints, include `X-API-Key: <tenant_api_key>`."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next: Any) -> Any:
    """Log every request with method, path, status, and latency."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.debug(
        "{} {} → {} ({:.1f} ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Enable DEBUG=true for details.",
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────

from app.api import admin, dashboard, ingest, chat  # noqa: E402
from app.api.dashboard import router as dashboard_router  # noqa: E402

API_PREFIX = "/api/v1"

# Mount dashboard static files (index.html + assets)
import os  # noqa: E402
_dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")

app.include_router(dashboard_router)   # GET / → redirect, GET /health/detail
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(ingest.router,  prefix=API_PREFIX)
app.include_router(admin.router,   prefix=API_PREFIX)


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard/")


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="System health status",
)
async def health_check() -> dict:
    """
    Returns liveness status of all subsystems.
    All future phases add their components here too.
    """
    db_ok = await ping_db()

    status_code = "healthy" if db_ok else "degraded"

    # Check Qdrant
    qdrant_ok = False
    qdrant_info: dict = {}
    try:
        from app.retrieval.qdrant_client import qdrant_service
        qdrant_info = qdrant_service.collection_info()
        qdrant_ok = True
    except Exception:
        pass

    all_ok = db_ok and qdrant_ok
    health_status = "healthy" if all_ok else "degraded"

    return {
        "status": health_status,
        "version": settings.app_version,
        "components": {
            "database": "ok" if db_ok else "error",
            "vector_store": "ok" if qdrant_ok else "error",
            "vector_store_info": qdrant_info,
            "embedding_model": settings.embedding_model,
            "mcp_servers": "pending_phase_3",
            "orchestrator": "pending_phase_4",
        },
    }


# ── ASCII banner ──────────────────────────────────────────────────────────────

def _print_banner() -> None:  # pragma: no cover
    banner = r"""
  __  __ _   _ _ _   _     _____                      _       _
 |  \/  | | | | | |_(_)   |_   _|__ _ __   __ _ _ __ | |_    / \   ___ _
 | |\/| | | | | | __| |___  | |/ _ \ '_ \ / _` | '_ \| __|  / _ \ | |_) |
 | |  | | |_| | | |_| |___| | |  __/ | | | (_| | | | | |_  / ___ \|  __/
 |_|  |_|\___/|_|\__|_|     |_|\___|_| |_|\__,_|_| |_|\__| /_/   \_\_|

  Web Dashboard · RAG · LangGraph · MCP · Zero Cost MVP
    """
    print(banner)
