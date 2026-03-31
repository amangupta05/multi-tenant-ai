"""
Dashboard Router — Phase 6 (Dashboard)
========================================
Provides:
  GET /              → redirect to /dashboard/
  GET /health/detail → JSON service status for the dashboard health panel
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

router = APIRouter(tags=["Dashboard"])


@router.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/dashboard/")


@router.get("/health/detail", summary="Detailed service health for the dashboard")
async def health_detail() -> dict:
    """
    Returns live status of every service component.
    Called by the dashboard Overview panel on load and refresh.
    """
    status: dict = {
        "fastapi": {"status": "ok", "detail": "Running"},
        "qdrant": {"status": "unknown", "detail": ""},
        "groq": {"status": "unknown", "detail": ""},
        "gemini": {"status": "unknown", "detail": ""},
        "embedding_model": {"status": "unknown", "detail": ""},
        "bm25_backend": {"status": "unknown", "detail": ""},
        "reranker": {"status": "unknown", "detail": ""},
        "collection": {},
    }

    # ── Qdrant ────────────────────────────────────────────────────────────
    try:
        from app.retrieval.qdrant_client import qdrant_service  # noqa
        info = qdrant_service.collection_info()
        status["qdrant"] = {"status": "ok", "detail": f"{info.get('vectors_count', 0):,} vectors"}
        status["collection"] = info
    except Exception as exc:
        status["qdrant"] = {"status": "error", "detail": str(exc)[:80]}

    # ── Groq API key (Core LLM) ──────────────────────────────────────────
    try:
        from app.config import settings  # noqa
        if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here":
            status["groq"] = {"status": "ok", "detail": settings.groq_model}
        else:
            status["groq"] = {"status": "error", "detail": "GROQ_API_KEY not set"}
    except Exception as exc:
        status["groq"] = {"status": "error", "detail": str(exc)[:80]}

    # ── Gemini API key (Vision) ─────────────────────────────────────────
    try:
        from app.config import settings  # noqa
        if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
            status["gemini"] = {"status": "ok", "detail": "Vision Enabled"}
        else:
            status["gemini"] = {"status": "warning", "detail": "Vision Disabled"}
    except Exception as exc:
        status["gemini"] = {"status": "error", "detail": str(exc)[:80]}

    # ── Embedding model ────────────────────────────────────────────────────
    try:
        from app.ingestion.embedder import _model as embedding_model  # noqa
        if embedding_model is not None:
            from app.config import settings
            status["embedding_model"] = {
                "status": "ok",
                "detail": f"{settings.embedding_model} ({settings.embedding_dimension}d)",
            }
        else:
            from app.config import settings
            status["embedding_model"] = {
                "status": "warning",
                "detail": f"{settings.embedding_model} (not loaded yet)",
            }
    except Exception as exc:
        status["embedding_model"] = {"status": "error", "detail": str(exc)[:80]}

    # ── BM25 sparse backend ────────────────────────────────────────────────
    try:
        from app.retrieval.bm25 import _backend, _model  # noqa
        if _backend == "fastembed":
            status["bm25_backend"] = {"status": "ok", "detail": "fastembed (Qdrant/BM25)"}
        elif _backend == "sklearn":
            status["bm25_backend"] = {"status": "warning", "detail": "sklearn HashingVectorizer (fallback)"}
        else:
            status["bm25_backend"] = {"status": "warning", "detail": "Not loaded (dense-only mode)"}
    except Exception as exc:
        status["bm25_backend"] = {"status": "warning", "detail": f"Not loaded: {exc}"[:80]}

    # ── Reranker ──────────────────────────────────────────────────────────
    try:
        from app.retrieval.reranker import _model as reranker_model  # noqa
        from app.config import settings
        if reranker_model is not None:
            status["reranker"] = {"status": "ok", "detail": settings.reranker_model}
        else:
            status["reranker"] = {"status": "warning", "detail": f"{settings.reranker_model} (not loaded yet)"}
    except Exception as exc:
        status["reranker"] = {"status": "warning", "detail": str(exc)[:80]}

    # ── Config summary ─────────────────────────────────────────────────────
    try:
        from app.config import settings
        status["config"] = {
            "top_k_retrieval": settings.top_k_retrieval,
            "top_k_rerank": settings.top_k_rerank,
            "min_relevance_score": settings.min_relevance_score,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "session_ttl_hours": settings.session_ttl_seconds // 3600,
            "max_conversation_turns": settings.max_conversation_turns,
        }
    except Exception:
        pass

    return status
