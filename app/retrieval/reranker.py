"""
Cross-Encoder Reranker — Phase 5
==================================
Singleton wrapper around sentence-transformers CrossEncoder.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  - Top-rated open-source reranker on BEIR/MS-MARCO
  - ~85 MB download, CPU-friendly (22 ms / pair on modern CPU)
  - Change RERANKER_MODEL in .env to swap models

The reranker runs in a thread executor to avoid blocking the event loop.
Falls back to score-ordered results if the model is not available.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from app.retrieval.qdrant_client import SearchResult


# ── Singleton state ───────────────────────────────────────────────────────────

_model: Any = None      # sentence_transformers.CrossEncoder
_model_lock = threading.Lock()


def _get_cross_encoder() -> Any:
    """Return the cached CrossEncoder model (thread-safe lazy init)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model_name = settings.reranker_model
            logger.info("Loading cross-encoder reranker '{}' …", model_name)
            _model = CrossEncoder(model_name, max_length=512)
            logger.success("Cross-encoder reranker ready.")
            return _model

        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc


# ── Public functions ──────────────────────────────────────────────────────────

def rerank(
    query: str,
    results: "list[SearchResult]",
    top_k: int,
) -> "list[SearchResult]":
    """
    Rerank retrieval results using the cross-encoder model.

    The cross-encoder sees (query, passage) pairs and produces a calibrated
    relevance score that is much more accurate than cosine similarity alone.

    Args:
        query:   User query string.
        results: Candidate passages from Qdrant (pre-filtered by dense/sparse search).
        top_k:   Maximum number of passages to return after reranking.

    Returns:
        Top ``top_k`` passages ordered by cross-encoder relevance (descending).
        Falls back to cosine-score ordering if the model is unavailable.
    """
    if len(results) <= 1:
        return results[:top_k]

    try:
        model = _get_cross_encoder()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(results, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top = [r for r, _ in ranked[:top_k]]
        logger.debug(
            "Reranked {} → {} results  (cross-encoder top score: {:.4f})",
            len(results), len(top), ranked[0][1],
        )
        return top

    except Exception as exc:
        logger.warning("Cross-encoder reranker failed ({}), using score ordering.", exc)
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


def warm_up() -> None:
    """Pre-load the reranker model so the first query is fast."""
    try:
        _get_cross_encoder()
        logger.info("Cross-encoder reranker warmed up.")
    except Exception as exc:
        logger.warning("Reranker warm-up failed (non-fatal): {}", exc)
