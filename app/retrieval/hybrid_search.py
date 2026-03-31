"""
Hybrid Retrieval Pipeline — Phase 5
======================================
Orchestrates dense + sparse retrieval with RRF fusion and cross-encoder reranking.

Pipeline
--------
  query
    ├── embed_query()            → 768-dim dense vector (nomic)
    └── encode_sparse_query()   → BM25 sparse vector (fastembed/sklearn)
          │
          ▼
    qdrant.hybrid_search()      → up to candidate_k results via Qdrant RRF
          │
          ▼
    reranker.rerank()           → top_k results, cross-encoder scored
          │
          ▼
    list[SearchResult]

Fallback chain (automatic, transparent)
-----------------------------------------
  1. Hybrid (dense + sparse) + cross-encoder rerank  ← best quality
  2. Dense-only + cross-encoder rerank               ← if sparse encoding fails
  3. Dense-only, cosine-score order                  ← if reranker also fails

Quality vs speed trade-off settings (via .env or per-call overrides):
  TOP_K_RETRIEVAL = 20   → candidates fetched from Qdrant
  TOP_K_RERANK    = 5    → final results after cross-encoder
"""

from __future__ import annotations

from loguru import logger

from app.config import settings
from app.ingestion.embedder import embed_query
from app.retrieval.bm25 import SparseVec, encode_sparse_query, get_backend
from app.retrieval.qdrant_client import SearchResult, qdrant_service
from app.retrieval.reranker import rerank


class HybridRetriever:
    """
    Stateless retriever — safe to use as a module-level singleton.
    All parameters can be overridden per-call.
    """

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        min_score: float | None = None,
        use_reranker: bool = True,
        use_sparse:   bool = True,
    ) -> list[SearchResult]:
        """
        Execute the full hybrid retrieval pipeline synchronously.
        (Call from a thread executor in async contexts.)

        Args:
            query:        User's question or search topic.
            tenant_id:    Tenant namespace — results are strictly isolated.
            top_k:        Final number of passages to return (default from settings).
            candidate_k:  Number of candidates fetched before reranking (default 3×top_k).
            min_score:    Minimum Qdrant score threshold to pass (default from settings).
            use_reranker: Apply cross-encoder reranking (recommended).
            use_sparse:   Include BM25 sparse leg of hybrid search (recommended).

        Returns:
            List of SearchResult ordered by relevance (best first).
        """
        _top_k       = top_k      or settings.top_k_rerank
        _candidate_k = candidate_k or settings.top_k_retrieval
        _min_score   = min_score  if min_score is not None else settings.min_relevance_score

        logger.debug(
            "HybridRetriever | tenant='{}' top_k={} candidates={} sparse={} rerank={}",
            tenant_id, _top_k, _candidate_k, use_sparse, use_reranker,
        )

        # ── Step 1: Encode query (dense + optional sparse) ────────────────
        dense_vec: list[float] = embed_query(query)
        sparse_vec: SparseVec | None = None

        if use_sparse and get_backend() != "none":
            try:
                sparse_vec = encode_sparse_query(query)
                if not sparse_vec.indices:      # empty sparse → treat as dense-only
                    sparse_vec = None
            except Exception as exc:
                logger.warning("Sparse query encoding error: {} — degrading to dense.", exc)

        # ── Step 2: Search Qdrant ─────────────────────────────────────────
        try:
            if sparse_vec is not None:
                candidates = qdrant_service.hybrid_search(
                    tenant_id=tenant_id,
                    dense_vector=dense_vec,
                    sparse_vector=sparse_vec,
                    candidate_k=_candidate_k,
                    min_score=0.0,       # RRF handles scoring; pre-filter off
                )
                logger.debug("Hybrid search → {} candidates", len(candidates))
            else:
                candidates = qdrant_service.search(
                    tenant_id=tenant_id,
                    query_vector=dense_vec,
                    top_k=_candidate_k,
                    score_threshold=_min_score,
                )
                logger.debug("Dense-only search → {} candidates", len(candidates))

        except Exception as exc:
            logger.error("Qdrant search failed: {}", exc)
            return []

        if not candidates:
            return []

        # ── Step 3: Cross-encoder rerank ──────────────────────────────────
        if use_reranker and len(candidates) > 1:
            try:
                results = rerank(query, candidates, _top_k)
                logger.info(
                    "✓ HybridRetriever | {} results (top_k={}) | reranked",
                    len(results), _top_k,
                )
                return results
            except Exception as exc:
                logger.warning("Reranker failed: {} — using candidate order.", exc)

        # Fallback: score-order, respect min_score, cap at top_k
        fallback = sorted(
            [c for c in candidates if c.score >= _min_score],
            key=lambda r: r.score,
            reverse=True,
        )[:_top_k]
        logger.info("✓ HybridRetriever | {} results (no reranker)", len(fallback))
        return fallback


# ── Module-level singleton ────────────────────────────────────────────────────

hybrid_retriever = HybridRetriever()
