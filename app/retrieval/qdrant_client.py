"""
Qdrant Vector Store Client — Phase 2
======================================
Local file-based Qdrant — no server, no Docker, no cloud account needed.
Data is persisted to disk at settings.qdrant_path.

Multi-tenancy strategy
-----------------------
Single collection ``knowledge_base`` with payload-based tenant isolation:
  - Every point carries a ``tenant_id`` payload field.
  - All searches include a mandatory ``tenant_id`` filter.
  - A ``is_tenant=True`` payload index creates per-tenant HNSW sub-graphs
    for O(log n) filtering (scales to 100k tenants without separate collections).

Vector layout
--------------
  Named vector "dense" → float vector from nomic-embed-text (768 dims by default).
  Sparse "sparse" (BM25) vectors are reserved for Phase 5 (hybrid retrieval).

Payload schema
--------------
  tenant_id      : str   — tenant isolation key (indexed)
  doc_id         : str   — Document table primary key
  chunk_index    : int   — 0-based position within the document
  text           : str   — raw text of the chunk (returned with results)
  source         : str   — original filename / URL
  section_heading: str   — Docling section heading
  metadata       : dict  — arbitrary extra fields
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger

from app.config import settings


# ── Result type ───────────────────────────────────────────────────────────────

class SearchResult:
    """A single retrieval hit with its text and score."""

    __slots__ = ("id", "text", "score", "tenant_id", "doc_id",
                 "chunk_index", "source", "section_heading", "metadata")

    def __init__(
        self,
        id: str,
        text: str,
        score: float,
        tenant_id: str,
        doc_id: str,
        chunk_index: int,
        source: str,
        section_heading: str,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.text = text
        self.score = score
        self.tenant_id = tenant_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.source = source
        self.section_heading = section_heading
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "section_heading": self.section_heading,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
        }


# ── Qdrant service singleton ──────────────────────────────────────────────────

class QdrantService:
    """
    Lazy-initialised wrapper around the local Qdrant client.

    Call ``await qdrant.init()`` once on startup, then use the instance freely.
    """

    def __init__(self) -> None:
        self._client: Any = None   # qdrant_client.QdrantClient

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def init(self) -> None:
        """
        Open (or create) the local Qdrant database and ensure the collection
        exists with the correct vector configuration.
        Call once during application startup.
        """
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import (  # type: ignore
                Distance,
                HnswConfigDiff,
                PayloadSchemaType,
                SparseVectorParams,
                VectorParams,
            )
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            ) from exc

        path = str(settings.qdrant_local_path)
        logger.info("Opening local Qdrant storage at '{}'", path)
        self._client = QdrantClient(path=path)

        collection_name = settings.qdrant_collection
        existing = {c.name for c in self._client.get_collections().collections}

        if collection_name not in existing:
            logger.info("Creating Qdrant collection '{}' (dense + sparse)", collection_name)
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.success("Collection '{}' ready (dense + sparse).", collection_name)
        else:
            logger.info("Qdrant collection '{}' exists — checking sparse config …", collection_name)
            # Migrate existing collection to add sparse vector field if missing
            try:
                info = self._client.get_collection(collection_name)
                has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
                if not has_sparse:
                    self._client.update_collection(
                        collection_name=collection_name,
                        sparse_vectors_config={"sparse": SparseVectorParams()},
                    )
                    logger.info("  ↳ Added sparse vector config to existing collection.")
            except Exception as exc:
                logger.warning("Sparse vector migration skipped: {}", exc)

    def _require_client(self) -> Any:
        """Return the Qdrant client, initialising it lazily if needed."""
        if self._client is None:
            logger.warning(
                "QdrantService was not pre-initialised — running lazy init now."
            )
            self.init()
        return self._client

    # ── Write operations ───────────────────────────────────────────────────

    def upsert_chunks(
        self,
        *,
        tenant_id: str,
        doc_id: str,
        texts: list[str],
        embeddings: list[list[float]],
        sources: list[str],
        section_headings: list[str],
        chunk_indices: list[int],
        metadata_list: list[dict[str, Any]],
        sparse_embeddings: "list | None" = None,  # list[SparseVec] | None
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunk vectors into Qdrant (dense + optional sparse).

        sparse_embeddings: if provided, each element must expose .indices / .values
            attributes (SparseVec NamedTuple from app.retrieval.bm25).
            Points without sparse vectors are upserted with dense only — fully
            backward-compatible; they will score 0 on the sparse leg of hybrid search.
        """
        from qdrant_client.models import PointStruct, SparseVector  # type: ignore

        client = self._require_client()
        n = len(texts)
        assert len(embeddings) == n == len(sources) == len(section_headings), \
            "All input lists must have the same length"
        has_sparse = (
            sparse_embeddings is not None
            and len(sparse_embeddings) == n
        )

        points: list[PointStruct] = []
        for i in range(n):
            vector_dict: dict[str, Any] = {"dense": embeddings[i]}
            if has_sparse:
                sv = sparse_embeddings[i]  # type: ignore[index]
                if sv.indices:             # skip genuinely empty vectors
                    vector_dict["sparse"] = SparseVector(
                        indices=sv.indices,
                        values=sv.values,
                    )
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_dict,
                payload={
                    "tenant_id":       tenant_id,
                    "doc_id":          doc_id,
                    "chunk_index":     chunk_indices[i],
                    "text":            texts[i],
                    "source":          sources[i],
                    "section_heading": section_headings[i],
                    "metadata":        metadata_list[i],
                    "has_sparse":      has_sparse and bool(sparse_embeddings[i].indices),  # type: ignore[index]
                },
            ))

        total = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=batch,
                wait=True,
            )
            total += len(batch)
            logger.debug(
                "Upserted batch {}/{} (sparse={})",
                min(start + batch_size, n), n, has_sparse,
            )
        return total

    def delete_by_document(self, *, tenant_id: str, doc_id: str) -> int:
        """
        Delete all Qdrant points belonging to a specific document + tenant.
        Returns the number of points deleted.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        client = self._require_client()
        doc_filter = Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="doc_id",    match=MatchValue(value=doc_id)),
            ]
        )

        # Count first so we can return it
        count_result = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=doc_filter,
            exact=True,
        )
        n = count_result.count

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=doc_filter,
            wait=True,
        )
        logger.info("Deleted {} Qdrant points for doc_id='{}' tenant='{}'", n, doc_id, tenant_id)
        return n

    def delete_by_tenant(self, tenant_id: str) -> int:
        """Delete all Qdrant points for a tenant (on tenant deletion)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        client = self._require_client()
        tenant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )
        count_result = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=tenant_filter,
            exact=True,
        )
        n = count_result.count
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=tenant_filter,
            wait=True,
        )
        logger.info("Deleted {} Qdrant points for tenant='{}'", n, tenant_id)
        return n

    # ── Read operations ────────────────────────────────────────────────────

    def search(
        self,
        *,
        tenant_id: str,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Dense vector search scoped to a single tenant.

        Returns up to ``top_k`` results with score ≥ ``score_threshold``.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        client = self._require_client()
        k = top_k or settings.top_k_retrieval
        threshold = score_threshold if score_threshold is not None else 0.0

        tenant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )

        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=("dense", query_vector),
            query_filter=tenant_filter,
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for h in hits:
            p = h.payload or {}
            results.append(SearchResult(
                id=str(h.id),
                text=p.get("text", ""),
                score=h.score,
                tenant_id=p.get("tenant_id", tenant_id),
                doc_id=p.get("doc_id", ""),
                chunk_index=p.get("chunk_index", 0),
                source=p.get("source", ""),
                section_heading=p.get("section_heading", ""),
                metadata=p.get("metadata", {}),
            ))

        return results

    def hybrid_search(
        self,
        *,
        tenant_id: str,
        dense_vector: list[float],
        sparse_vector: "Any",           # SparseVec NamedTuple from bm25.py
        candidate_k: int = 20,
        min_score: float = 0.0,
    ) -> "list[SearchResult]":
        """
        Hybrid dense + sparse search using Qdrant's RRF fusion.

        Qdrant fetches up to ``candidate_k`` results from each vector leg
        (dense HNSW, sparse inverted index), then combines the ranked lists
        with Reciprocal Rank Fusion — no score normalisation needed.

        Args:
            dense_vector:  768-dim float list from the embedding model.
            sparse_vector: SparseVec(indices, values) from the BM25 encoder.
            candidate_k:   Candidates per leg before fusion (should be ≥ top_k × 3).
            min_score:     Post-fusion score threshold (0.0 = keep all).

        Returns:
            SearchResult list ordered by RRF fusion score.
        """
        from qdrant_client.models import (  # type: ignore
            FieldCondition, Filter, Fusion, FusionQuery, MatchValue, Prefetch, SparseVector,
        )

        client = self._require_client()
        tenant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )
        sv = SparseVector(indices=sparse_vector.indices, values=sparse_vector.values)

        hits = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(
                    query=sv,
                    using="sparse",
                    filter=tenant_filter,
                    limit=candidate_k,
                ),
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=tenant_filter,
                    limit=candidate_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=tenant_filter,
            limit=candidate_k,
            with_payload=True,
        ).points

        results: list[SearchResult] = []
        for h in hits:
            if h.score < min_score:
                continue
            p = h.payload or {}
            results.append(SearchResult(
                id=str(h.id),
                text=p.get("text", ""),
                score=h.score,
                tenant_id=p.get("tenant_id", tenant_id),
                doc_id=p.get("doc_id", ""),
                chunk_index=p.get("chunk_index", 0),
                source=p.get("source", ""),
                section_heading=p.get("section_heading", ""),
                metadata=p.get("metadata", {}),
            ))
        return results

    # ── Stats ──────────────────────────────────────────────────────────────

    def count_chunks(self, tenant_id: str) -> int:
        """Return the total number of indexed chunks for a tenant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

        client = self._require_client()
        result = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            ),
            exact=True,
        )
        return result.count

    def collection_info(self) -> dict[str, Any]:
        """Return high-level collection statistics."""
        client = self._require_client()
        info = client.get_collection(settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "points_count": info.points_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", getattr(info, "points_count", 0)),
            "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
            "status": str(info.status),
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

qdrant_service = QdrantService()
