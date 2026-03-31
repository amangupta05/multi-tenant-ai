"""
app.retrieval — Vector store and retrieval components.

Phase 2: Dense search
  qdrant_client  → QdrantService, SearchResult, qdrant_service

Phase 5: Hybrid search + reranking
  bm25           → SparseVec, encode_sparse_documents, encode_sparse_query
  reranker       → rerank() cross-encoder reranker
  hybrid_search  → HybridRetriever, hybrid_retriever (singleton)
"""

from app.retrieval.bm25 import SparseVec, encode_sparse_documents, encode_sparse_query
from app.retrieval.hybrid_search import HybridRetriever, hybrid_retriever
from app.retrieval.qdrant_client import SearchResult, qdrant_service
from app.retrieval.reranker import rerank

__all__ = [
    # Core
    "qdrant_service",
    "SearchResult",
    # Sparse
    "SparseVec",
    "encode_sparse_documents",
    "encode_sparse_query",
    # Retrieval
    "hybrid_retriever",
    "HybridRetriever",
    "rerank",
]
