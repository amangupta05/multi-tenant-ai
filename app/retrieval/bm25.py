"""
BM25 Sparse Encoder — Phase 5
================================
Generates sparse keyword vectors for hybrid retrieval.

Primary backend: fastembed (``pip install fastembed``)
  - Uses Qdrant's "Qdrant/BM25" model: classic BM25 scores with corpus IDF,
    stored as sparse int→float maps.  Very fast, no GPU needed.

Fallback backend: scikit-learn HashingVectorizer
  - Activated automatically when fastembed is not installed.
  - No vocabulary fitting needed — deterministic feature hashing.
  - Slightly lower quality than BM25 but still valuable for hybrid retrieval.

Exported interface
------------------
  SparseVec                    — NamedTuple(indices, values)
  encode_sparse_documents(texts) → list[SparseVec]   (batch, for indexing)
  encode_sparse_query(query)     → SparseVec          (single, for search)
"""

from __future__ import annotations

import threading
from typing import Any, NamedTuple

from loguru import logger


# ── Output type ───────────────────────────────────────────────────────────────

class SparseVec(NamedTuple):
    """Sparse vector represented as parallel lists of (index, value) pairs."""
    indices: list[int]
    values:  list[float]


# ── Singleton sparse model ────────────────────────────────────────────────────

_model: Any = None          # fastembed SparseTextEmbedding | sklearn HashingVectorizer
_backend: str = "none"      # "fastembed" | "sklearn" | "none"
_lock = threading.Lock()

_FASTEMBED_MODEL = "Qdrant/BM25"   # vocabulary-based BM25, very small download


def _init_model() -> None:
    """Lazy-initialise the best available sparse encoder (thread-safe)."""
    global _model, _backend

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        # ── Try fastembed (preferred) ──────────────────────────────────────
        try:
            from fastembed import SparseTextEmbedding  # type: ignore

            logger.info("Loading BM25 sparse model '{}' via fastembed …", _FASTEMBED_MODEL)
            _model = SparseTextEmbedding(model_name=_FASTEMBED_MODEL)
            _backend = "fastembed"
            logger.success("BM25 sparse encoder ready (fastembed/{})", _FASTEMBED_MODEL)
            return
        except ImportError:
            logger.info("fastembed not installed — trying sklearn sparse fallback …")
        except Exception as exc:
            logger.warning("fastembed init failed ({}), falling back to sklearn.", exc)

        # ── Fallback: sklearn HashingVectorizer ────────────────────────────
        try:
            from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

            _model = HashingVectorizer(
                n_features=2**17,       # 131 072 hash bins → low collision rate
                norm=None,              # Keep raw TF counts (like BM25 TF)
                alternate_sign=False,   # Positive values only for Qdrant
                analyzer="word",
                ngram_range=(1, 2),    # Unigrams + bigrams
            )
            _backend = "sklearn"
            logger.success("BM25 sparse encoder ready (sklearn/HashingVectorizer fallback)")
        except ImportError:
            _backend = "none"
            logger.warning(
                "No sparse encoder available (install fastembed or scikit-learn). "
                "Hybrid search will use dense-only."
            )


# ── Encoding functions ────────────────────────────────────────────────────────

def _empty_sparse() -> SparseVec:
    """Return an empty (all-zero) sparse vector as a safe fallback."""
    return SparseVec(indices=[], values=[])


def _fastembed_encode(texts: list[str], *, is_query: bool = False) -> list[SparseVec]:
    embedder_fn = _model.query_embed if is_query else _model.embed
    results: list[SparseVec] = []
    for emb in embedder_fn(texts, batch_size=32):
        results.append(SparseVec(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        ))
    return results


def _sklearn_encode(texts: list[str]) -> list[SparseVec]:
    mat = _model.transform(texts)
    results: list[SparseVec] = []
    for i in range(mat.shape[0]):
        row = mat[i].tocsr()
        results.append(SparseVec(
            indices=row.indices.tolist(),
            values=row.data.tolist(),
        ))
    return results


def encode_sparse_documents(texts: list[str]) -> list[SparseVec]:
    """
    Generate sparse BM25 vectors for a batch of document chunks (for indexing).
    Thread-safe. Degrades gracefully to empty vectors if no backend is available.
    """
    if not texts:
        return []

    _init_model()

    try:
        if _backend == "fastembed":
            return _fastembed_encode(texts, is_query=False)
        elif _backend == "sklearn":
            return _sklearn_encode(texts)
    except Exception as exc:
        logger.warning("Sparse document encoding failed: {} — using empty vectors.", exc)

    return [_empty_sparse() for _ in texts]


def encode_sparse_query(query: str) -> SparseVec:
    """
    Generate a sparse BM25 vector for a single search query.
    Thread-safe. Degrades gracefully if no backend is available.

    fastembed uses query_embed() (optimised query-time IDF weighting).
    sklearn uses the same transform() as documents (acceptable approximation).
    """
    _init_model()

    try:
        if _backend == "fastembed":
            results = _fastembed_encode([query], is_query=True)
            return results[0] if results else _empty_sparse()
        elif _backend == "sklearn":
            results = _sklearn_encode([query])
            return results[0] if results else _empty_sparse()
    except Exception as exc:
        logger.warning("Sparse query encoding failed: {} — returning empty vector.", exc)

    return _empty_sparse()


def get_backend() -> str:
    """Return the name of the active sparse encoder backend."""
    _init_model()
    return _backend
