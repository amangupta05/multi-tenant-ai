"""
Embedding Service — Phase 2
=============================
Local, free embedding using sentence-transformers.

Default model: ``nomic-ai/nomic-embed-text-v1.5``  (768 dims)
  - Best-in-class open-source dense retrieval model
  - Requires ``trust_remote_code=True``
  - Uses task-specific prefixes:
      document  → "search_document: {text}"
      query     → "search_query: {text}"

Fallback model (simpler setup): ``sentence-transformers/all-MiniLM-L6-v2`` (384 dims)
  - No trust_remote_code needed
  - Change EMBEDDING_MODEL + EMBEDDING_DIMENSION in .env

The service is a lazily-initialised singleton so the model is loaded
only once per process, regardless of how many times it is imported.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ── Singleton state ───────────────────────────────────────────────────────────

_model: "SentenceTransformer | None" = None
_model_lock = threading.Lock()          # thread-safe lazy init
_nomic_models = {"nomic-ai/nomic-embed-text-v1.5"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_nomic(model_name: str) -> bool:
    return any(n in model_name for n in _nomic_models)


def _add_prefix(texts: list[str], prefix: str, model_name: str) -> list[str]:
    """Add nomic-style task prefix only when needed (other models ignore it)."""
    if _is_nomic(model_name):
        return [f"{prefix}{t}" for t in texts]
    return texts


# ── Lazy loader ───────────────────────────────────────────────────────────────

def _get_model() -> "SentenceTransformer":
    """Return the cached model, loading it on first call (thread-safe)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:   # double-checked locking
            return _model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = settings.embedding_model
            logger.info(
                "Loading embedding model '{}' on device='{}' …",
                model_name, settings.embedding_device,
            )

            trust_remote = _is_nomic(model_name)
            _model = SentenceTransformer(
                model_name,
                device=settings.embedding_device,
                trust_remote_code=trust_remote,
            )
            logger.success(
                "Embedding model ready — dim={}, device={}",
                settings.embedding_dimension, settings.embedding_device,
            )
            return _model

        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers torch"
            ) from exc


# ── Public embedding functions ────────────────────────────────────────────────

def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of document texts for indexing.

    For nomic models the ``search_document:`` prefix is automatically prepended.
    Returns a list of float vectors (one per input text).
    """
    if not texts:
        return []

    model = _get_model()
    model_name = settings.embedding_model
    prefixed = _add_prefix(texts, "search_document: ", model_name)

    logger.debug("Embedding {} document chunks (batch_size={}) …", len(texts), batch_size)
    vectors: np.ndarray = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,   # cosine similarity needs unit vectors
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query.

    For nomic models the ``search_query:`` prefix is automatically prepended.
    """
    model = _get_model()
    model_name = settings.embedding_model
    prefixed = _add_prefix([query], "search_query: ", model_name)

    vector: np.ndarray = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector[0].tolist()


def get_embedding_dimension() -> int:
    """
    Return the dimension of the model's output vectors.
    Reads the setting directly so this can be called before the model loads.
    """
    return settings.embedding_dimension


# ── EmbeddingService class (OOP alias for DI / testing) ──────────────────────

class EmbeddingService:
    """
    Thin class wrapper around the module-level functions.
    Useful for dependency injection and mocking in tests.
    """

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return embed_documents(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> list[float]:
        return embed_query(query)

    @property
    def dimension(self) -> int:
        return get_embedding_dimension()

    def warm_up(self) -> None:
        """Pre-load the model during startup so the first query is fast."""
        _get_model()
        logger.info("Embedding model warmed up.")
