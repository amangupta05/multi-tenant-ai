"""
Application settings loaded from environment variables / .env file.
Only GEMINI_API_KEY is required. All other settings have sensible defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────
    app_name: str = "Multi-Tenant AI System"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    # Admin API key to protect /admin/* routes (change this!)
    admin_api_key: str = "admin-secret"

    # ── LLM Configuration ────────────────────────────────────────
    # Core orchestration API key
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    
    # Optional Gemini key for Vision fallbacks
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 2048

    # ── Database ─────────────────────────────────────────────────
    # SQLite for MVP — switch to postgresql+asyncpg://... for production
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # ── Local Storage ─────────────────────────────────────────────
    data_dir: str = "./data"

    # ── Qdrant (local file-based, no server needed) ───────────────
    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "knowledge_base"

    # ── Embeddings (local, free, CPU-friendly) ────────────────────
    # Best quality  : nomic-ai/nomic-embed-text-v1.5   → 768 dims
    # Simpler setup : sentence-transformers/all-MiniLM-L6-v2 → 384 dims
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # ── Reranker (local cross-encoder, free) ──────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Whisper (local speech-to-text, free) ──────────────────────
    # tiny=fastest, base=recommended, small/medium/large=better quality
    whisper_model_size: str = "base"

    # ── Chunking ─────────────────────────────────────────────────
    chunk_size: int = 512       # tokens per chunk
    chunk_overlap: int = 64     # token overlap between consecutive chunks

    # ── Retrieval ─────────────────────────────────────────────────
    top_k_retrieval: int = 20           # candidates fetched from Qdrant
    top_k_rerank: int = 5               # final results after reranking
    min_relevance_score: float = 0.3    # drop chunks below this score

    # ── Session Memory ────────────────────────────────────────────
    session_ttl_seconds: int = 86_400   # 24 h — sessions expire after this
    max_conversation_turns: int = 10    # turns loaded into active LLM context

    # ── Computed paths (not env vars) ─────────────────────────────
    @property
    def data_path(self) -> Path:
        """Root data directory as a Path object."""
        return Path(self.data_dir)

    @property
    def tenants_path(self) -> Path:
        """Per-tenant document storage root."""
        return self.data_path / "tenants"

    @property
    def qdrant_local_path(self) -> Path:
        """Qdrant on-disk storage directory."""
        return Path(self.qdrant_path)

    def tenant_docs_path(self, tenant_id: str) -> Path:
        """Raw uploaded documents for a specific tenant."""
        return self.tenants_path / tenant_id / "documents"

    def ensure_dirs(self) -> None:
        """Create all required local directories on startup."""
        dirs = [
            self.data_path,
            self.tenants_path,
            self.qdrant_local_path,
            self.data_path / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()


# Module-level singleton — import this everywhere
settings: Settings = get_settings()
