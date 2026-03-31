"""
Async SQLAlchemy engine + session factory for SQLite.
Swap DATABASE_URL in .env to use PostgreSQL in production — zero code change required.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ── Declarative base shared by all ORM models ─────────────────────────────────

class Base(DeclarativeBase):
    """SQLAlchemy ORM declarative base."""

    # Provide a default __repr__ to make debugging easier
    def __repr__(self) -> str:  # pragma: no cover
        cols = {c.name: getattr(self, c.name, None) for c in self.__table__.columns}  # type: ignore[attr-defined]
        pairs = ", ".join(f"{k}={v!r}" for k, v in list(cols.items())[:4])
        return f"<{self.__class__.__name__} {pairs}>"


# ── Engine + session factory (initialised lazily on startup) ──────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised. Call init_db() first.")
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Session factory not initialised. Call init_db() first.")
    return _session_factory


async def init_db() -> None:
    """
    Initialise the async engine, enable WAL mode for SQLite,
    and create all tables defined in the ORM models.

    Call once on application startup.
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,          # Log SQL only in debug mode
        future=True,
        connect_args={"check_same_thread": False},  # Required for SQLite
    )

    # Enable SQLite Write-Ahead Logging for better concurrency
    if "sqlite" in settings.database_url:

        @event.listens_for(_engine.sync_engine, "connect")  # type: ignore[misc]
        def set_sqlite_pragma(dbapi_conn: Any, _: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    _session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,   # Avoid lazy-load errors after commit in async
        class_=AsyncSession,
    )

    # Import models here so their metadata is registered before create_all
    from app.db import models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine on application shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields a managed async DB session.

    Usage::

        @app.get("/items")
        async def read_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_connection() -> AsyncGenerator[AsyncConnection, None]:
    """Yield a raw async connection — useful for DDL/migrations."""
    engine = _get_engine()
    async with engine.connect() as conn:
        yield conn


async def ping_db() -> bool:
    """Health-check — returns True if the database is reachable."""
    try:
        engine = _get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
