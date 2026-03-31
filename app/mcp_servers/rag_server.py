"""
RAG Knowledge Base MCP Server — Phase 3
=========================================
Exposes a single MCP tool:  ``search_knowledge_base``

The server can run in two modes:

  1. **Subprocess / stdio** (production-style MCP):
         python -m app.mcp_servers.rag_server
     LangGraph connects via StdioServerParameters and reads back MCP JSON-RPC.

  2. **In-process** (imported directly by the orchestrator):
         from app.mcp_servers.rag_server import search_knowledge_base
     The function is a plain Python callable — no subprocess overhead.

The `mcp` FastMCP instance (`mcp`) handles both modes transparently.

Search pipeline inside the tool (Phase 5 — hybrid)
---------------------------------------------------
  query  →  embed_query()         →  768-dim dense vector (nomic)
         →  encode_sparse_query() →  BM25 sparse vector (fastembed/sklearn)
         →  Qdrant hybrid_search()→  RRF fusion of dense + sparse candidates
         →  cross-encoder rerank  →  ms-marco MiniLM reranker
         →  format context string with numbered citations

Fallback chain (automatic)
--------------------------
  1. Hybrid (dense + sparse) + cross-encoder  ← default, best quality
  2. Dense-only + cross-encoder               ← if sparse encoding fails
  3. Dense-only, cosine order                 ← if reranker also unavailable
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

from loguru import logger

# ── MCP server instance ───────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
    mcp = FastMCP(
        "RAG Knowledge Base",
        instructions=(
            "Use `search_knowledge_base` to retrieve relevant passages from the "
            "tenant's document store before answering any factual questions.  "
            "Always prefer retrieved context over internal knowledge."
        ),
    )
    _MCP_AVAILABLE = True
except ImportError:
    logger.warning("mcp package not installed — RAG server running in direct-call-only mode.")
    mcp = None  # type: ignore
    _MCP_AVAILABLE = False


# ── Lazy Qdrant / embedding initialisation ────────────────────────────────────

def _ensure_qdrant_ready() -> None:
    """
    Ensure Qdrant is open when the server runs as a standalone subprocess.
    When running inside the FastAPI process it is already initialised by
    the lifespan hook in main.py.
    """
    from app.retrieval.qdrant_client import qdrant_service
    if qdrant_service._client is None:
        logger.info("RAG server: initialising Qdrant (standalone mode) …")
        qdrant_service.init()


# (Cross-encoder reranker is now handled by app.retrieval.reranker — Phase 5)


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_context(query: str, results: list[Any]) -> str:
    """
    Convert a list of SearchResult objects into a structured context string
    that is easy for an LLM to parse and cite.
    """
    if not results:
        return (
            "No relevant information was found in the knowledge base for this query.\n"
            "Please answer based on general knowledge or ask the user for clarification."
        )

    lines: list[str] = [
        f"Found {len(results)} relevant passage(s) for: \"{query}\"\n",
        "=" * 60,
    ]

    for i, r in enumerate(results, start=1):
        source_label = r.source or "unknown source"
        section_label = (
            f" › {r.section_heading}" if r.section_heading else ""
        )
        score_label = f"{r.score:.2f}"

        lines += [
            "",
            f"[{i}] 📄 {source_label}{section_label}  (relevance: {score_label})",
            "─" * 55,
            textwrap.fill(r.text, width=88, subsequent_indent="    "),
        ]

    lines += [
        "",
        "=" * 60,
        "Use the passages above to answer the user's question accurately.",
        "Cite sources as: [Source N] when referencing specific information.",
    ]

    return "\n".join(lines)


# ── Core search logic (pure Python — importable without MCP) ──────────────────

def search_knowledge_base_impl(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    min_score: float = 0.3,
    use_reranker: bool = True,
) -> str:
    """
    Core RAG retrieval — Phase 5 (hybrid dense + BM25 sparse + cross-encoder reranker).
    Called by the MCP tool wrapper AND can be imported directly by the orchestrator.

    Args:
        query:        The user's question or topic to search for.
        tenant_id:    Tenant identifier — results are strictly namespace-scoped.
        top_k:        Maximum number of passages to return (1 – 20).
        min_score:    Minimum passage relevance score (0.0 – 1.0).
        use_reranker: Apply cross-encoder reranking after hybrid retrieval.

    Returns:
        Formatted multi-line string with numbered citations for LLM prompts.
    """
    top_k = max(1, min(top_k, 20))

    try:
        from app.retrieval.hybrid_search import hybrid_retriever  # noqa: PLC0415

        _ensure_qdrant_ready()

        logger.debug(
            "RAG hybrid search | tenant='{}' query='{}'", tenant_id, query[:80]
        )

        results = hybrid_retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            min_score=min_score,
            use_reranker=use_reranker,
        )

        if not results:
            logger.info("Hybrid search → 0 results for tenant='{}'", tenant_id)
        else:
            logger.info(
                "Hybrid search | tenant='{}' → {} results (top score: {:.3f})",
                tenant_id, len(results), results[0].score,
            )

        return _format_context(query, results)

    except Exception as exc:
        logger.error("RAG search failed | tenant='{}': {}", tenant_id, exc)
        return (
            f"Knowledge base search encountered an error: {exc}\n"
            "Please answer the question based on your general knowledge."
        )


# ── MCP tool definition ───────────────────────────────────────────────────────

if _MCP_AVAILABLE:

    @mcp.tool()
    def search_knowledge_base(
        query: str,
        tenant_id: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> str:
        """
        Search the tenant's knowledge base for relevant information.

        Use this tool BEFORE answering any question that might require
        specific business knowledge (products, policies, FAQs, pricing,
        procedures, etc.).

        Args:
            query:     The user's question or search topic. Be specific.
            tenant_id: The tenant's unique identifier (injected by orchestrator).
            top_k:     Number of passages to return (default 5, max 20).
            min_score: Minimum relevance score 0.0–1.0 (default 0.3).

        Returns:
            Numbered passages with source citations.
            Returns a "not found" message if no relevant documents exist.
        """
        return search_knowledge_base_impl(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            min_score=min_score,
        )


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _MCP_AVAILABLE:
        raise SystemExit("Install the 'mcp' package: pip install mcp")
    logger.info("Starting RAG Knowledge Base MCP server (stdio) …")
    mcp.run()
