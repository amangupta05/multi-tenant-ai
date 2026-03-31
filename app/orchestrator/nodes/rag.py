"""
RAG Node — retrieves relevant context from the knowledge base.
Calls the MCP RAG server's impl function directly (Strategy A, in-process).
"""

from __future__ import annotations

from loguru import logger

from app.orchestrator.state import AgentState


async def rag_node(state: AgentState) -> dict:
    """
    Execute knowledge base retrieval for the user's query.
    Populates state['retrieved_context'] with formatted citations.
    """
    import asyncio

    from app.mcp_servers.rag_server import search_knowledge_base_impl  # noqa: PLC0415
    from app.config import settings  # noqa: PLC0415

    query = state["user_message"]
    tenant_id = state["tenant_id"]

    logger.info("◎ rag_node | tenant='{}' query='{}'", tenant_id, query[:80])

    try:
        # search_knowledge_base_impl is CPU-bound (embedding + Qdrant I/O)
        # run it in a thread so the event loop stays free
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            None,
            lambda: search_knowledge_base_impl(
                query=query,
                tenant_id=tenant_id,
                top_k=settings.top_k_rerank,
                min_score=settings.min_relevance_score,
                use_reranker=True,
            ),
        )
        logger.debug("RAG context length: {} chars", len(context))
        return {"retrieved_context": context}

    except Exception as exc:
        logger.error("RAG node error: {}", exc)
        return {
            "retrieved_context": (
                "Knowledge base search is temporarily unavailable. "
                "Please answer based on general knowledge or ask the user to try again."
            ),
            "error": str(exc),
        }
