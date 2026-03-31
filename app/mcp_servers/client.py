"""
MCP Client Manager — Phase 3
==============================
Provides LangChain-compatible tools assembled from both MCP servers.
Used by the Phase 4 LangGraph orchestrator.

Two connection strategies
--------------------------
Strategy A — **Direct (default, MVP)**
  Imports the Python impl functions directly and wraps them as LangChain
  StructuredTool objects.  Zero subprocess overhead.  Ideal for
  development and single-process deployment.

Strategy B — **MCP stdio subprocess**
  Uses ``langchain-mcp-adapters`` MultiServerMCPClient to spawn each
  server as a subprocess connected via stdio JSON-RPC.  Matches the
  production multi-process deployment model.

Switch between strategies by calling:
  tools = await get_all_tools(use_subprocess=False)   # Strategy A (default)
  tools = await get_all_tools(use_subprocess=True)    # Strategy B

In Phase 4's orchestrator, Strategy A is used.
In a production deployment, promote to Strategy B by flipping the flag.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


# ── Strategy A — direct LangChain tool wrappers ───────────────────────────────

def _build_direct_tools(tenant_id: str) -> list[Any]:
    """
    Wrap each server's impl functions as LangChain StructuredTool objects
    with the tenant_id pre-bound so the LLM only needs to supply the
    query/order_id/etc. parameters.

    Returns a flat list of BaseTool-compatible objects.
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required. Run: pip install langchain-core"
        ) from exc

    # ── Import impl functions ──────────────────────────────────────────────
    from app.mcp_servers.rag_server import search_knowledge_base_impl
    from app.mcp_servers.tools_server import (
        check_inventory_impl,
        get_business_hours_impl,
        get_order_status_impl,
        list_products_impl,
        lookup_customer_impl,
    )

    # ── Pydantic input schemas (for LangChain structured tools) ───────────

    class SearchKBInput(BaseModel):
        query: str = Field(..., description="The search query or question to look up in the knowledge base.")
        top_k: int = Field(5, description="Number of passages to return (1–20).")
        min_score: float = Field(0.3, description="Minimum relevance score (0.0–1.0).")

    class OrderStatusInput(BaseModel):
        order_id: str = Field(..., description="Order ID (e.g. ORD1234, #1234, or just 1234).")

    class LookupCustomerInput(BaseModel):
        identifier: str = Field(..., description="Customer phone number, email address, or full name.")

    class CheckInventoryInput(BaseModel):
        product_name: str = Field(..., description="Full or partial product name to check stock for.")

    class ListProductsInput(BaseModel):
        category: str | None = Field(None, description="Optional category filter: electronics | accessories | wearables.")

    class BusinessHoursInput(BaseModel):
        pass  # No user inputs besides tenant_id (which is pre-bound)

    # ── Build tools with tenant_id pre-bound ──────────────────────────────

    def _search_kb(query: str, top_k: int = 5, min_score: float = 0.3) -> str:
        return search_knowledge_base_impl(query=query, tenant_id=tenant_id, top_k=top_k, min_score=min_score)

    def _get_order(order_id: str) -> str:
        return get_order_status_impl(order_id=order_id, tenant_id=tenant_id)

    def _lookup_customer(identifier: str) -> str:
        return lookup_customer_impl(identifier=identifier, tenant_id=tenant_id)

    def _check_inventory(product_name: str) -> str:
        return check_inventory_impl(product_name=product_name, tenant_id=tenant_id)

    def _list_products(category: str | None = None) -> str:
        return list_products_impl(tenant_id=tenant_id, category=category)

    def _business_hours() -> str:
        return get_business_hours_impl(tenant_id=tenant_id)

    tools = [
        StructuredTool.from_function(
            _search_kb,
            name="search_knowledge_base",
            description=(
                "Search the tenant's document knowledge base for relevant information. "
                "Use this FIRST before answering questions about products, policies, FAQs, "
                "pricing, or any business-specific information."
            ),
            args_schema=SearchKBInput,
        ),
        StructuredTool.from_function(
            _get_order,
            name="get_order_status",
            description=(
                "Retrieve the current status, delivery estimate, and tracking information "
                "for a customer order.  Use when the user asks about their order."
            ),
            args_schema=OrderStatusInput,
        ),
        StructuredTool.from_function(
            _lookup_customer,
            name="lookup_customer",
            description=(
                "Find a customer's profile by phone number, email, or name.  "
                "Returns tier, total orders, and spend history."
            ),
            args_schema=LookupCustomerInput,
        ),
        StructuredTool.from_function(
            _check_inventory,
            name="check_inventory",
            description=(
                "Check the current stock level and availability of a product by name.  "
                "Use when the user asks whether something is in stock or how many are left."
            ),
            args_schema=CheckInventoryInput,
        ),
        StructuredTool.from_function(
            _list_products,
            name="list_products",
            description=(
                "Browse the full product catalogue, optionally filtered by category "
                "(electronics, accessories, wearables).  Use when the user asks what "
                "products are available."
            ),
            args_schema=ListProductsInput,
        ),
        StructuredTool.from_function(
            _business_hours,
            name="get_business_hours",
            description=(
                "Return the business operating hours and support availability.  "
                "Use when the user asks about opening times or when they can get help."
            ),
            args_schema=BusinessHoursInput,
        ),
    ]

    logger.debug("Built {} direct-call tools for tenant='{}'", len(tools), tenant_id)
    return tools


# ── Strategy B — MCP stdio subprocess ────────────────────────────────────────

async def _build_mcp_subprocess_tools() -> list[Any]:
    """
    Connect to both MCP servers as subprocesses via stdio.
    Returns a combined list of LangChain tools loaded from the MCP protocol.

    Requires:  pip install langchain-mcp-adapters mcp
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "langchain-mcp-adapters is required for subprocess MCP mode. "
            "Run: pip install langchain-mcp-adapters"
        ) from exc

    python_exe = sys.executable
    base = Path(__file__).parent.parent.parent  # repo root
    # Resolve absolute paths so subprocess can find the modules
    rag_path    = str(base / "app" / "mcp_servers" / "rag_server.py")
    tools_path  = str(base / "app" / "mcp_servers" / "tools_server.py")

    server_config = {
        "rag": {
            "command": python_exe,
            "args": [rag_path],
            "transport": "stdio",
        },
        "business_tools": {
            "command": python_exe,
            "args": [tools_path],
            "transport": "stdio",
        },
    }

    logger.info("Connecting to MCP servers via stdio subprocess …")
    async with MultiServerMCPClient(server_config) as client:
        tools = client.get_tools()
        logger.success("Loaded {} MCP tools via subprocess", len(tools))
        return tools


# ── Public interface ──────────────────────────────────────────────────────────

async def get_all_tools(
    tenant_id: str,
    use_subprocess: bool = False,
) -> list[Any]:
    """
    Return a list of LangChain-compatible tools for use in LangGraph.

    Args:
        tenant_id:      Tenant ID to pre-bind into every tool (for isolation).
        use_subprocess: If True, uses MCP stdio transport (production-style).
                        If False (default), calls functions directly in-process.

    Returns:
        List of ``StructuredTool`` / ``BaseTool`` objects ready for LangGraph.
    """
    if use_subprocess:
        return await _build_mcp_subprocess_tools()
    return _build_direct_tools(tenant_id)


def get_tool_names() -> list[str]:
    """Return the canonical tool names (used by the supervisor to route intent)."""
    return [
        "search_knowledge_base",
        "get_order_status",
        "lookup_customer",
        "check_inventory",
        "list_products",
        "get_business_hours",
    ]
