"""
app.mcp_servers — MCP server implementations and client bridge.

Servers
-------
  rag_server   → search_knowledge_base tool
  tools_server → get_order_status, lookup_customer, check_inventory,
                 list_products, get_business_hours tools

Client bridge
-------------
  client.get_all_tools(tenant_id)  →  list of LangChain StructuredTool objects
                                       ready for LangGraph (Phase 4)
"""

from app.mcp_servers.client import get_all_tools, get_tool_names
from app.mcp_servers.rag_server import search_knowledge_base_impl
from app.mcp_servers.tools_server import (
    check_inventory_impl,
    get_business_hours_impl,
    get_order_status_impl,
    list_products_impl,
    lookup_customer_impl,
)

__all__ = [
    # Client bridge
    "get_all_tools",
    "get_tool_names",
    # Direct-call impl functions (for testing / orchestrator fast-path)
    "search_knowledge_base_impl",
    "get_order_status_impl",
    "lookup_customer_impl",
    "check_inventory_impl",
    "list_products_impl",
    "get_business_hours_impl",
]
