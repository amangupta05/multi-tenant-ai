"""
LangGraph StateGraph — assembles all nodes into the AI pipeline.

Flow diagram
------------

  START
    │
    ▼
  preprocess          (sanitise input)
    │
    ▼
  memory_load         (load last N turns from TTLCache)
    │
    ▼
  supervisor          (Gemini classifies intent)
    │
    ├─── intent="rag"       ──► rag_node ─────────────┐
    ├─── intent="tool_call" ──► tools_node ────────────┤
    ├─── intent="chitchat"  ──────────────────────────►│
    └─── intent="escalate"  ──────────────────────────►│
                                                        ▼
                                                    generate_node   (Gemini → response)
                                                        │
                                                        ▼
                                                    guardrail_node  (safety check)
                                                        │
                                                        ▼
                                                    memory_save     (persist turn)
                                                        │
                                                       END
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from app.orchestrator.nodes.generate import generate_node
from app.orchestrator.nodes.guardrail import guardrail_node
from app.orchestrator.nodes.memory_nodes import memory_load_node, memory_save_node
from app.orchestrator.nodes.preprocess import preprocess_node
from app.orchestrator.nodes.rag import rag_node
from app.orchestrator.nodes.supervisor import supervisor_node
from app.orchestrator.nodes.tools import tools_node
from app.orchestrator.state import AgentState


# ── Routing function ──────────────────────────────────────────────────────────

def _route_by_intent(state: AgentState) -> str:
    """
    Conditional edge after the supervisor node.
    Maps intent string → next node name.
    """
    intent = state.get("intent", "chitchat")

    if intent == "rag":
        return "rag"
    if intent == "tool_call":
        return "tools"
    # chitchat, escalate, unknown → generate directly
    return "generate"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Build and compile the LangGraph StateGraph."""
    g = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────
    g.add_node("preprocess",   preprocess_node)
    g.add_node("memory_load",  memory_load_node)
    g.add_node("supervisor",   supervisor_node)
    g.add_node("rag",          rag_node)
    g.add_node("tools",        tools_node)
    g.add_node("generate",     generate_node)
    g.add_node("guardrail",    guardrail_node)
    g.add_node("memory_save",  memory_save_node)

    # ── Set entry point ─────────────────────────────────────────────────
    g.set_entry_point("preprocess")

    # ── Linear edges ────────────────────────────────────────────────────
    g.add_edge("preprocess",  "memory_load")
    g.add_edge("memory_load", "supervisor")

    # ── Conditional routing after supervisor ────────────────────────────
    g.add_conditional_edges(
        "supervisor",
        _route_by_intent,
        {
            "rag":      "rag",
            "tools":    "tools",
            "generate": "generate",
        },
    )

    # ── Both retrieval paths converge at generate ────────────────────────
    g.add_edge("rag",   "generate")
    g.add_edge("tools", "generate")

    # ── Linear tail ─────────────────────────────────────────────────────
    g.add_edge("generate",    "guardrail")
    g.add_edge("guardrail",   "memory_save")
    g.add_edge("memory_save", END)

    compiled = g.compile()
    logger.success("LangGraph compiled successfully (nodes={})", 8)
    return compiled


# ── Process-wide singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_compiled_graph() -> Any:
    """Return the cached compiled graph (built once per process)."""
    return build_graph()
