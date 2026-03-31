"""
Tools Executor Node — lets Gemini select and call MCP tools.

Flow
----
  1. Build tenant-scoped StructuredTools from MCP client (Strategy A, in-process).
  2. Bind tools to Gemini → llm_with_tools.
  3. Invoke with conversation history + user message.
  4. If Gemini makes tool calls → execute each tool → collect results.
  5. If Gemini answers directly (no tool calls) → treat as pre-answer.
  6. Populate state['tool_results'] for the generate node to format.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from app.core.llm import extract_text, get_llm
from app.core.memory import format_history_for_prompt
from app.orchestrator.state import AgentState

# ── Prompts ───────────────────────────────────────────────────────────────────

_TOOL_SYSTEM = """You are a precise tool-calling assistant for a business.
Given the user's request, call the MOST RELEVANT tool to get the information needed.
Call at most 2 tools per turn. Be concise — the results will be shown to the customer.

Available context:
{history}"""


# ── Node ──────────────────────────────────────────────────────────────────────

async def tools_node(state: AgentState) -> dict:
    """
    Execute one or more MCP tools selected by Gemini.
    Updates state with tool_results; generate_node formats the final response.
    """
    tenant_id = state["tenant_id"]
    user_message = state["user_message"]
    history = state.get("conversation_history", [])

    logger.info("⚙ tools_node | tenant='{}' msg='{}'", tenant_id, user_message[:80])

    # ── Step 1: Get tools (Strategy A — direct, in-process) ───────────────
    try:
        from app.mcp_servers.client import get_all_tools  # noqa: PLC0415
        tools = await get_all_tools(tenant_id=tenant_id, use_subprocess=False)
    except Exception as exc:
        logger.error("Failed to load MCP tools: {}", exc)
        return {
            "tool_results": [],
            "error": f"Tool system unavailable: {exc}",
        }

    if not tools:
        logger.warning("No tools available for tenant='{}'", tenant_id)
        return {"tool_results": []}

    # ── Step 2: Bind tools to LLM ─────────────────────────────────────────
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    history_str = format_history_for_prompt(history, max_turns=4)
    system_msg = SystemMessage(content=_TOOL_SYSTEM.format(history=history_str))
    human_msg = HumanMessage(content=user_message)

    # ── Step 3: First LLM call — tool selection ───────────────────────────
    try:
        response: AIMessage = await llm_with_tools.ainvoke([system_msg, human_msg])
    except Exception as exc:
        logger.error("Tool-selecting LLM call failed: {}", exc)
        return {"tool_results": [], "error": str(exc)}

    # ── Step 4: Execute selected tools ────────────────────────────────────
    tool_results: list[dict[str, Any]] = []

    if not response.tool_calls:
        # Gemini answered directly (no tool needed) — treat as pre-answer
        logger.debug("No tool calls made — Gemini answered directly")
        text_content = extract_text(response.content)
        if text_content:
            tool_results.append({
                "tool": "direct_answer",
                "args": {},
                "result": text_content,
            })
        return {"tool_results": tool_results}

    # Build a tool name → callable map for efficient lookup
    tool_map = {t.name: t for t in tools}

    for call in response.tool_calls[:2]:  # cap at 2 tool calls per turn
        tool_name: str = call["name"]
        tool_args: dict = call.get("args", {})

        logger.info("  → calling tool='{}' args={}", tool_name, tool_args)

        try:
            tool_obj = tool_map.get(tool_name)
            if tool_obj is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            # Tools are sync — run in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=tool_obj, a=tool_args: t.invoke(a)
            )
            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result,
            })
            logger.debug("  ✓ tool='{}' result_len={}", tool_name, len(str(result)))

        except Exception as exc:
            logger.error("Tool '{}' execution error: {}", tool_name, exc)
            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": f"Error executing {tool_name}: {exc}",
                "error": True,
            })

    return {"tool_results": tool_results}
