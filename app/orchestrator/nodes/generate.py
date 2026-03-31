"""
Generate Node — produces the final Chat response with the core LLM.

Handles all intent types:
  rag       → uses retrieved_context to answer the question
  tool_call → formats tool_results into a customer-facing answer
  chitchat  → friendly conversational reply
  escalate  → empathetic escalation message + human handoff notice

Formatting rules applied:
  - Base markdown logic
  - Max ~300 words (general chat UX best practice)
  - No markdown headers (#, ##)
  - Paragraph breaks with blank lines
  - Numbered lists use 1. 2. 3. format
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.llm import extract_text, get_llm
from app.core.memory import format_history_for_prompt
from app.orchestrator.state import AgentState

# ── System prompt template ────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """You are a helpful AI assistant for *{tenant_name}*.
You assist customers via a web Chat interface in a friendly, professional, and empathetic manner.

{custom_instructions}

*FORMATTING RULES:*
- Use rich Markdown where appropriate (bolding, italics, links)
- Use markdown headers (#, ##, ###) if structuring a long response
- Use numbered lists (1. 2. 3.) or bullet points (- ) where helpful
- If you don't know something, say so honestly — never make up information
- End with a helpful follow-up question when appropriate
- Use a warm, friendly tone"""

_RAG_CONTEXT_BLOCK = """
*KNOWLEDGE BASE CONTEXT:*
{context}

Answer the customer's question using ONLY the information above.
If the answer is not in the context, say: "I don't have that specific information in my knowledge base. Let me connect you with our team who can help!"
Cite the source naturally (e.g., "According to our policy...").
"""

_TOOL_RESULT_BLOCK = """
*REAL-TIME DATA RETRIEVED:*
{results}

Present this information to the customer clearly and helpfully.
"""

_ESCALATE_BLOCK = """
The customer needs to be connected to a human agent. Be empathetic and reassuring.
Let them know a team member will reach out shortly. DO NOT attempt to resolve the issue further.
"""

_HISTORY_BLOCK = """
*CONVERSATION SO FAR:*
{history}
"""


# ── Formatters ────────────────────────────────────────────────────────────────

def _format_tool_results(tool_results: list[dict[str, Any]]) -> str:
    """Convert tool results list into a readable block for the LLM."""
    if not tool_results:
        return "(No tool results available)"

    parts: list[str] = []
    for r in tool_results:
        tool_name = r.get("tool", "unknown")
        result = r.get("result", "")

        if tool_name == "direct_answer":
            parts.append(str(result))
            continue

        # Try to pretty-print JSON results
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            pretty = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pretty = str(result)

        parts.append(f"[{tool_name}]\n{pretty}")

    return "\n\n".join(parts)


# ── Node ──────────────────────────────────────────────────────────────────────

async def generate_node(state: AgentState) -> dict:
    """
    Generate the final Chat response using the core LLM.
    Adapts the prompt based on intent (RAG / tool / chitchat / escalate).
    """
    llm = get_llm()
    intent = state.get("intent", "chitchat")
    tenant_name = state.get("tenant_name", "our company")
    custom_instructions = state.get("tenant_config", {}).get("custom_system_prompt", "")
    history = state.get("conversation_history", [])

    logger.info("✎ generate | tenant='{}' intent='{}'", state["tenant_id"], intent)

    # ── Build system prompt ────────────────────────────────────────────────
    system_content = _SYSTEM_TEMPLATE.format(
        tenant_name=tenant_name,
        custom_instructions=custom_instructions or "",
    )

    # Append intent-specific context block
    if intent == "escalate":
        system_content += _ESCALATE_BLOCK
    elif intent == "rag" and state.get("retrieved_context"):
        system_content += _RAG_CONTEXT_BLOCK.format(
            context=state["retrieved_context"]
        )
    elif intent == "tool_call" and state.get("tool_results"):
        tool_str = _format_tool_results(state["tool_results"])
        system_content += _TOOL_RESULT_BLOCK.format(results=tool_str)

    # Append conversation history
    if history:
        history_str = format_history_for_prompt(history, max_turns=5)
        system_content += _HISTORY_BLOCK.format(history=history_str)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["user_message"]),
    ]

    # ── Call Gemini ────────────────────────────────────────────────────────
    try:
        response = await llm.ainvoke(messages)
        text = extract_text(response.content)

        if not text:
            text = (
                "I'm sorry, I wasn't able to generate a response right now. "
                "Please try again or contact our support team."
            )

        logger.info(
            "✓ generate | intent='{}' response_len={}", intent, len(text)
        )
        return {"response": text}

    except Exception as exc:
        logger.error("Generate node error: {}", exc)
        fallback = (
            "I apologise, but I'm experiencing a technical issue right now. "
            "Please try again in a moment or contact us directly."
        )
        return {"response": fallback, "error": str(exc)}
