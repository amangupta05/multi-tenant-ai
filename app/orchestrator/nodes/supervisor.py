"""
Supervisor Node — classifies user intent with Gemini.

Intent classes
--------------
  rag        → question requiring business knowledge from documents
  tool_call  → action/lookup (orders, customers, inventory, hours)
  chitchat   → greeting, small talk, general knowledge
  escalate   → complaint, request for human, sensitive topic

Returns a partial state dict with: intent, intent_confidence, intent_reasoning.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.llm import extract_text, get_llm
from app.core.memory import format_history_for_prompt
from app.orchestrator.state import AgentState

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an intent classifier for a general business AI assistant.

Classify the user's message into EXACTLY ONE of these intents:

• "rag"       — A question that requires specific business knowledge (products,
                policies, FAQs, procedures, pricing, company information).
                Examples: "What is your return policy?", "Tell me about Plan X"

• "tool_call" — A request requiring real-time data lookup or action.
                Examples: "What's the status of order 1234?",
                          "Is the iPhone 15 Pro in stock?",
                          "Look up customer +919876543210",
                          "What are your business hours?"

• "chitchat"  — General conversation, greetings, small talk, or simple
                questions answerable without business-specific knowledge.
                Examples: "Hello!", "Thanks", "What is 2+2?", "How are you?"

• "escalate"  — User is upset, requesting a human agent, making threats,
                or the topic is sensitive (legal, medical, emergency).
                Examples: "I want to speak to a manager!",
                          "This is completely unacceptable",
                          "I'm going to sue you"

Respond ONLY with valid JSON — no explanation, no markdown:
{
  "intent": "rag|tool_call|chitchat|escalate",
  "confidence": 0.0,
  "reasoning": "one concise sentence"
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", raw)
    text = re.sub(r"```", "", text).strip()
    # Handle cases where the LLM wraps the JSON in extra text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


_VALID_INTENTS = {"rag", "tool_call", "chitchat", "escalate"}


# ── Node ──────────────────────────────────────────────────────────────────────

async def supervisor_node(state: AgentState) -> dict:
    """
    Classifies user intent to route the message to the right processing node.
    Falls back to 'rag' on parse errors to avoid dead-ends.
    """
    llm = get_llm()
    history_str = format_history_for_prompt(state.get("conversation_history", []))

    user_prompt = (
        f"Conversation so far:\n{history_str}\n\n"
        f"New user message: {state['user_message']}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        parsed = _clean_json(extract_text(response.content))

        intent = parsed.get("intent", "chitchat")
        if intent not in _VALID_INTENTS:
            intent = "rag"  # safe fallback

        confidence = float(parsed.get("confidence", 0.8))
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "✦ supervisor | intent='{}' conf={:.2f} | {}",
            intent, confidence, reasoning[:80],
        )

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
        }

    except Exception as exc:
        logger.warning("Supervisor parse error: {} — defaulting to 'rag'", exc)
        return {
            "intent": "rag",
            "intent_confidence": 0.5,
            "intent_reasoning": f"Parse error: {exc}",
        }
