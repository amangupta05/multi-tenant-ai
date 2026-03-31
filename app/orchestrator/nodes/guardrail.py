"""
Guardrail Node — post-generation safety and quality check.

Checks for:
  1. Hallucination signals (unsupported claims in RAG context)
  2. PII leakage (phone numbers, emails, addresses in response)
  3. Harmful / offensive content
  4. Inappropriate professional boundaries (legal / medical / financial advice)

Actions:
  - PASS  → state unchanged, response flows to memory_save
  - WARN  → safety_flags populated, response passed through
  - BLOCK → response replaced with safe fallback, escalate=True
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.llm import extract_text, get_llm
from app.orchestrator.state import AgentState

# ── Safety prompt ─────────────────────────────────────────────────────────────

_GUARDRAIL_SYSTEM = """You are a safety reviewer for a web AI customer service assistant.

Review the AI response below and check for these issues:
1. *Hallucination* — claims not supported by the provided context or tool results
2. *PII exposure* — phone numbers, emails, addresses, ID numbers in the response
3. *Harmful content* — offensive language, threats, discrimination
4. *Boundary violations* — specific legal, medical, or financial advice

Respond ONLY with valid JSON:
{
  "is_safe": true,
  "severity": "none|low|high",
  "flags": [],
  "action": "pass|warn|block",
  "corrected_response": ""
}

Rules:
- severity "none" → action must be "pass"
- severity "low"  → action "warn" (flag but keep response)
- severity "high" → action "block" (replace with corrected_response)
- corrected_response should be a safe, helpful alternative (empty if passing)"""

_SAFE_FALLBACK = (
    "I'm sorry, I'm not able to provide a complete answer to that right now. "
    "Please contact our support team directly for assistance. "
    "We're here to help! 😊"
)

# ── PII quick-scan (before calling Gemini to save API calls) ──────────────────

_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
]


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


# ── JSON parser ───────────────────────────────────────────────────────────────

def _clean_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", raw)
    text = re.sub(r"```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# ── Node ──────────────────────────────────────────────────────────────────────

async def guardrail_node(state: AgentState) -> dict:
    """
    Safety-check the generated response before it is sent to the user.
    Returns a partial state update: safety_flags, escalate, and possibly
    an overwritten response.
    """
    response = state.get("response", "")
    intent = state.get("intent", "")

    # ── Fast-path: escalate intent already set ────────────────────────────
    if intent == "escalate" or state.get("escalate"):
        logger.info("⚑ guardrail | escalate already set — skipping LLM check")
        escalation_msg = (
            "I completely understand your frustration and I sincerely apologise "
            "for the inconvenience. 🙏\n\n"
            "I'm escalating your case to a senior team member who will reach out "
            "to you within *2 business hours*. You have my assurance that we will "
            "resolve this for you.\n\n"
            "Is there anything else I can note down for the team?"
        )
        return {
            "response": escalation_msg,
            "safety_flags": ["escalate_intent"],
            "escalate": True,
        }

    if not response:
        return {}  # Nothing to check

    # ── PII quick-scan (no LLM call needed) ──────────────────────────────
    flags: list[str] = []
    if _has_pii(response):
        flags.append("possible_pii_in_response")
        logger.warning("⚑ guardrail | PII pattern detected in response")

    # ── LLM safety review (only for non-trivial responses) ───────────────
    if len(response) < 20:
        # Very short responses are trivially safe
        return {"safety_flags": flags, "escalate": False}

    llm = get_llm()
    context_snippet = (state.get("retrieved_context") or "")[:500]
    tool_snippet = str(state.get("tool_results") or "")[:300]

    user_prompt = (
        f"User message: {state['user_message']}\n\n"
        f"Retrieved context (first 500 chars): {context_snippet}\n\n"
        f"Tool results (first 300 chars): {tool_snippet}\n\n"
        f"AI response to review:\n{response}"
    )

    try:
        llm_response = await llm.ainvoke([
            SystemMessage(content=_GUARDRAIL_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        parsed = _clean_json(extract_text(llm_response.content))

        action = parsed.get("action", "pass")
        severity = parsed.get("severity", "none")
        new_flags = parsed.get("flags", [])
        corrected = parsed.get("corrected_response", "")

        all_flags = list(set(flags + new_flags))

        logger.info(
            "⚑ guardrail | action='{}' severity='{}' flags={}",
            action, severity, all_flags,
        )

        if action == "block":
            return {
                "response": corrected or _SAFE_FALLBACK,
                "safety_flags": all_flags,
                "escalate": True,
            }
        elif action == "warn":
            # Log and continue — response goes through with flags noted
            return {"safety_flags": all_flags, "escalate": False}
        else:
            return {"safety_flags": all_flags, "escalate": False}

    except Exception as exc:
        # Guardrail failure must NOT block the response
        logger.warning("Guardrail LLM check failed (non-fatal): {}", exc)
        return {"safety_flags": flags, "escalate": False}
