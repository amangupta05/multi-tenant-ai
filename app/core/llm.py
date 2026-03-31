"""
Shared LLM singleton — Groq Llama-3 via langchain-groq.
Instantiated once per process and reused across all nodes.
Free tier: https://console.groq.com/
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from langchain_groq import ChatGroq


@lru_cache(maxsize=1)
def get_llm() -> "ChatGroq":
    """Return the cached Groq LLM instance (thread-safe via lru_cache)."""
    try:
        from langchain_groq import ChatGroq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "langchain-groq is not installed. "
            "Run: pip install langchain-groq"
        ) from exc

    logger.debug("Initialising Groq LLM model='{}'", settings.groq_model)
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
    )


def extract_text(content: str | list | dict | None) -> str:
    """Safely extract plain text from Langchain's varying content structures."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return " ".join(parts).strip()
    return str(content)
