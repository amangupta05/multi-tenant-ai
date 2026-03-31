"""
Text Chunker — Phase 2
=======================
Splits processed document text into overlapping chunks suitable for embedding.

Strategy (in priority order):
  1. If Docling produced sections → chunk each section independently
     (preserves semantic heading context).
  2. Use LangChain's RecursiveCharacterTextSplitter as the universal fallback
     (handles Markdown, code, prose, tables gracefully).

Every chunk carries rich metadata so retrieval results are fully traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.config import settings
from app.ingestion.processor import ProcessedDocument


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single indexable unit of text, ready for embedding and upsert."""

    # The text that will be embedded and stored
    text: str

    # --- Provenance ---
    doc_id: str       # Document DB primary key
    chunk_index: int  # 0-based position within the document
    tenant_id: str

    # --- Source context (shown to user as citation) ---
    source: str = ""            # filename or URL
    section_heading: str = ""   # heading under which this chunk falls

    # --- Extra metadata stored in Qdrant payload ---
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Chunker ───────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Converts a ``ProcessedDocument`` into a list of ``TextChunk`` objects.

    Parameters mirror the env-var settings so they can be tuned per-tenant
    if needed in a future version.
    """

    # Approx chars per token (used to convert token limits → char limits)
    _CHARS_PER_TOKEN: int = 4

    # Markdown-aware separators tried in order by RecursiveCharacterTextSplitter
    _MD_SEPARATORS: list[str] = [
        "\n## ", "\n### ", "\n#### ",   # Markdown headings
        "\n\n",                          # Paragraph breaks
        "\n",                            # Line breaks
        ". ",                            # Sentence breaks
        " ",                             # Word breaks
        "",                              # Character fallback
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size_tokens = chunk_size or settings.chunk_size
        self._chunk_overlap_tokens = chunk_overlap or settings.chunk_overlap
        # Convert to chars for the text splitter
        self._chunk_size_chars = self._chunk_size_tokens * self._CHARS_PER_TOKEN
        self._chunk_overlap_chars = self._chunk_overlap_tokens * self._CHARS_PER_TOKEN

    def chunk(
        self,
        doc: ProcessedDocument,
        doc_id: str,
        tenant_id: str,
    ) -> list[TextChunk]:
        """
        Main entry point.
        Returns an ordered list of ``TextChunk`` objects for the given document.
        """
        source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")

        if doc.sections:
            chunks = self._chunk_by_sections(doc, doc_id, tenant_id, source)
        else:
            chunks = self._chunk_flat(doc.text, doc_id, tenant_id, source, section_heading="")

        logger.debug(
            "Chunked '{}' ({} sections) → {} chunks",
            source, len(doc.sections), len(chunks),
        )
        return chunks

    # ── Section-aware chunking ─────────────────────────────────────────────

    def _chunk_by_sections(
        self,
        doc: ProcessedDocument,
        doc_id: str,
        tenant_id: str,
        source: str,
    ) -> list[TextChunk]:
        """Chunk each Docling section separately so heading context is preserved."""
        all_chunks: list[TextChunk] = []
        global_index = 0

        for section in doc.sections:
            # Prepend heading to give the chunk semantic context
            section_text = f"## {section.heading}\n\n{section.content}"
            section_chunks = self._split_text(section_text)

            for raw_text in section_chunks:
                if not raw_text.strip():
                    continue
                all_chunks.append(TextChunk(
                    text=raw_text.strip(),
                    doc_id=doc_id,
                    chunk_index=global_index,
                    tenant_id=tenant_id,
                    source=source,
                    section_heading=section.heading,
                    metadata={
                        "section_level": section.level,
                        **doc.metadata,
                    },
                ))
                global_index += 1

        # If sections produced nothing, fall back to flat chunking
        if not all_chunks:
            return self._chunk_flat(doc.text, doc_id, tenant_id, source)

        return all_chunks

    def _chunk_flat(
        self,
        text: str,
        doc_id: str,
        tenant_id: str,
        source: str,
        section_heading: str = "",
    ) -> list[TextChunk]:
        """Split a single block of text without section metadata."""
        chunks: list[TextChunk] = []
        for idx, raw_text in enumerate(self._split_text(text)):
            if not raw_text.strip():
                continue
            chunks.append(TextChunk(
                text=raw_text.strip(),
                doc_id=doc_id,
                chunk_index=idx,
                tenant_id=tenant_id,
                source=source,
                section_heading=section_heading,
            ))
        return chunks

    # ── Core splitter ──────────────────────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        """
        Use LangChain's RecursiveCharacterTextSplitter which handles
        Markdown structure, tables, and code blocks correctly.
        Falls back to a naive split if LangChain is not available.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

            splitter = RecursiveCharacterTextSplitter(
                separators=self._MD_SEPARATORS,
                chunk_size=self._chunk_size_chars,
                chunk_overlap=self._chunk_overlap_chars,
                length_function=len,
                is_separator_regex=False,
                keep_separator=True,
            )
            return splitter.split_text(text)

        except ImportError:
            # Minimal fallback — no overlap, split on double newlines
            logger.warning("langchain_text_splitters not installed — using naive splitter.")
            return self._naive_split(text)

    def _naive_split(self, text: str) -> list[str]:
        """
        Simple fallback splitter when LangChain is not available.
        Splits on paragraph boundaries, then merges short paragraphs.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self._chunk_size_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # If a single paragraph exceeds the limit, hard-cut it
                while len(para) > self._chunk_size_chars:
                    chunks.append(para[: self._chunk_size_chars])
                    para = para[self._chunk_size_chars - self._chunk_overlap_chars :]
                current = para

        if current:
            chunks.append(current)

        return chunks
