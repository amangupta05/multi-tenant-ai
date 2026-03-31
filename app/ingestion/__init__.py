"""app.ingestion — document processing, chunking, and embedding."""

from app.ingestion.chunker import TextChunk, TextChunker
from app.ingestion.embedder import EmbeddingService, embed_documents, embed_query
from app.ingestion.processor import DocumentProcessor, ProcessedDocument, get_doc_type

__all__ = [
    "DocumentProcessor",
    "ProcessedDocument",
    "get_doc_type",
    "TextChunker",
    "TextChunk",
    "EmbeddingService",
    "embed_documents",
    "embed_query",
]
