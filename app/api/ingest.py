"""
Ingestion API — Phase 2
========================
Handles document upload, web URL ingestion, status polling, and deletion.

Auth
----
Tenant endpoints require ``X-API-Key: sk-<key>`` (returned when tenant was created).
Get it from: POST /api/v1/admin/tenants → api_key field.

Ingestion flow
--------------
  POST /ingest/file
    1. Validate tenant via X-API-Key header.
    2. Save uploaded file to data/tenants/{tenant_id}/documents/.
    3. Create Document record in SQLite (status=pending).
    4. Return 202 immediately.
    5. Background task → processor → chunker → embedder → Qdrant → status=completed.

  POST /ingest/url
    1. Same flow but downloads the HTML page first, saves as .html file.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud, get_session
from app.db.schemas import DocumentResponse, SuccessResponse
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import embed_documents
from app.ingestion.processor import DocumentProcessor, SUPPORTED_EXTENSIONS, get_doc_type
from app.retrieval.qdrant_client import qdrant_service

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# ── Max file size: 50 MB ──────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# ── Shared processor and chunker instances ────────────────────────────────────
_processor = DocumentProcessor()
_chunker = TextChunker()


# ── Tenant auth (shared logic) ────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_tenant_from_api_key(
    x_api_key: str = Depends(_api_key_header),
    session: AsyncSession = Depends(get_session),
) -> Any:
    """
    Resolve X-API-Key header → Tenant ORM object.
    Raises 401 if key is missing/invalid, 403 if tenant is inactive.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    tenant = await crud.get_tenant_by_api_key(session, x_api_key)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unrecognised X-API-Key.",
        )
    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is deactivated.",
        )
    return tenant


# ── Background ingestion pipeline ─────────────────────────────────────────────

async def _run_ingestion_pipeline(
    doc_id: str,
    tenant_id: str,
    filepath: Path,
    doc_type: str,
    source_url: str | None = None,
) -> None:
    """
    Full ingestion pipeline executed as a FastAPI background task.
    Updates the Document record in SQLite at each stage.
    """
    # We need a fresh DB session — background tasks run outside the request context
    from app.db.database import _get_session_factory  # noqa: PLC0415

    factory = _get_session_factory()
    async with factory() as session:
        # ── Step 1: Mark as processing ────────────────────────────────────
        await crud.update_document_status(session, doc_id, status="processing")
        await session.commit()
        logger.info("▶  Ingestion started  doc_id='{}' file='{}'", doc_id, filepath.name)

        try:
            # ── Step 2: Extract text ──────────────────────────────────────
            processed = await _processor.process(filepath, doc_type, source_url=source_url)
            logger.debug("   Extracted {} chars from '{}'", len(processed.text), filepath.name)

            if not processed.text.strip():
                raise ValueError("Processor returned empty text — nothing to index.")

            # ── Step 3: Chunk ─────────────────────────────────────────────
            chunks = _chunker.chunk(processed, doc_id=doc_id, tenant_id=tenant_id)
            if not chunks:
                raise ValueError("Chunker produced zero chunks.")
            logger.debug("   Created {} chunks", len(chunks))

            # ── Step 4: Embed (Dense + Sparse) with Progress ──────────────
            import time
            texts = [c.text for c in chunks]
            loop = asyncio.get_event_loop()
            
            embeddings: list[list[float]] = []
            sparse_embeddings = None
            
            use_sparse = False
            encode_sparse_docs = None
            try:
                from app.retrieval.bm25 import encode_sparse_documents, get_backend  # noqa: PLC0415
                if get_backend() != "none":
                    use_sparse = True
                    encode_sparse_docs = encode_sparse_documents
                    sparse_embeddings = []
            except Exception as _sparse_exc:
                logger.warning("   Sparse encoding skipped (non-fatal): {}", _sparse_exc)

            batch_size = 32
            total_chunks = len(texts)
            start_time = time.time()
            
            # Initial UI update (Text extraction done = ~10%)
            await crud.update_document_status(
                session, doc_id, status="processing", 
                metadata_update={"progress_percent": 10, "eta_seconds": None}
            )
            await session.commit()

            for i in range(0, total_chunks, batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Dense
                batch_embeddings = await loop.run_in_executor(None, embed_documents, batch_texts)
                embeddings.extend(batch_embeddings)
                
                # Sparse
                if use_sparse and encode_sparse_docs:
                    try:
                        batch_sparse = await loop.run_in_executor(None, encode_sparse_docs, batch_texts)
                        sparse_embeddings.extend(batch_sparse)
                    except Exception:
                        use_sparse = False
                        sparse_embeddings = None
                
                # Update progress
                chunks_done = len(embeddings)
                progress = 10 + int((chunks_done / total_chunks) * 90)
                
                elapsed = time.time() - start_time
                rate = chunks_done / elapsed if elapsed > 0 else 0
                eta = int((total_chunks - chunks_done) / rate) if rate > 0 else 0
                
                await crud.update_document_status(
                    session, doc_id, status="processing",
                    metadata_update={"progress_percent": progress, "eta_seconds": eta}
                )
                await session.commit()
                
            logger.debug("   Embedded {} dense vectors (dim={})", len(embeddings), len(embeddings[0]))

            # ── Step 5: Upsert into Qdrant (dense + sparse) ───────────────
            _sparse = sparse_embeddings  # capture for lambda closure
            upserted = await loop.run_in_executor(
                None,
                lambda: qdrant_service.upsert_chunks(
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    texts=texts,
                    embeddings=embeddings,
                    sources=[c.source for c in chunks],
                    section_headings=[c.section_heading for c in chunks],
                    chunk_indices=[c.chunk_index for c in chunks],
                    metadata_list=[c.metadata for c in chunks],
                    sparse_embeddings=_sparse,
                ),
            )
            logger.debug(
                "   Upserted {} points (hybrid={})", upserted, _sparse is not None
            )

            # ── Step 6: Mark completed ────────────────────────────────────
            await crud.update_document_status(
                session,
                doc_id,
                status="completed",
                chunk_count=len(chunks),
                metadata_update={
                    "char_count": len(processed.text),
                    "section_count": len(processed.sections),
                    **processed.metadata,
                },
            )
            await session.commit()
            logger.success(
                "✅  Ingested '{}' → {} chunks indexed  (doc_id='{}')",
                filepath.name, len(chunks), doc_id,
            )

        except Exception as exc:
            logger.error("❌  Ingestion failed doc_id='{}': {}", doc_id, exc)
            await crud.update_document_status(
                session,
                doc_id,
                status="failed",
                error_message=str(exc),
            )
            await session.commit()


# ── Request / response helpers ────────────────────────────────────────────────

class UrlIngestRequest(BaseModel):
    url: HttpUrl
    title: str | None = None   # Optional override for the document title


class IngestAcceptedResponse(BaseModel):
    doc_id: str
    filename: str
    status: str = "pending"
    message: str = "Document accepted for ingestion. Poll /ingest/documents/{doc_id} for status."


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/file",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestAcceptedResponse,
    summary="Upload and ingest a document",
    description=(
        "Accepts PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, PNG, JPG, MP3, WAV, M4A files. "
        "Processing happens asynchronously — poll the status endpoint."
    ),
)
async def ingest_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    tenant: Any = Depends(get_tenant_from_api_key),
    session: AsyncSession = Depends(get_session),
) -> IngestAcceptedResponse:
    # ── Validate extension ────────────────────────────────────────────────
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix.lower().lstrip(".")

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File extension '.{ext}' is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # ── Read & size-check ─────────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum allowed size of {MAX_UPLOAD_BYTES // 1_048_576} MB.",
        )

    # ── Save to disk ──────────────────────────────────────────────────────
    docs_dir = settings.tenant_docs_path(tenant.id)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Prefix with UUID fragment to handle duplicate filenames
    safe_stem = Path(original_name).stem[:80]           # Trim very long names
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_stem}.{ext}"
    dest: Path = docs_dir / unique_name
    dest.write_bytes(content)
    logger.debug("Saved upload '{}' → '{}'", original_name, dest)

    # ── Create DB record ──────────────────────────────────────────────────
    doc_type = get_doc_type(dest)
    # Path relative to data_dir for portability
    rel_path = dest.relative_to(Path(settings.data_dir)).as_posix()

    doc = await crud.create_document(
        session,
        tenant_id=tenant.id,
        filename=original_name,
        file_path=rel_path,
        doc_type=doc_type,
        metadata={"original_filename": original_name, "size_bytes": len(content)},
    )
    # Force commit before returning so the UI's instant polling doesn't 404
    await session.commit()

    # ── Kick off background ingestion ─────────────────────────────────────
    background_tasks.add_task(
        _run_ingestion_pipeline,
        doc_id=doc.id,
        tenant_id=tenant.id,
        filepath=dest,
        doc_type=doc_type,
    )

    return IngestAcceptedResponse(doc_id=doc.id, filename=original_name)


@router.post(
    "/url",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestAcceptedResponse,
    summary="Ingest a web page by URL",
    description="Downloads the HTML content of a URL and ingests it as a document.",
)
async def ingest_url(
    body: UrlIngestRequest,
    background_tasks: BackgroundTasks,
    tenant: Any = Depends(get_tenant_from_api_key),
    session: AsyncSession = Depends(get_session),
) -> IngestAcceptedResponse:
    url_str = str(body.url)

    # ── Download the page ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url_str, headers={"User-Agent": "MultiTenantAI-bot/0.1"})
            resp.raise_for_status()
            html_bytes = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch URL: {exc}",
        )

    # ── Derive a safe filename from the URL ───────────────────────────────
    url_hash = hashlib.md5(url_str.encode()).hexdigest()[:8]
    title = body.title or url_str.split("/")[-1][:60] or "webpage"
    safe_name = f"{url_hash}_{title}.html"

    # ── Save to disk ──────────────────────────────────────────────────────
    docs_dir = settings.tenant_docs_path(tenant.id)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / safe_name
    dest.write_bytes(html_bytes)

    rel_path = dest.relative_to(Path(settings.data_dir)).as_posix()
    doc = await crud.create_document(
        session,
        tenant_id=tenant.id,
        filename=safe_name,
        file_path=rel_path,
        doc_type="html",
        metadata={"source_url": url_str, "size_bytes": len(html_bytes)},
    )
    # Force commit before returning so the UI's instant polling doesn't 404
    await session.commit()

    background_tasks.add_task(
        _run_ingestion_pipeline,
        doc_id=doc.id,
        tenant_id=tenant.id,
        filepath=dest,
        doc_type="html",
        source_url=url_str,
    )

    return IngestAcceptedResponse(doc_id=doc.id, filename=safe_name)


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    summary="List documents for tenant",
)
async def list_documents(
    status_filter: str | None = Query(None, alias="status", description="pending|processing|completed|failed"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    tenant: Any = Depends(get_tenant_from_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentResponse]:
    docs = await crud.list_documents(
        session,
        tenant_id=tenant.id,
        status=status_filter,
        offset=offset,
        limit=limit,
    )
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document ingestion status",
)
async def get_document(
    doc_id: str,
    tenant: Any = Depends(get_tenant_from_api_key),
    session: AsyncSession = Depends(get_session),
) -> DocumentResponse:
    doc = await crud.get_document(session, doc_id)

    if not doc or doc.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return DocumentResponse.model_validate(doc)


@router.delete(
    "/documents/{doc_id}",
    response_model=SuccessResponse,
    summary="Delete a document and its vectors",
)
async def delete_document(
    doc_id: str,
    tenant: Any = Depends(get_tenant_from_api_key),
    session: AsyncSession = Depends(get_session),
) -> SuccessResponse:
    doc = await crud.get_document(session, doc_id)

    if not doc or doc.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # 1. Remove Qdrant vectors
    loop = asyncio.get_event_loop()
    deleted_vectors = await loop.run_in_executor(
        None,
        lambda: qdrant_service.delete_by_document(
            tenant_id=tenant.id, doc_id=doc_id
        ),
    )

    # 2. Remove file from disk (best-effort)
    try:
        file_path = Path(settings.data_dir) / doc.file_path
        if file_path.exists():
            file_path.unlink()
    except Exception as exc:
        logger.warning("Could not delete file '{}': {}", doc.file_path, exc)

    # 3. Remove DB record
    await crud.delete_document(session, doc_id)

    return SuccessResponse(
        message=f"Document '{doc_id}' deleted ({deleted_vectors} vectors removed)."
    )


@router.get(
    "/stats",
    summary="Get vector store stats for this tenant",
    tags=["Ingestion"],
)
async def get_stats(
    tenant: Any = Depends(get_tenant_from_api_key),
) -> dict:
    loop = asyncio.get_event_loop()
    chunk_count = await loop.run_in_executor(
        None, lambda: qdrant_service.count_chunks(tenant.id)
    )
    collection_info = await loop.run_in_executor(
        None, qdrant_service.collection_info
    )
    return {
        "tenant_id": tenant.id,
        "indexed_chunks": chunk_count,
        "collection": collection_info,
    }
