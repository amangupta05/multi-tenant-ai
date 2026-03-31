# Multi-Tenant AI System — MVP Implementation Plan (Zero Cost + MCP)

## Overview

A **zero-cost, locally runnable** MVP of the multi-tenant AI WhatsApp backend.
All paid services replaced with free/local alternatives. Architecture kept clean enough to swap in paid services later.

**MCP (Model Context Protocol)** is built-in as the tool layer — any new tool is just a new MCP server. The LangGraph orchestrator connects to MCP servers as a client, making tool extensibility plug-and-play.

---

## Zero-Cost Stack

| Component | Production Choice | MVP (Free) Choice |
|---|---|---|
| LLM (Generation) | Gemini 2.5 Flash | **Gemini 2.5 Flash — Google AI Studio free tier** (1500 req/day, no billing) |
| Embeddings | OpenAI text-embedding-3-large | **`nomic-ai/nomic-embed-text-v1.5`** via `sentence-transformers` (100% local) |
| Reranker | Cohere Rerank | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** via `sentence-transformers` (local) |
| Vector DB | Qdrant Cloud | **Qdrant local mode** (in-process, no server, just a folder on disk) |
| Database | Postgres (schema-per-tenant) | **SQLite** + `aiosqlite` (zero setup, file on disk) |
| Session/Cache | Redis | **In-memory dict** (`TTLCache` from `cachetools`) |
| Queue | S3 + SQS | **`asyncio.Queue`** (in-process, no infra needed) |
| Storage | S3 | **Local filesystem** (`./data/tenants/{tenant_id}/`) |
| WhatsApp | Meta Cloud API | **Mock webhook** (POST JSON to `/webhook/mock`) |
| Speech-to-Text | OpenAI Whisper API | **`openai-whisper`** local model (tiny/base, runs on CPU) |
| Document Parsing | Docling | **Docling** (free, open-source, runs locally) |
| Architecture | Microservices | **Single FastAPI monolith** |

**Total running cost: $0** (Gemini free tier is generous enough for MVP testing)

---

## What MCP Adds

MCP (Model Context Protocol) is an open standard by Anthropic that lets LLMs call external tools/resources in a consistent way.

In this system:
- **Our tools ARE MCP servers** (retrieval, CRM mock, order mock, etc.)
- **LangGraph agent IS the MCP client** (via `langchain-mcp-adapters`)
- Adding a new tenant tool = spinning up a new MCP server. Zero changes to orchestrator.

```
LangGraph Agent (MCP Client)
        │
        ├── mcp://localhost:3001  →  RAG Knowledge Base Server
        ├── mcp://localhost:3002  →  CRM Lookup Server (mock)
        ├── mcp://localhost:3003  →  Order Status Server (mock)
        └── mcp://...            →  Any future tool
```

For MVP, all MCP servers run **in-process** (no network overhead) using stdio transport.
For production, they can be promoted to independent services over HTTP/SSE transport.

---

## MVP Architecture

```
Developer / Tester
       │
       ▼
POST /webhook/mock  ←── simulates WhatsApp message (text/image/audio)
       │
       ▼
[FastAPI Monolith - main.py]
       │
  ┌────┴────────────────────────────────────────────────────┐
  │                    ROUTES                                │
  │  POST /webhook/mock     → simulate WhatsApp input        │
  │  POST /ingest/file      → upload PDF/doc/image/audio     │
  │  POST /admin/tenant     → create tenant                  │
  │  GET  /admin/tenant/{id}→ get tenant info                │
  │  GET  /health           → health check                   │
  └────────────────────────────────────────────────────────┘
       │
       ▼
[Message Pipeline]
  1. Identify tenant (SQLite lookup)
  2. Multimodal preprocessing
     - text → pass through
     - image → Gemini Vision description
     - audio → local Whisper transcript
  3. Load session memory (TTLCache)
  4. Run LangGraph agent
  5. Save memory
  6. Return response JSON (mock WhatsApp response)
       │
       ▼
[LangGraph StateGraph]
  ┌────────────────────────────────────────────┐
  │  preprocess → memory_load → supervisor     │
  │                                ↓           │
  │               ┌────────────────┤           │
  │               ↓                ↓           │
  │           rag_node        tool_node        │
  │               │                │           │
  │           [MCP: RAG]    [MCP: Tools]       │
  │               └────────────────┤           │
  │                                ↓           │
  │                          generate_node     │
  │                                ↓           │
  │                          guardrail_node    │
  │                                ↓           │
  │                          memory_save       │
  └────────────────────────────────────────────┘
       │
       ▼
[MCP Servers - in-process stdio]
  rag_server.py      → search_knowledge_base(query, tenant_id, top_k)
  tools_server.py    → get_order_status(order_id), lookup_customer(phone)
```

---

## Project Structure (MVP)

```
multi-tenant AI system/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (pydantic-settings + .env)
│   │
│   ├── api/
│   │   ├── webhook.py             # POST /webhook/mock
│   │   ├── ingest.py              # POST /ingest/file, /ingest/url
│   │   └── admin.py               # Tenant CRUD
│   │
│   ├── core/
│   │   ├── pipeline.py            # Main message processing pipeline
│   │   ├── multimodal.py          # Image (Gemini Vision) + Audio (Whisper)
│   │   └── memory.py              # TTLCache session memory
│   │
│   ├── db/
│   │   ├── database.py            # SQLite + aiosqlite setup
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   └── crud.py                # DB operations
│   │
│   ├── ingestion/
│   │   ├── processor.py           # Docling PDF/web/Excel → markdown
│   │   ├── chunker.py             # Semantic chunking
│   │   └── embedder.py            # sentence-transformers embedder
│   │
│   ├── retrieval/
│   │   ├── qdrant_client.py       # Local Qdrant setup
│   │   ├── retriever.py           # Hybrid search (dense + BM25)
│   │   └── reranker.py            # Local cross-encoder reranker
│   │
│   ├── orchestrator/
│   │   ├── state.py               # LangGraph AgentState TypedDict
│   │   ├── graph.py               # StateGraph definition
│   │   └── nodes/
│   │       ├── preprocess.py      # Multimodal → text
│   │       ├── supervisor.py      # Intent classification (Gemini)
│   │       ├── rag.py             # RAG node (calls MCP RAG server)
│   │       ├── tools.py           # Tool execution node (calls MCP)
│   │       ├── generate.py        # Response generation (Gemini)
│   │       └── guardrail.py       # Safety check (Gemini)
│   │
│   └── mcp_servers/
│       ├── rag_server.py          # MCP server: search_knowledge_base tool
│       └── tools_server.py        # MCP server: mock CRM/order tools
│
├── data/                          # Local storage (gitignored)
│   └── tenants/
│       └── {tenant_id}/
│           ├── documents/         # Uploaded raw files
│           └── db.sqlite          # Per-tenant SQLite (or shared)
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_orchestrator.py
│
├── scripts/
│   ├── seed_tenant.py             # Create a test tenant + ingest sample docs
│   └── test_chat.py               # CLI chat tester
│
├── .env.example                   # Only GEMINI_API_KEY needed!
├── pyproject.toml                 # All dependencies
└── README.md
```

---

## Key Dependencies

```toml
[dependencies]
fastapi = ">=0.115"
uvicorn = {extras = ["standard"]}
langchain-google-genai = "*"      # Gemini via LangChain
langgraph = ">=0.2"
langchain-mcp-adapters = "*"      # MCP ↔ LangGraph bridge
mcp = "*"                          # MCP Python SDK
sentence-transformers = "*"       # Local embeddings + cross-encoder reranker
qdrant-client = "*"               # Local Qdrant (in-memory/disk mode)
docling = "*"                     # Document parsing (PDF, DOCX, XLSX, HTML)
openai-whisper = "*"              # Local speech-to-text
sqlalchemy = {extras = ["aiosqlite"]}
aiosqlite = "*"
pydantic-settings = "*"
cachetools = "*"                  # TTLCache for session memory
httpx = "*"                       # Async HTTP client
python-multipart = "*"            # File uploads
pillow = "*"                      # Image handling
```

---

## Proposed Changes (All New Files)

### Phase 1: Foundation
- `.env.example` — Only `GEMINI_API_KEY` required to run
- `pyproject.toml` — All deps
- `app/config.py` — Settings
- `app/main.py` — FastAPI app
- `app/db/` — SQLite models (tenants, conversations, documents)

### Phase 2: Ingestion
- `app/ingestion/processor.py` — Docling for all file types
- `app/ingestion/chunker.py` — Semantic chunking
- `app/ingestion/embedder.py` — `nomic-embed-text` via sentence-transformers
- `app/retrieval/qdrant_client.py` — Local Qdrant, single collection, `tenant_id` payload filter
- `app/api/ingest.py` — Upload endpoint

### Phase 3: MCP Servers
- `app/mcp_servers/rag_server.py` — Exposes `search_knowledge_base` as MCP tool
- `app/mcp_servers/tools_server.py` — Exposes mock `get_order_status`, `lookup_customer`, `check_inventory`

### Phase 4: Orchestrator (LangGraph + MCP)
- `app/orchestrator/state.py`
- `app/orchestrator/graph.py` — StateGraph with all nodes + conditional routing
- `app/orchestrator/nodes/*.py` — All 6 nodes
- `app/core/pipeline.py` — Connects everything

### Phase 5: Retrieval
- `app/retrieval/retriever.py` — Hybrid dense + sparse, tenant-scoped
- `app/retrieval/reranker.py` — Local cross-encoder

### Phase 6: Multimodal
- `app/core/multimodal.py` — Gemini Vision + local Whisper

### Phase 7: API & Webhook
- `app/api/webhook.py` — Mock WhatsApp endpoint
- `app/api/admin.py` — Tenant management

### Phase 8: Scripts & Tests
- `scripts/seed_tenant.py`
- `scripts/test_chat.py`
- `tests/`

---

## Open Questions (Resolved)

| Question | Decision |
|---|---|
| Production vs MVP | ✅ MVP monolith first |
| Cost | ✅ Zero cost — Gemini free tier + all local models |
| WhatsApp | ✅ Mock webhook for now |
| Auth | ✅ Simple API keys in header for MVP (`X-Tenant-ID`) |
| Tools | ✅ Generic MCP tool pattern + mock CRM/order tools |
| MCP | ✅ Yes — in-process stdio MCP servers, promotable to HTTP later |

---

## Verification Plan

1. `scripts/seed_tenant.py` — create tenant "demo", ingest `sample.pdf`
2. `scripts/test_chat.py` — interactive CLI that posts to `/webhook/mock`
3. Verify: text query → RAG response with source citations
4. Verify: image upload → Gemini Vision description → RAG on description
5. Verify: audio file → Whisper transcript → RAG on transcript  
6. Verify: "What is my order #1234?" → MCP tool `get_order_status` called → answer returned
7. Verify: unsafe query → guardrail intercepts, returns safe fallback
8. Verify: two different tenants → completely isolated responses (different knowledge bases)

**Estimated setup time:** `pip install` + set `GEMINI_API_KEY` in `.env` → `uvicorn app.main:app` → done.
