# 🚀 Multi-Tenant AI System

A production-ready, highly-scalable, multi-tenant AI Backend and Chat Assistant engine built with **FastAPI**, **LangGraph**, **Groq**, **Qdrant**, and **MCP**.

This system handles infinite distinct tenants (clients), isolating their custom knowledge bases (RAG), connected business tools, conversational memory, and chat capabilities—all powered by an incredibly fast Llama 3 LLM via Groq. Includes a visually stunning built-in Local Dashboard for administration and chat testing.

## ✨ Features

- **True Multi-Tenancy:** Complete isolation of conversational memory (SQLite) and semantic knowledge bases (Qdrant filtering) across infinite tenants.
- **Microsecond Latency Responses:** Runs completely via Groq inference (`llama-3.3-70b-versatile` API) delivering end-to-end response times in under two seconds.
- **Intelligent Orchestration (LangGraph):** The conversational engine leverages a custom LangGraph pipeline:
  - *Supervisor Node:* Classifies user intent.
  - *Knowledge Retrieval (RAG):* Uses Qdrant + Sentence-Transformers for domain-specific Q&A.
  - *Tool Caller (MCP):* Plugs dynamically into external APIs to fetch Real-time CRM/Inventory data.
  - *Safety Guardrail:* Audits AI output against hallucinations and PII leaks before sending.
- **Multimodal Document Ingestion:** Powerful document ingestion pipeline natively processing PDFs (via Docling), Docx, CSVs, Audio, and Image parsing before storing them in the tenant's vector database.
- **Beautiful Dashboard GUI:** A sleek, fully dark-themed interactive HTML dashboard to create tenants, upload documents, tweak settings, and live-test the chat pipeline.

## 🛠️ Tech Stack
* **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Database & Memory:** [SQLite](https://www.sqlite.org/) + SQLAlchemy (Async)
* **Vector DB:** [Qdrant](https://qdrant.tech/) 
* **Model Inference (Orchestrator):** [Groq Cloud](https://groq.com/)
* **Routing & Workflow:** [LangGraph](https://www.langchain.com/langgraph)
* **External Integration Interface:** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

---

## ⚡ Quick Start

### 1. Requirements
Ensure you have Python 3.11+ installed and create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file from the starter template and grab your Free Groq API Key (from `console.groq.com`):
```bash
cp .env.example .env
```
Inside `.env`, edit the following standard variables:
```properties
DEBUG=true
GROQ_API_KEY=gsk_your_groq_key_here
ADMIN_SECRET_KEY=super-secret-admin-key
```

### 3. Spin Up the System!

Start the Uvicorn live-server from your project root:
```bash
uvicorn app.main:app --reload
```

---

## 🎮 How to Use the System

The easiest way to interact with your system is via the Dashboard.

1. **Access the Dashboard:** 
Open your browser to [**http://localhost:8000/dashboard/**](http://localhost:8000/dashboard/)

2. **Connect the App:** 
Upon first load, it will ask for an `Admin API Key`. Put in whatever you typed as `ADMIN_SECRET_KEY` in your `.env` file (e.g. `super-secret-admin-key`).

3. **Create a Tenant:**
Click the **"+ Create Tenant"** widget. Give them a name (like `Acme Corp`). The system will register them in SQLite and hand you back a *Tenant API Key* (`sk-tenant-...`). 
*Save this key into the Setup Overlay!*

4. **Upload Knowledge (RAG):**
Drag and drop any documents, PDFs, or CSVs into the Ingestion zone! The pipeline will chunk them, encode them using Sentence-Transformers, and insert them into the local Qdrant Vector Store strictly segregated to your current Tenant ID.

5. **Test the Chatbot:**
Now type a message logically testing the graph!
- Ask *"What is the return policy?"* → This will trigger the RAG Node.
- Ask *"Check the status of Order #1234"* → This will trigger the MCP Server Tool Node.

### API Integration (For Frontend Devs)
To build your own Web UI instead, you just hit the chat endpoint:
```http
POST /api/v1/chat/message
Headers:
  X-API-Key: <your_tenant_key>

Body:
{
  "session_id": "user-unique-thread-id",
  "message": "Hi, what's my order status?",
  "message_type": "text"
}
```

---

## 📂 Project Structure

```text
├── app/
│   ├── api/          # FastAPI Routers (admin, chat, ingestion, dashboard)
│   ├── core/         # LLM Init, Prompts, Memory (History TTLCache)
│   ├── db/           # Async SQLAlchemy ORM, Models, CRUD
│   ├── ingestion/    # Docling Parsers, Semantic Text Splitters, PDF Handlers
│   ├── orchestrator/ # Core LangGraph Nodes (Supervisor, Generating, Guardrail)
│   ├── retrieval/    # Qdrant client, Sentence-Transformer Embeddings
│   ├── mcp_servers/  # Tool definitions & external API mock responses
│   └── main.py       # FastAPI lifecycle
├── dashboard/        # Local Frontend interface (Vanilla JS / CSS)
├── data/             # Persistent local sqlite DB & qdrant storage (git-ignored)
└── requirements.txt  # Python packages
```

## ⚖️ License
[MIT License](LICENSE)
