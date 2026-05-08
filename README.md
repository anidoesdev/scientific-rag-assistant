# Scientific RAG Assistant

A retrieval-augmented generation (RAG) system for question-answering over scientific papers. Ask natural-language questions and get grounded, cited answers backed by a multi-stage retrieval pipeline.

---

## Architecture

```mermaid
flowchart TD
    U([User]) -->|question| FE[Next.js Frontend]
    FE -->|POST /api/ask| API[FastAPI]

    API -->|cache hit| FE
    API --> R[Retriever]

    R -->|embed query| OL[Ollama\nnomic-embed-text]
    R -->|cosine search top-20| PG[(PostgreSQL + pgvector)]
    R -->|keyword ILIKE top-20| PG
    R -->|RRF fusion + threshold filter| RR[Reranker]

    RR -->|score 0-10 per chunk| LLM[OpenAI LLM]
    RR -->|top-k chunks| GEN[Generator]

    GEN -->|synthesise answer + sources| LLM
    GEN --> CACHE[Redis Cache\n1-hr TTL]
    CACHE -->|answer + citations| FE
```

**Query pipeline:**
1. Check Redis cache (SHA-256 key, 1-hour TTL)
2. Embed query — Ollama `nomic-embed-text`, 768-dim
3. Dense search — cosine similarity, top-20 candidates from pgvector
4. Keyword search — `ILIKE`, top-20 candidates
5. RRF fusion — merge both lists by reciprocal rank
6. Similarity threshold filter (≥ 0.3); fallback to top-k if none pass
7. LLM reranker — score each chunk 0–10 for relevance
8. LLM generator — synthesise grounded answer with inline source numbers
9. Store in Redis, return with citations

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 18, Tailwind CSS, TypeScript |
| Backend API | FastAPI + Uvicorn |
| Embeddings | Ollama `nomic-embed-text` (local, 768-dim) |
| LLM | OpenAI `gpt-4o-mini` |
| Vector DB | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| ORM | SQLAlchemy 2 |
| PDF parsing | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/) installed and running
- OpenAI API key

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd scientific-rag-assistant
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

`.env` reference:

```env
OPENAI_API_KEY=sk-...

DB_HOST=localhost
DB_PORT=5732
DB_USER=raguser
DB_PASSWORD=ragpassword
DB_NAME=ragdb

REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_SECONDS=3600
```

### 2. Start infrastructure

```bash
docker-compose up -d
```

Starts PostgreSQL 16 + pgvector on port 5732 and Redis 7 on port 6379.

### 3. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 4. Install Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Ingest papers

```bash
# Chunk PDFs in data/raw/ → data/parsed/chunks.jsonl
python -m app.services.chunker

# Embed chunks and upsert into PostgreSQL
python scripts/embed_chunks.py
```

### 6. Start the API

```bash
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

### 7. Start the frontend

```bash
cd frontend
cp .env.example .env.local   # sets NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Frontend: `http://localhost:3000`

---

## API Reference

### `POST /api/ask`

Ask a question over the indexed papers.

**Request body**

```json
{
  "question": "How do transformer models handle long-range dependencies?",
  "k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Natural-language question |
| `k` | int 1–20 | 5 | Number of chunks to retrieve and cite |

**Response**

```json
{
  "answer": "Transformer models handle long-range dependencies through self-attention [1]...",
  "unsupported": false,
  "citations": [
    {
      "source_number": 1,
      "chunk_id": "paper_003_chunk_12",
      "paper_id": "paper_003",
      "file_name": "attention_is_all_you_need.pdf",
      "preview": "The attention mechanism allows the model to..."
    }
  ],
  "from_cache": false,
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

When `unsupported: true`, the indexed papers did not contain sufficient evidence. `citations` will be empty.

### `GET /health`

Check liveness and all dependencies.

**Response**

```json
{
  "status": "ok",
  "uptime_seconds": 142.3,
  "checks": {
    "database": { "status": "ok" },
    "ollama":   { "status": "ok" },
    "redis":    { "status": "ok" }
  }
}
```

`status` is `"degraded"` if any dependency check fails. Individual checks include a `"detail"` field explaining the failure.

---

## Evaluation

### Retrieval — Hit@K and MRR

```bash
python scripts/eval_retrieval.py
```

### Reranker — baseline vs reranked comparison

```bash
python scripts/eval_reranker.py
```

### Answer quality — faithfulness, relevance

```bash
python scripts/eval_answers.py
```

| Metric | Score |
|--------|-------|
| Hit@5 | — |
| MRR | — |
| Avg Faithfulness | — |
| Avg Answer Relevance | — |
| Avg Context Relevance | — |

> Run the eval scripts and fill in the table before sharing the project.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Project Structure

```
scientific-rag-assistant/
├── app/
│   ├── api/
│   │   ├── ask.py            # POST /api/ask
│   │   └── health.py         # GET /health
│   ├── core/
│   │   └── config.py         # Pydantic settings (env-driven)
│   ├── db/
│   │   └── session.py        # SQLAlchemy engine + SessionLocal
│   ├── prompts/              # Prompt templates
│   ├── schemas/
│   │   └── ask.py            # AskRequest / AskResponse / Citation
│   └── services/
│       ├── cache.py          # Redis answer cache (1-hr TTL)
│       ├── chunker.py        # PDF → text chunks
│       ├── embedder.py       # Ollama embedding client
│       ├── evaluator.py      # LLM-based RAG quality evaluator
│       ├── generator.py      # LLM answer synthesis with citations
│       ├── reranker.py       # LLM chunk relevance scorer
│       └── retriever.py      # pgvector search + RRF fusion
├── data/
│   ├── raw/                  # Source PDF papers
│   └── parsed/
│       └── chunks.jsonl      # Pre-chunked text
├── eval/
│   └── retrieval_eval.json   # Evaluation questions + expected papers
├── frontend/                 # Next.js app
├── scripts/
│   ├── embed_chunks.py       # Batch ingestion
│   ├── eval_retrieval.py     # Hit@K / MRR metrics
│   └── eval_reranker.py      # Reranker comparison
├── tests/                    # pytest unit tests
├── docker-compose.yml        # PostgreSQL + Redis
├── init.sql                  # DB schema (chunks table + indexes)
├── main.py                   # FastAPI app entry point + CORS
├── requirements.txt          # Runtime dependencies
└── requirements-dev.txt      # Test/dev dependencies
```
