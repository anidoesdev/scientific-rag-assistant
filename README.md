# Scientific RAG Assistant

A retrieval-augmented generation (RAG) system for question-answering over scientific papers. It combines vector similarity search with LLM-based reranking and grounded answer synthesis, with citations back to source chunks.

---

## Architecture

```
PDF Papers → Chunking → Embeddings (Ollama) → PostgreSQL pgvector
                                                        ↓
User Question → Cache Check → Vector Retrieval → LLM Reranking → Answer Generation → Response + Citations
```

**Multi-stage retrieval pipeline:**

1. **Ingest** — PDFs are chunked (1000 chars, 150 overlap) and stored as JSONL
2. **Embed** — Chunks are embedded via Ollama (`nomic-embed-text`) and upserted into PostgreSQL with pgvector
3. **Retrieve** — Query is embedded and top-10 candidates are fetched via cosine similarity (`<=>`)
4. **Rerank** — OpenAI LLM scores each chunk 0–10 for relevance; top-k selected
5. **Generate** — OpenAI LLM synthesizes a grounded answer using only the provided chunks, with source citations
6. **Cache** — Results are cached in-memory (1-hour TTL, keyed by SHA256 of normalized question)

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM (reranking + generation) | OpenAI `gpt-4o-mini` |
| Embeddings | Ollama `nomic-embed-text` (local) |
| Vector DB | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.0 |
| PDF parsing | PyMuPDF (fitz) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Validation | Pydantic v2 |

---

## Project Structure

```
├── app/
│   ├── api/ask.py              # POST /api/ask endpoint
│   ├── core/config.py          # Settings (env vars, DB, model config)
│   ├── db/session.py           # SQLAlchemy session factory
│   ├── prompts/                # Prompt templates for generation & query rewrite
│   ├── schemas/ask.py          # Request/response Pydantic models
│   └── services/
│       ├── cache.py            # In-memory query cache (1hr TTL)
│       ├── chunker.py          # PDF loading and text chunking
│       ├── embedder.py         # Ollama embedding client
│       ├── generator.py        # LLM answer synthesis
│       ├── reranker.py         # LLM-based chunk reranking
│       └── retriever.py        # pgvector similarity search
├── data/
│   ├── raw/                    # Input PDF papers
│   └── parsed/chunks.jsonl     # Chunked output
├── eval/
│   └── retrieval_eval.json     # Test questions with expected papers
├── scripts/
│   ├── embed_chunks.py         # Batch embed chunks → PostgreSQL
│   ├── eval_retrieval.py       # Evaluate retriever (Hit@K, MRR)
│   └── eval_reranker.py        # Compare baseline vs reranked retrieval
├── logs/retrieval.log          # Query logs with similarity scores
├── main.py                     # FastAPI app entry point
├── docker-compose.yml          # PostgreSQL + pgvector container
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker (for PostgreSQL)
- [Ollama](https://ollama.ai) running locally with `nomic-embed-text` pulled
- OpenAI API key

### 1. Start PostgreSQL

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 container with pgvector on port **5732**.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
openai_api_key=sk-...
db_host=localhost
db_port=5732
db_user=raguser
db_password=ragpassword
db_name=ragdb
embedding_model=nomic-embed-text
```

### 4. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 5. Ingest papers

Place PDF files in `data/raw/`, then run the ingestion pipeline:

```bash
# Chunk PDFs and save to data/parsed/chunks.jsonl
python -c "from app.services.chunker import chunk_papers; chunk_papers()"

# Embed chunks and upsert into PostgreSQL
python scripts/embed_chunks.py
```

### 6. Start the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API

### `POST /api/ask`

Ask a question over the indexed papers.

**Request:**
```json
{
  "question": "What are the main approaches to few-shot learning?",
  "k": 5
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question |
| `k` | int | 5 | Number of chunks to retrieve (1–20) |

**Response:**
```json
{
  "answer": "Few-shot learning approaches include...",
  "unsupported": false,
  "citations": [
    {
      "source_number": 1,
      "chunk_id": "paper_001_chunk_0042",
      "paper_id": "paper_001",
      "file_name": "arxiv-2024.pdf",
      "preview": "...first 220 characters of the chunk..."
    }
  ],
  "from_cache": false,
  "request_id": "abc123"
}
```

If the retrieved chunks don't support an answer, `unsupported` is set to `true` and the answer explains the gap.

---

## Database Schema

```sql
CREATE TABLE chunks (
  chunk_id   TEXT PRIMARY KEY,
  paper_id   TEXT,
  file_name  TEXT,
  source     TEXT,
  chunk_index             INT,
  total_chunks_for_paper  INT,
  text       TEXT,
  embedding  vector(768)
);
```

---

## Evaluation

### Retrieval quality

```bash
python scripts/eval_retrieval.py
```

Measures **Hit@K** and **MRR** against `eval/retrieval_eval.json` (test questions with expected paper IDs).

### Reranker impact

```bash
python scripts/eval_reranker.py
```

Compares baseline top-K (by cosine similarity) against LLM-reranked results, reporting Hit@K and MRR for both.

---

## Reranking Scoring Rubric

The reranker prompt instructs the LLM to score each chunk on a 0–10 scale:

| Score | Meaning |
|---|---|
| 10 | Directly answers the question |
| 7–9 | Highly relevant supporting evidence |
| 4–6 | Somewhat relevant background |
| 1–3 | Loosely related |
| 0 | Irrelevant |

Chunks are sorted by rerank score (primary) then cosine similarity (tiebreaker).

---

## Caching

Repeated questions bypass the retrieval and generation pipeline entirely. The cache key is a SHA256 hash of the lowercased, stripped question string. Entries expire after 1 hour. Cached responses include `"from_cache": true` in the response body.