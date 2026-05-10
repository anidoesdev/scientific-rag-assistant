# ── Scientific RAG Assistant – FastAPI Backend ──────────────────────────────
# Build:  docker build -t rag-api .
# Run:    docker compose up api
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# System libraries required to compile psycopg2 and run PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies (cached layer) ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY app/        ./app/
COPY scripts/    ./scripts/
COPY main.py     .

# Ensure data directories exist inside the image;
# docker-compose mounts ./data as a named volume so uploads survive restarts.
RUN mkdir -p data/raw data/parsed

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000

# Use --workers 1 to keep the Ollama embedding client single-threaded.
# Increase workers only after confirming thread-safety.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
