CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id                SERIAL PRIMARY KEY,
    chunk_id          TEXT UNIQUE NOT NULL,
    paper_id          TEXT NOT NULL,
    file_name         TEXT,
    source            TEXT,
    chunk_index       INTEGER NOT NULL,
    total_chunks_for_paper INTEGER NOT NULL,
    text              TEXT NOT NULL,
    embedding         VECTOR(1536), -- adjust dimension to your embedding model
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id
    ON chunks (paper_id);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id
    ON chunks (chunk_id);

-- vector index (use cosine distance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);