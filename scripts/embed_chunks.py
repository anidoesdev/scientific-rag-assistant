import json
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy import text as sql_text

from app.db.session import SessionLocal
from app.services.embedder import embed_texts
from app.services.chunker import CHUNKS_PATH  # reuse path constant


BATCH_SIZE = 64  # tune based on your embedding API limits


def load_chunks_iter(path: Path):
    """Stream chunks.jsonl line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def insert_embeddings():
    """Embed chunk texts and upsert into Postgres."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found; run chunker first")

    session = SessionLocal()

    try:
        batch: List[Dict[str, Any]] = []
        total = 0

        for chunk_obj in load_chunks_iter(CHUNKS_PATH):
            batch.append(chunk_obj)

            if len(batch) >= BATCH_SIZE:
                total += process_batch(session, batch)
                batch.clear()

        # process remainder
        if batch:
            total += process_batch(session, batch)

        session.commit()
        print(f"Inserted/updated embeddings for {total} chunks.")

    finally:
        session.close()


def process_batch(session, batch: List[Dict[str, Any]]) -> int:
    texts = [c["text"] for c in batch]
    embeddings = embed_texts(texts)

    if len(embeddings) != len(batch):
        raise ValueError("Embedding count does not match batch size")

    # Build parameter list for bulk upsert
    params_list = []
    for chunk_obj, emb in zip(batch, embeddings):
        params_list.append(
            {
                "chunk_id": chunk_obj["chunk_id"],
                "paper_id": chunk_obj["paper_id"],
                "file_name": chunk_obj.get("file_name"),
                "source": chunk_obj.get("source"),
                "chunk_index": chunk_obj["chunk_index"],
                "total_chunks_for_paper": chunk_obj["total_chunks_for_paper"],
                "text": chunk_obj["text"],
                "embedding": emb,
            }
        )

    # Use a parameterized SQL with ON CONFLICT for idempotency
    stmt = sql_text(
        """
        INSERT INTO chunks (
            chunk_id,
            paper_id,
            file_name,
            source,
            chunk_index,
            total_chunks_for_paper,
            text,
            embedding
        )
        VALUES (
            :chunk_id,
            :paper_id,
            :file_name,
            :source,
            :chunk_index,
            :total_chunks_for_paper,
            :text,
            :embedding
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            paper_id = EXCLUDED.paper_id,
            file_name = EXCLUDED.file_name,
            source = EXCLUDED.source,
            chunk_index = EXCLUDED.chunk_index,
            total_chunks_for_paper = EXCLUDED.total_chunks_for_paper,
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding;
        """
    )

    for params in params_list:
        # psycopg2 understands Python list[float] -> pgvector if installed correctly.
        session.execute(stmt, params)

    session.commit()
    return len(batch)


if __name__ == "__main__":
    insert_embeddings()