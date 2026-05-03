import ollama
from sqlalchemy import text as sql_text
from app.db.session import SessionLocal
#improving retriever
import logging
from pathlib import Path

OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 10
FINAL_K = 5
SIMILARITY_THRESHOLD = 0.35

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag.retriever")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "retrieval.log",encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def retrieve_chunks(question:str,k:int = FINAL_K) -> list[dict]:
    response = ollama.embed(
        model=OLLAMA_EMBED_MODEL,
        input=question
    )
    query_embeddings = response.embeddings[0]
    stmt = sql_text(
        """
        SELECT
        chunk_id,
        paper_id,
        text,
        1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM chunks
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_k;
        """
    )
    with SessionLocal() as session:
        rows = session.execute(
            stmt,
            {
                "query_embedding": f"[{','.join(map(str, query_embeddings))}]",
                "top_k": max(k,DEFAULT_TOP_K)
            }
        ).mappings().all()
    results =  [dict(row) for row in rows]
    filtered = [r for r in results if float(r["similarity"]) >= SIMILARITY_THRESHOLD][:k]
    logger.info("Question: %s",question)
    for r in filtered:
        preview = r["text"][:180].replace("\n"," ")
        logger.info(
            "chunk_id=%s paper_id=%s similarity=%.4f preview=%s",
            r["chunk_id"],
            r["paper_id"],
            float(r["similarity"]),
            preview,
        )
    return filtered 


# results = retrieve_chunks("What are unidirectional error correcting codes?", k=3)

# for row in results:
#     print(row["chunk_id"], row["similarity"])
#     print(row["text"][:300])
#     print("-" * 80)