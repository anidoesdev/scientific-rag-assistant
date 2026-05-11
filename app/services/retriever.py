from openai import OpenAI
from sqlalchemy import text as sql_text
from app.db.session import SessionLocal
import logging
from pathlib import Path
from functools import lru_cache
from app.core.config import get_settings

settings = get_settings()
_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag.retriever")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "retrieval.log",encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

@lru_cache(maxsize=512)
def _cached_query_embedding(normalized_question: str) -> list[float]:
    response = _get_client().embeddings.create(model=settings.embedding_model, input=normalized_question)
    embeddings = [item.embedding for item in response.data]
    if not embeddings:
        raise ValueError("Embedding provider returned no embeddings")
    vector = embeddings[0]
    if len(vector) != settings.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension mismatch: got {len(vector)}, expected {settings.embedding_dimension}"
        )
    return vector

def get_query_embedding(question:str) -> list[float]:
    normalized = question.strip().lower()
    return _cached_query_embedding(normalized)

def _vector_search(query_embeddings: list[float], top_k: int) -> list[dict]:
    stmt = sql_text(
        """
        SELECT
            chunk_id,
            paper_id,
            file_name,
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
                "top_k": top_k,
            },
        ).mappings().all()
    return [dict(row) for row in rows]

def _keyword_search(question: str, top_k: int) -> list[dict]:
    stmt = sql_text(
        """
        SELECT
            chunk_id,
            paper_id,
            file_name,
            text,
            0.0 AS similarity
        FROM chunks
        WHERE text ILIKE :q
        ORDER BY id DESC
        LIMIT :top_k;
        """
    )
    with SessionLocal() as session:
        rows = session.execute(
            stmt,
            {
                "q": f"%{question.strip()}%",
                "top_k": top_k,
            },
        ).mappings().all()
    return [dict(row) for row in rows]

def _rrf_fuse(dense: list[dict], keyword: list[dict], k: int) -> list[dict]:
    dense_rank = {row["chunk_id"]: i + 1 for i, row in enumerate(dense)}
    keyword_rank = {row["chunk_id"]: i + 1 for i, row in enumerate(keyword)}
    by_id: dict[str, dict] = {}

    for row in dense + keyword:
        cid = row["chunk_id"]
        if cid not in by_id:
            by_id[cid] = dict(row)
        by_id[cid]["rrf_score"] = (1.0 / (60 + dense_rank.get(cid, 10_000))) + (
            1.0 / (60 + keyword_rank.get(cid, 10_000))
        )

    fused = sorted(
        by_id.values(),
        key=lambda r: (float(r.get("rrf_score", 0.0)), float(r.get("similarity", 0.0))),
        reverse=True,
    )
    return fused[:k]

def retrieve_chunks(question:str, k:int|None = None) -> list[dict]:
    final_k = k or settings.retrieval_final_k
    query_embeddings = get_query_embedding(question)
    dense = _vector_search(query_embeddings, top_k=max(final_k, settings.retrieval_candidate_k))
    keyword = _keyword_search(question, top_k=settings.retrieval_keyword_candidate_k)
    fused = _rrf_fuse(dense, keyword, k=max(final_k, settings.retrieval_candidate_k))
    filtered = [
        r for r in fused if float(r.get("similarity", 0.0)) >= settings.retrieval_similarity_threshold
    ]
    if not filtered:
        filtered = fused[:final_k]

    filtered = filtered[:final_k]
    logger.info("Question: %s",question)
    for r in filtered:
        preview = r["text"][:180].replace("\n"," ")
        logger.info(
            "chunk_id=%s paper_id=%s similarity=%.4f rrf=%.6f preview=%s",
            r["chunk_id"],
            r["paper_id"],
            float(r["similarity"]),
            float(r.get("rrf_score", 0.0)),
            preview,
        )
    return filtered


# results = retrieve_chunks("What are unidirectional error correcting codes?", k=3)

# for row in results:
#     print(row["chunk_id"], row["similarity"])
#     print(row["text"][:300])
#     print("-" * 80)