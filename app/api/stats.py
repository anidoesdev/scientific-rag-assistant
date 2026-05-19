from fastapi import APIRouter
from sqlalchemy import text as sql_text

from app.core.config import get_settings
from app.db.session import SessionLocal

router = APIRouter()


@router.get("/stats")
async def get_stats():
    settings = get_settings()
    with SessionLocal() as session:
        paper_count = session.execute(
            sql_text("SELECT COUNT(DISTINCT paper_id) FROM chunks WHERE uploaded_by_user_id IS NULL")
        ).scalar() or 0

        chunk_count = session.execute(
            sql_text("SELECT COUNT(*) FROM chunks WHERE uploaded_by_user_id IS NULL")
        ).scalar() or 0

        avg_chunks = session.execute(
            sql_text(
                "SELECT COALESCE(AVG(c), 0) FROM "
                "(SELECT COUNT(*) c FROM chunks WHERE uploaded_by_user_id IS NULL GROUP BY paper_id) sub"
            )
        ).scalar() or 0

    return {
        "papers": int(paper_count),
        "chunks": int(chunk_count),
        "avg_chunks_per_paper": round(float(avg_chunks)),
        "embedding_model": settings.embedding_model,
        "embedding_dims": settings.embedding_dimension,
        "generation_model": settings.generation_model,
        "reranker_model": settings.reranker_model,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "retrieval_final_k": settings.retrieval_final_k,
        "similarity_threshold": settings.retrieval_similarity_threshold,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
    }
