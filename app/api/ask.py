from fastapi import APIRouter
import uuid
from app.schemas.ask import AskRequest, AskResponse
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer
from app.services.reranker import rerank_chunks
from app.services.cache import get_answer_cache
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

def build_no_answer_response(request_id: str) -> dict:
    return {
        "answer": "I couldn’t find enough evidence in the indexed papers to answer that confidently.",
        "unsupported": True,
        "citations": [],
        "from_cache": False,
        "request_id": request_id,
    }


@router.post('/ask', response_model=AskResponse)
async def ask(req: AskRequest):
    request_id = str(uuid.uuid4())
    cache = get_answer_cache()
    
    cached = cache.get(req.question)
    if cached:
        return {
            **cached,
            "from_cache": True,
            "request_id": request_id,
        }
    chunks = retrieve_chunks(req.question, k=max(req.k, settings.retrieval_final_k))
    
    if not chunks:
        print("chunks are not there!!")
        result = build_no_answer_response(request_id)
        cache.set(req.question, result)
        return result

    reranked = rerank_chunks(req.question, chunks, top_n=req.k)
    scores_present = any("rerank_score" in c for c in reranked)
    if reranked and scores_present and max(int(c.get("rerank_score", 0)) for c in reranked) < settings.reranker_min_score:
        result = build_no_answer_response(request_id)
        cache.set(req.question, result)
        return result

    result = generate_answer(req.question, reranked)
    cache.set(req.question, result)
    return {**result, "from_cache": False, "request_id": request_id}
