from fastapi import APIRouter
# from app.schemas.ask import 
from app.schemas.ask import AskRequest, AskResponse
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer
from app.services.reranker import rerank_chunks
from app.services.cache import get_answer_cache

router = APIRouter()

def build_no_answer_response() -> dict:
    return {
        "answer": "I couldn’t find enough evidence in the indexed papers to answer that confidently.",
        "unsupported": True,
        "citations": [],
        "from_cache": False,
    }


@router.post('/ask')
async def ask(req: AskRequest):
    cache = get_answer_cache()
    
    cached = cache.get(req.question)
    if cached:
        return {
            **cached,
            "from_cache": True
        }
    chunks = retrieve_chunks(req.question, k=10)
    
    if not chunks:
        cache.set(req.question, result)
        return build_no_answer_response()
        # result = {
        #     "answer": "I couldn't find relevant evidence in the indexed papers.",
        #     "unsupported": True,
        #     "citations": []
        # }
        
        # return {**result, "from_cache": False}
    reranked = rerank_chunks(req.question,chunks, top_n=req.k)
    result = generate_answer(req.question, reranked)
    cache.set(req.question, result)
    return {**result, "from_cache": False}
