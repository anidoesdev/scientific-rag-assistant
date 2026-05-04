from fastapi import APIRouter
# from app.schemas.ask import 
from app.schemas.ask import AskRequest, AskResponse
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer
from app.services.reranker import rerank_chunks

router = APIRouter()


@router.post('/ask')
async def ask(req: AskRequest):
    chunks = retrieve_chunks(req.question, k=5)
    
    if not chunks:
        return {
            "answer": "I couldn't find relevant evidence in the indexed papers.",
            "unsupported": True,
            "citations": []
        }
    # reranked = rerank_chunks(req.question,chunks, top_n=req.k)
    result = generate_answer(req.question, chunks)
    return result
