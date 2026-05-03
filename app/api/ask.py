from fastapi import APIRouter
# from app.schemas.ask import 
from app.schemas.ask import AskRequest, AskResponse
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer
from app.services.reranker import rerank_chunks

router = APIRouter()


@router.post('/ask',response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    chunks = retrieve_chunks(req.question, k=10)
    reranked = rerank_chunks(req.question,chunks, top_n=req.k)
    result = generate_answer(req.question, reranked)
    return AskResponse(**result)
