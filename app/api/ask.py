from fastapi import APIRouter
# from app.schemas.ask import 
from app.schemas.ask import AskRequest, AskResponse
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer


router = APIRouter()


@router.post('/ask',response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    chunks = retrieve_chunks(req.question, k=req.k)
    result = generate_answer(req.question, chunks)
    return AskResponse(**result)
