from pydantic import BaseModel
from typing import List

class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    paper_id: str
    title: str
    chunk_id: str

class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
