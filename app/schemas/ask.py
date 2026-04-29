from pydantic import BaseModel, Field
from typing import List


class Citation(BaseModel):
    chunk_id: str
    paper_id: str
    similarity: float


class RetrievedChunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    similarity: float


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]