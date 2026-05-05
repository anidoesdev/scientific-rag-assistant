from pydantic import BaseModel, Field
from typing import List,Optional


class Citation(BaseModel):
    source_number: int
    chunk_id: str
    paper_id: str
    file_name: Optional[str] = None
    preview: str


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
    unsupported: bool
    citations: List[Citation]
    from_cache: bool = False
    request_id: Optional[str] = None