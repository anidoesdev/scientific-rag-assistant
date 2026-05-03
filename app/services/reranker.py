from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

settings = get_settings()

OPENAI_MODEL = "gpt-5-mini"

class ChunkScore(BaseModel):
    chunk_id: str = Field(description="The chunk id being scored")
    score: int = Field(description="Relevance score from 0 to 10",ge=0,le=10)

class RerankOutput(BaseModel):
    rankings: List[ChunkScore]


def build_rerank_prompt(question: str, chunks: list[dict]) -> str:
    parts = []

    for chunk in chunks:
        parts.append(
            f"""
        chunk_id: {chunk['chunk_id']}
        paper_id: {chunk['paper_id']}
        text:
        {chunk['text']}
        """.strip()
                )

    joined_chunks = "\n\n---\n\n".join(parts)

    return f"""
    You are a scientific retrieval reranker.

    Your task is to score each chunk for how well it answers the user's question.

    Scoring rules:
    - 10 = directly answers the question
    - 7-9 = highly relevant supporting evidence
    - 4-6 = somewhat relevant but incomplete
    - 1-3 = only loosely related
    - 0 = irrelevant

    Return scores for every chunk.
    Be strict. Prefer chunks that directly answer the question, not chunks that are just broadly related.

    Question:
    {question}

    Chunks:
    {joined_chunks}
    """.strip()

def rerank_chunks(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    structured_llm = llm.with_structured_output(RerankOutput)

    prompt = build_rerank_prompt(question, chunks)
    result = structured_llm.invoke(prompt)

    score_map = {item.chunk_id: item.score for item in result.rankings}

    reranked = []
    for chunk in chunks:
        reranked.append({
            **chunk,
            "rerank_score": score_map.get(chunk["chunk_id"], 0)
        })

    reranked.sort(
        key=lambda x: (
            x["rerank_score"],
            float(x.get("similarity", 0.0))
        ),
        reverse=True
    )

    return reranked[:top_n]