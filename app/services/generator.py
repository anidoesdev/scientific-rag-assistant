from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel,Field
from app.core.config import get_settings

settings = get_settings()


class AnswerOutput(BaseModel):
    answer: str = Field(description="Grounded answer to the question")
    used_sources: List[int] = Field(description="List of source numbers actually used")


OPENAI_MODEL = "gpt-5-mini"


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i,chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"paper_id: {chunk['paper_id']}\n"
            f"similarity: {chunk['similarity']}\n"
            f"text:\n{chunk['text']}\n"
        )
    return "\n\n".join(parts)


def build_prompt(question: str,chunks: list[dict]) -> str:
    context = build_context(chunks)

    return f"""
        You are a scientific research assistant.

        Answer the user's question using only the provided retrieved context.
        If the answer is not supported by the context, say that clearly.
        Do not make up facts.
        Cite sources inline using [Source 1], [Source 2], etc. based on the provided context.

        User question:
        {question}

        Retrieved context:
        {context}

        Return a concise but helpful answer with inline citations.
        """.strip()


def generate_answer(question: str,chunks: list[dict]) -> dict:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=settings.openai_api_key
    )
    
    structured_llm = llm.with_structured_output(AnswerOutput)
    
    prompt = build_prompt(question, chunks)
    response = structured_llm.invoke(prompt)
    
    used_indices = set(response.used_sources)
    
    
    citations=[]
    retrieved_chunks = []
    
    for i, chunk in enumerate(chunks, start=1):
        if i in used_indices:
            citations.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "similarity": float(chunk["similarity"])
                }
            )
        retrieved_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "text": chunk["text"],
                "similarity": float(chunk["similarity"])            }
        )
    
    
    
    return {
        "answer":response.answer,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks
    }
    