from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel,Field
from app.core.config import get_settings

settings = get_settings()


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Grounded answer to the user's question")
    used_sources: List[int] = Field(
        default_factory=list,
        description="1-based source numbers actually used in the answer"
    )
    unsupported: bool = Field(
        default=False,
        description="True if the retrieved sources do not support a grounded answer"
    )

OPENAI_MODEL = "gpt-5-mini"


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i,chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"paper_id: {chunk['paper_id']}\n"
            f"file_name: {chunk.get('file_name','')}\n"
            f"text:\n{chunk['text']}\n"
        )
    return "\n\n".join(parts)


def build_prompt(question: str,chunks: list[dict]) -> str:
    context = build_context(chunks)

    return f"""
    You are a scientific research assistant.

    Answer the question using ONLY the provided sources.

    Rules:
    1. Do not use outside knowledge.
    2. If the sources do not support an answer, set unsupported=true.
    3. Return used_sources as the 1-based source numbers actually used.
    4. Only include source numbers that directly support the answer.
    5. Do not cite a source you did not use.
    6. Keep the answer concise and factual.

    Question:
    {question}

    Sources:
    {context}
    """.strip()


def generate_answer(question: str,chunks: list[dict]) -> dict:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=settings.openai_api_key
    )
    
    structured_llm = llm.with_structured_output(GeneratedAnswer)
    
    prompt = build_prompt(question, chunks)
    response = structured_llm.invoke(prompt)
    
    
    valid_sources = []
    max_source_num = len(chunks)

    seen = set()
    for src_num in response.used_sources:
        if isinstance(src_num, int) and 1 <= src_num <= max_source_num and src_num not in seen:
            seen.add(src_num)
            chunk = chunks[src_num - 1]
            valid_sources.append({
                "source_number": src_num,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "file_name": chunk.get("file_name"),
                "preview": chunk["text"][:220]
            })

    return {
        "answer": response.answer,
        "unsupported": response.unsupported,
        "citations": valid_sources
    }
    