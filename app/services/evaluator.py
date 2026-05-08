"""
RAG answer quality evaluator.

Scores three dimensions for every generated answer:
  - faithfulness:      are all claims grounded in the provided chunks?
  - answer_relevance:  does the answer actually address the question?
  - context_relevance: were the retrieved chunks relevant to the question?

Uses the same LLM + structured-output pattern as generator.py / reranker.py.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.core.config import get_settings

settings = get_settings()


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #

class EvalScores(BaseModel):
    faithfulness: int = Field(
        description=(
            "0-10. How well every claim in the answer is supported by the "
            "provided chunks. 10 = fully grounded, 0 = complete hallucination."
        ),
        ge=0, le=10,
    )
    answer_relevance: int = Field(
        description=(
            "0-10. How directly the answer addresses the question. "
            "10 = fully answers, 0 = completely off-topic."
        ),
        ge=0, le=10,
    )
    context_relevance: int = Field(
        description=(
            "0-10. How relevant the retrieved chunks are to the question, "
            "regardless of the answer. 10 = highly relevant, 0 = irrelevant."
        ),
        ge=0, le=10,
    )
    faithfulness_reason: str = Field(description="One sentence explaining the faithfulness score.")
    answer_relevance_reason: str = Field(description="One sentence explaining the answer relevance score.")
    context_relevance_reason: str = Field(description="One sentence explaining the context relevance score.")


class EvalResult(BaseModel):
    faithfulness: int
    answer_relevance: int
    context_relevance: int
    overall: float
    faithfulness_reason: str
    answer_relevance_reason: str
    context_relevance_reason: str
    error: Optional[str] = None

    @classmethod
    def from_scores(cls, scores: EvalScores) -> "EvalResult":
        overall = round((scores.faithfulness + scores.answer_relevance + scores.context_relevance) / 3, 2)
        return cls(
            faithfulness=scores.faithfulness,
            answer_relevance=scores.answer_relevance,
            context_relevance=scores.context_relevance,
            overall=overall,
            faithfulness_reason=scores.faithfulness_reason,
            answer_relevance_reason=scores.answer_relevance_reason,
            context_relevance_reason=scores.context_relevance_reason,
        )

    @classmethod
    def error_result(cls, reason: str) -> "EvalResult":
        return cls(
            faithfulness=0,
            answer_relevance=0,
            context_relevance=0,
            overall=0.0,
            faithfulness_reason="",
            answer_relevance_reason="",
            context_relevance_reason="",
            error=reason,
        )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def _build_eval_prompt(question: str, answer: str, chunks: List[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Chunk {i}]\n"
            f"paper_id: {chunk.get('paper_id', '')}\n"
            f"text: {chunk.get('text', '')[:800]}"
        )
    context = "\n\n".join(context_parts)

    return f"""
You are a RAG system quality evaluator. Score the following on three dimensions.

QUESTION:
{question}

RETRIEVED CHUNKS:
{context}

GENERATED ANSWER:
{answer}

Scoring guide:
  faithfulness      — does every claim in the answer trace back to the chunks?
  answer_relevance  — does the answer directly address the question asked?
  context_relevance — are the chunks relevant to the question (ignore the answer)?

Be strict. A score of 10 requires near-perfect performance on that dimension.
""".strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def evaluate_answer(question: str, answer: str, chunks: List[dict]) -> EvalResult:
    """Score one generated answer against the question and retrieved chunks."""
    llm = ChatOpenAI(
        model=settings.generation_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    structured_llm = llm.with_structured_output(EvalScores)
    prompt = _build_eval_prompt(question, answer, chunks)

    try:
        scores: EvalScores = structured_llm.invoke(prompt)
        return EvalResult.from_scores(scores)
    except Exception as e:
        return EvalResult.error_result(str(e))
