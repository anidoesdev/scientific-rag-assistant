from unittest.mock import MagicMock, patch
from conftest import SAMPLE_CHUNKS
from app.services.evaluator import EvalResult, EvalScores, evaluate_answer


def _make_scores(f=8, ar=9, cr=7):
    return EvalScores(
        faithfulness=f,
        answer_relevance=ar,
        context_relevance=cr,
        faithfulness_reason="All claims are traced to the chunks.",
        answer_relevance_reason="Directly answers the question.",
        context_relevance_reason="Chunks are highly relevant.",
    )


# ── EvalResult.from_scores ────────────────────────────────────────────────────

def test_overall_is_mean_of_three_scores():
    result = EvalResult.from_scores(_make_scores(6, 8, 10))
    assert result.overall == round((6 + 8 + 10) / 3, 2)


def test_from_scores_copies_all_fields():
    scores = _make_scores(8, 9, 7)
    result = EvalResult.from_scores(scores)
    assert result.faithfulness == 8
    assert result.answer_relevance == 9
    assert result.context_relevance == 7
    assert result.error is None


# ── EvalResult.error_result ───────────────────────────────────────────────────

def test_error_result_sets_error_field():
    result = EvalResult.error_result("timeout")
    assert result.error == "timeout"


def test_error_result_zeros_all_scores():
    result = EvalResult.error_result("timeout")
    assert result.faithfulness == 0
    assert result.answer_relevance == 0
    assert result.context_relevance == 0
    assert result.overall == 0.0


# ── evaluate_answer ───────────────────────────────────────────────────────────

def test_evaluate_answer_returns_correct_scores():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = _make_scores(8, 9, 7)

    with patch("app.services.evaluator.ChatOpenAI", return_value=mock_llm):
        result = evaluate_answer("What is attention?", "It is a mechanism.", SAMPLE_CHUNKS)

    assert isinstance(result, EvalResult)
    assert result.faithfulness == 8
    assert result.answer_relevance == 9
    assert result.context_relevance == 7
    assert result.error is None


def test_evaluate_answer_returns_error_result_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")

    with patch("app.services.evaluator.ChatOpenAI", return_value=mock_llm):
        result = evaluate_answer("What is attention?", "Answer.", SAMPLE_CHUNKS)

    assert result.error is not None
    assert result.overall == 0.0


def test_evaluate_answer_handles_empty_chunks():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = _make_scores(5, 5, 0)

    with patch("app.services.evaluator.ChatOpenAI", return_value=mock_llm):
        result = evaluate_answer("test", "answer", [])

    assert result.context_relevance == 0
