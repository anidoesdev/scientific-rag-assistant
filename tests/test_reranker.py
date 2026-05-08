from unittest.mock import MagicMock, patch
from conftest import SAMPLE_CHUNKS
from app.services.reranker import rerank_chunks


def _mock_llm(rankings):
    mock_output = MagicMock(rankings=rankings)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_output
    return mock_llm


# ── edge cases ────────────────────────────────────────────────────────────────

def test_rerank_empty_input_returns_empty():
    result = rerank_chunks("what is attention?", [], top_n=5)
    assert result == []


# ── sorting ───────────────────────────────────────────────────────────────────

def test_rerank_sorts_by_score_descending():
    rankings = [
        MagicMock(chunk_id="paper_002_chunk_05", score=9),
        MagicMock(chunk_id="paper_001_chunk_01", score=6),
        MagicMock(chunk_id="paper_001_chunk_02", score=3),
    ]
    with patch("app.services.reranker.ChatOpenAI", return_value=_mock_llm(rankings)):
        result = rerank_chunks("what is attention?", SAMPLE_CHUNKS, top_n=3)

    assert result[0]["chunk_id"] == "paper_002_chunk_05"
    assert result[1]["chunk_id"] == "paper_001_chunk_01"
    assert result[2]["chunk_id"] == "paper_001_chunk_02"


def test_rerank_attaches_rerank_score_to_chunks():
    rankings = [MagicMock(chunk_id=c["chunk_id"], score=5) for c in SAMPLE_CHUNKS]
    with patch("app.services.reranker.ChatOpenAI", return_value=_mock_llm(rankings)):
        result = rerank_chunks("test", SAMPLE_CHUNKS, top_n=3)

    assert all("rerank_score" in r for r in result)


# ── top_n ─────────────────────────────────────────────────────────────────────

def test_rerank_truncates_to_top_n():
    rankings = [MagicMock(chunk_id=c["chunk_id"], score=7) for c in SAMPLE_CHUNKS]
    with patch("app.services.reranker.ChatOpenAI", return_value=_mock_llm(rankings)):
        result = rerank_chunks("test", SAMPLE_CHUNKS, top_n=2)

    assert len(result) == 2


# ── fallback ──────────────────────────────────────────────────────────────────

def test_rerank_falls_back_to_retrieval_order_on_llm_error():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")

    with patch("app.services.reranker.ChatOpenAI", return_value=mock_llm):
        result = rerank_chunks("test", SAMPLE_CHUNKS, top_n=2)

    assert result == SAMPLE_CHUNKS[:2]


def test_rerank_fallback_respects_top_n():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("timeout")

    with patch("app.services.reranker.ChatOpenAI", return_value=mock_llm):
        result = rerank_chunks("test", SAMPLE_CHUNKS, top_n=1)

    assert len(result) == 1
