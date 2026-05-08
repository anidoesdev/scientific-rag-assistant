import pytest
from app.services.retriever import _rrf_fuse

# Pure-logic tests — no DB or Ollama required.

DENSE = [
    {"chunk_id": "a", "paper_id": "p1", "text": "...", "similarity": 0.91},
    {"chunk_id": "b", "paper_id": "p1", "text": "...", "similarity": 0.85},
    {"chunk_id": "c", "paper_id": "p2", "text": "...", "similarity": 0.70},
]

KEYWORD = [
    {"chunk_id": "b", "paper_id": "p1", "text": "...", "similarity": 0.0},  # overlap with dense
    {"chunk_id": "d", "paper_id": "p3", "text": "...", "similarity": 0.0},
]


def test_rrf_fuse_respects_top_k():
    assert len(_rrf_fuse(DENSE, KEYWORD, k=2)) == 2


def test_rrf_fuse_returns_all_when_k_exceeds_total():
    result = _rrf_fuse(DENSE, KEYWORD, k=100)
    assert len(result) == 4  # a, b, c, d


def test_rrf_fuse_deduplicates_overlapping_chunks():
    result = _rrf_fuse(DENSE, KEYWORD, k=10)
    ids = [r["chunk_id"] for r in result]
    assert len(ids) == len(set(ids))


def test_rrf_fuse_boosts_chunks_in_both_lists():
    # chunk "b" appears in dense (rank 2) and keyword (rank 1) — should outscore "a" or "c"
    result = _rrf_fuse(DENSE, KEYWORD, k=4)
    ids = [r["chunk_id"] for r in result]
    assert ids.index("b") < ids.index("c")


def test_rrf_fuse_handles_empty_keyword():
    result = _rrf_fuse(DENSE, [], k=3)
    assert len(result) == 3
    assert {r["chunk_id"] for r in result} == {"a", "b", "c"}


def test_rrf_fuse_handles_empty_dense():
    result = _rrf_fuse([], KEYWORD, k=2)
    assert len(result) == 2


def test_rrf_fuse_handles_both_empty():
    assert _rrf_fuse([], [], k=5) == []


def test_rrf_fuse_adds_rrf_score_to_results():
    result = _rrf_fuse(DENSE, KEYWORD, k=4)
    for r in result:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0
