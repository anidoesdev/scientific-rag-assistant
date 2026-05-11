from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from conftest import SAMPLE_CHUNKS
from main import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200_when_all_ok():
    with patch("app.api.health._check_database", return_value={"status": "ok"}), \
         patch("app.api.health._check_openai", return_value={"status": "ok"}), \
         patch("app.api.health._check_redis", return_value={"status": "ok"}):
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "uptime_seconds" in body
    assert set(body["checks"].keys()) == {"database", "openai", "redis"}


def test_health_status_degraded_when_db_is_down():
    with patch("app.api.health._check_database", return_value={"status": "error", "detail": "refused"}), \
         patch("app.api.health._check_openai", return_value={"status": "ok"}), \
         patch("app.api.health._check_redis", return_value={"status": "ok"}):
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_health_status_degraded_when_redis_is_down():
    with patch("app.api.health._check_database", return_value={"status": "ok"}), \
         patch("app.api.health._check_openai", return_value={"status": "ok"}), \
         patch("app.api.health._check_redis", return_value={"status": "error", "detail": "refused"}):
        resp = client.get("/health")

    assert resp.json()["status"] == "degraded"


# ── /api/ask — cache hit ──────────────────────────────────────────────────────

def test_ask_returns_cached_result():
    cached = {
        "answer": "Cached answer.",
        "unsupported": False,
        "citations": [],
        "from_cache": True,
        "request_id": "old-id",
    }
    mock_cache = MagicMock()
    mock_cache.get.return_value = cached

    with patch("app.api.ask.get_answer_cache", return_value=mock_cache):
        resp = client.post("/api/ask", json={"question": "what is attention?", "k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["from_cache"] is True
    assert body["answer"] == "Cached answer."


# ── /api/ask — cache miss, no chunks ─────────────────────────────────────────

def test_ask_returns_unsupported_when_no_chunks_retrieved():
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("app.api.ask.get_answer_cache", return_value=mock_cache), \
         patch("app.api.ask.retrieve_chunks", return_value=[]):
        resp = client.post("/api/ask", json={"question": "nonsense query", "k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["unsupported"] is True
    assert body["citations"] == []


# ── /api/ask — full pipeline ──────────────────────────────────────────────────

def test_ask_returns_answer_when_pipeline_succeeds():
    generated = {
        "answer": "Attention is a mechanism.",
        "unsupported": False,
        "citations": [
            {
                "source_number": 1,
                "chunk_id": "paper_001_chunk_01",
                "paper_id": "paper_001",
                "file_name": "paper_001.pdf",
                "preview": "Attention mechanisms allow...",
            }
        ],
    }
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    reranked = [{**c, "rerank_score": 8} for c in SAMPLE_CHUNKS]
    with patch("app.api.ask.get_answer_cache", return_value=mock_cache), \
         patch("app.api.ask.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
         patch("app.api.ask.rerank_chunks", return_value=reranked), \
         patch("app.api.ask.generate_answer", return_value=generated):
        resp = client.post("/api/ask", json={"question": "what is attention?", "k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Attention is a mechanism."
    assert body["from_cache"] is False
    assert "request_id" in body


# ── /api/ask — low rerank score ───────────────────────────────────────────────

def test_ask_returns_unsupported_when_rerank_score_below_threshold():
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    low_score_chunks = [{**c, "rerank_score": 1} for c in SAMPLE_CHUNKS]
    with patch("app.api.ask.get_answer_cache", return_value=mock_cache), \
         patch("app.api.ask.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
         patch("app.api.ask.rerank_chunks", return_value=low_score_chunks):
        resp = client.post("/api/ask", json={"question": "vague question", "k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["unsupported"] is True
    assert body["citations"] == []


# ── /api/ask — input validation ───────────────────────────────────────────────

def test_ask_rejects_k_below_minimum():
    resp = client.post("/api/ask", json={"question": "test", "k": 0})
    assert resp.status_code == 422


def test_ask_rejects_k_above_maximum():
    resp = client.post("/api/ask", json={"question": "test", "k": 21})
    assert resp.status_code == 422


def test_ask_rejects_missing_question():
    resp = client.post("/api/ask", json={"k": 5})
    assert resp.status_code == 422
