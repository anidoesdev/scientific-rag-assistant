import json
from unittest.mock import MagicMock, patch

from app.services.cache import AnswerCache, _cache_key


# ── key helpers ──────────────────────────────────────────────────────────────

def test_cache_key_is_normalised():
    assert _cache_key("Hello World") == _cache_key("hello world")
    assert _cache_key("  test  ") == _cache_key("test")
    assert _cache_key("ATTENTION") == _cache_key("attention")


def test_cache_key_has_namespace_prefix():
    assert _cache_key("anything").startswith("rag:answer:")


def test_cache_key_is_deterministic():
    assert _cache_key("same question") == _cache_key("same question")


def test_different_questions_have_different_keys():
    assert _cache_key("question A") != _cache_key("question B")


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_returns_none_on_cache_miss():
    mock_client = MagicMock()
    mock_client.get.return_value = None
    with patch("app.services.cache._get_client", return_value=mock_client):
        assert AnswerCache().get("what is attention?") is None


def test_get_returns_parsed_dict_on_hit():
    payload = {"answer": "It is a mechanism.", "citations": [], "unsupported": False}
    mock_client = MagicMock()
    mock_client.get.return_value = json.dumps(payload)
    with patch("app.services.cache._get_client", return_value=mock_client):
        result = AnswerCache().get("what is attention?")
    assert result == payload


def test_get_returns_none_on_redis_exception():
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("connection refused")
    with patch("app.services.cache._get_client", return_value=mock_client):
        assert AnswerCache().get("test") is None


def test_get_returns_none_when_client_unavailable():
    with patch("app.services.cache._get_client", return_value=None):
        assert AnswerCache().get("test") is None


# ── set ───────────────────────────────────────────────────────────────────────

def test_set_calls_redis_with_json_and_ttl():
    payload = {"answer": "test"}
    mock_client = MagicMock()
    with patch("app.services.cache._get_client", return_value=mock_client):
        AnswerCache(ttl_seconds=1800).set("what is attention?", payload)

    mock_client.set.assert_called_once()
    _key, stored, *_ = mock_client.set.call_args.args
    assert json.loads(stored) == payload
    assert mock_client.set.call_args.kwargs.get("ex") == 1800


def test_set_is_silent_on_redis_exception():
    mock_client = MagicMock()
    mock_client.set.side_effect = Exception("timeout")
    with patch("app.services.cache._get_client", return_value=mock_client):
        AnswerCache().set("test", {"answer": "x"})  # must not raise


def test_set_is_noop_when_client_unavailable():
    with patch("app.services.cache._get_client", return_value=None):
        AnswerCache().set("test", {"answer": "x"})  # must not raise
