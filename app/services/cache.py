import hashlib
import json
import logging
from typing import Dict, Optional

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client: Optional[redis.Redis] = None


def _get_client() -> Optional[redis.Redis]:
    global _client
    if _client is None:
        try:
            _client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            _client.ping()
        except Exception as e:
            logger.warning("Redis unavailable: %s — cache disabled", e)
            _client = None
    return _client


def _cache_key(question: str) -> str:
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return f"rag:answer:{digest}"


class AnswerCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds

    def get(self, question: str) -> Optional[Dict]:
        client = _get_client()
        if client is None:
            return None
        try:
            raw = client.get(_cache_key(question))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            return None

    def set(self, question: str, value: Dict) -> None:
        client = _get_client()
        if client is None:
            return
        try:
            client.set(_cache_key(question), json.dumps(value), ex=self.ttl)
        except Exception as e:
            logger.warning("Cache set failed: %s", e)


_cache = AnswerCache(ttl_seconds=settings.cache_ttl_seconds)


def get_answer_cache() -> AnswerCache:
    return _cache
