from typing import Dict, Tuple
from functools import lru_cache
import hashlib
import json
import time

CacheValue = Dict  # your answer schema dict


def _hash_query(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: Dict[str, Tuple[float, CacheValue]] = {}
        self.ttl = ttl_seconds

    def get(self, question: str) -> CacheValue | None:
        key = _hash_query(question)
        entry = self._store.get(key)
        if not entry:
            return None

        ts, value = entry
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None

        return value

    def set(self, question: str, value: CacheValue) -> None:
        key = _hash_query(question)
        self._store[key] = (time.time(), value)


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCache:
    return AnswerCache(ttl_seconds=3600)