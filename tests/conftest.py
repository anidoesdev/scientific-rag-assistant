import os
import sys
from unittest.mock import MagicMock

# ── Env vars ─────────────────────────────────────────────────────────────────
# Must be set before any app module is imported, because several services
# call get_settings() at module level and openai_api_key is required.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5732")
os.environ.setdefault("DB_USER", "raguser")
os.environ.setdefault("DB_PASSWORD", "ragpassword")
os.environ.setdefault("DB_NAME", "ragdb")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

# ── Stub unavailable runtime deps ────────────────────────────────────────────
# These packages live inside Docker (Redis, heavy LangChain extras).
# Every test that exercises these code paths patches at a higher level,
# so a MagicMock stub at import time is sufficient and correct.
_STUB_MODULES = [
    "redis",
    "psycopg2",
    "psycopg2.extensions",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_text_splitters",
    "langchain_openai",
    "langchain_classic",
    "langchain_classic.schema",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


SAMPLE_CHUNKS = [
    {
        "chunk_id": "paper_001_chunk_01",
        "paper_id": "paper_001",
        "file_name": "paper_001.pdf",
        "text": "Attention mechanisms allow models to focus on relevant parts of the input.",
        "similarity": 0.91,
    },
    {
        "chunk_id": "paper_001_chunk_02",
        "paper_id": "paper_001",
        "file_name": "paper_001.pdf",
        "text": "Self-attention computes queries, keys, and values from the same sequence.",
        "similarity": 0.85,
    },
    {
        "chunk_id": "paper_002_chunk_05",
        "paper_id": "paper_002",
        "file_name": "paper_002.pdf",
        "text": "Gradient descent updates parameters in the direction of steepest loss decrease.",
        "similarity": 0.62,
    },
]
