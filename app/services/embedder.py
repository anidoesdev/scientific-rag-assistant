from typing import List
from openai import OpenAI
from app.core.config import get_settings

settings = get_settings()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = _get_client().embeddings.create(model=settings.embedding_model, input=texts)
    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(texts):
        raise ValueError(
            f"OpenAI returned {len(embeddings)} embeddings for {len(texts)} texts"
        )

    first_dim = len(embeddings[0]) if embeddings else 0
    if first_dim != settings.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: got {first_dim}, expected {settings.embedding_dimension}. "
            "Update settings.embedding_dimension or use a matching model."
        )

    return embeddings