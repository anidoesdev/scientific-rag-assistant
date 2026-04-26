from typing import List
from app.core.config import get_settings

settings = get_settings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts.

    Replace this stub with a real call to your embedding provider.
    Ensure the returned vectors have a fixed dimension that matches the
    `VECTOR(d)` dimension in Postgres.
    """
    # Example shape documentation:
    # return [[0.0] * 1536 for _ in texts]  # dummy vectors

    raise NotImplementedError("Implement embed_texts with your embedding API")