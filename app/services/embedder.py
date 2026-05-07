from typing import List
from app.core.config import get_settings
import ollama


settings = get_settings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts.

    Replace this stub with a real call to your embedding provider.
    Ensure the returned vectors have a fixed dimension that matches the
    `VECTOR(d)` dimension in Postgres.
    """
    # Example shape documentation:
    # return [[0.0] * 1536 for _ in texts]  # dummy vectors
    
    if not texts:
        return []

    response = ollama.embed(model=settings.embedding_model, input=texts)
    
    
    embeddings = response.embeddings
    if embeddings is None:
        raise ValueError(f"Ollama response missing 'embeddings' key: ")
    
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )
    
    first_dim = len(embeddings[0]) if embeddings else 0
    if first_dim != settings.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: got {first_dim}, expected {settings.embedding_dimension}. "
            "Update settings.embedding_dimension or use a matching model."
        )

    return embeddings

   
# resp = ollama.embed(model=OLLAMA_EMBED_MODEL,input=["test string"])
# print(len(resp.embeddings[0]))
# print(len(resp.model_dump_json()["embeddings"][0]))
