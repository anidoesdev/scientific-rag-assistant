from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5732
    db_user: str = "raguser"
    db_password: str = "ragpassword"
    db_name: str = "ragdb"

    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    openai_api_key: str
    generation_model: str = "gpt-5-mini"
    reranker_model: str = "gpt-5-mini"
    retrieval_candidate_k: int = 20
    retrieval_final_k: int = 5
    retrieval_similarity_threshold: float = 0.3
    retrieval_keyword_candidate_k: int = 20
    reranker_max_chars_per_chunk: int = 1800
    reranker_min_score: int = 4

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()