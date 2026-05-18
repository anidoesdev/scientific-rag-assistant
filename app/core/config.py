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

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    openai_api_key: str
    generation_model: str = "gpt-4o-mini"
    reranker_model: str = "gpt-4o-mini"
    retrieval_candidate_k: int = 20
    retrieval_final_k: int = 5
    retrieval_similarity_threshold: float = 0.3
    retrieval_keyword_candidate_k: int = 20
    reranker_max_chars_per_chunk: int = 1800
    reranker_min_score: int = 4

    redis_host: str = "localhost"
    redis_port: int = 6380
    cache_ttl_seconds: int = 3600

    google_client_id: str = ""
    jwt_secret_key: str = "change-me-in-production"
    jwt_expire_hours: int = 168  # 1 week

    # Comma-separated list of allowed CORS origins, e.g.:
    # ALLOWED_ORIGINS=http://localhost:3000,http://10.0.0.5:3000
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()