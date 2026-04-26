from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "raguser"
    db_password: str = "ragpassword"
    db_name: str = "ragdb"

    # embedding model settings – adjust to what you actually use
    embedding_model: str = "text-embedding-3-small"

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()