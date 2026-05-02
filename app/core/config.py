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
    openai_api_key: str

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()