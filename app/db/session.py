from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings


settings = get_settings()

DATABASE_URL = (
    f"postgresql+psycopg2://{settings.db_user}:"
    f"{settings.db_password}@{settings.db_host}:"
    f"{settings.db_port}/{settings.db_name}"
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)