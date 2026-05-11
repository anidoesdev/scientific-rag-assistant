import time
import redis
from fastapi import APIRouter, Request
from sqlalchemy import text
from app.db.session import engine
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


def _check_database() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_openai() -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        client.models.retrieve(settings.embedding_model)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_redis() -> dict:
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=0,
            socket_connect_timeout=2,
        )
        client.ping()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/health", tags=["health"])
def health(request: Request):
    db = _check_database()
    openai = _check_openai()
    cache = _check_redis()

    all_ok = all(c["status"] == "ok" for c in [db, openai, cache])
    uptime = round(time.time() - request.app.state.start_time, 1)

    return {
        "status": "ok" if all_ok else "degraded",
        "uptime_seconds": uptime,
        "checks": {
            "database": db,
            "openai": openai,
            "redis": cache,
        },
    }
