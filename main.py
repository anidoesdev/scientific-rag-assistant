#domain:AI/ML + LLM system papers from arXiv
#question answering and synthesis over papers
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.upload import router as upload_router
from app.auth.router import router as auth_router
from app.core.config import get_settings

START_TIME = time.time()


def _cleanup_uploads_on_startup() -> None:
    """Remove all session-uploaded PDFs and their DB chunks on server start."""
    try:
        from app.services.pipeline import UPLOADS_DIR
        from app.db.session import SessionLocal
        from sqlalchemy import text as sql_text

        if UPLOADS_DIR.exists():
            for f in UPLOADS_DIR.glob("*.pdf"):
                f.unlink(missing_ok=True)

        base = str(UPLOADS_DIR)
        win_pat = base.replace("/", "\\") + "%"
        unix_pat = base.replace("\\", "/") + "%"
        with SessionLocal() as session:
            session.execute(
                sql_text("DELETE FROM chunks WHERE source LIKE :win OR source LIKE :unix"),
                {"win": win_pat, "unix": unix_pat},
            )
            session.commit()
    except Exception:
        pass  # DB may not be ready on very first boot; uploads will be cleaned on next start


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_uploads_on_startup()
    yield


app = FastAPI(title="Scientific Rag Assistant", lifespan=lifespan)
app.state.start_time = START_TIME

_origins = [o.strip() for o in get_settings().allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(ask_router, prefix="/api")
app.include_router(health_router)
app.include_router(upload_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
