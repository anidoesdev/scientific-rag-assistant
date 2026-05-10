#domain:AI/ML + LLM system papers from arXiv
#question answering and synthesis over papers
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.upload import router as upload_router

START_TIME = time.time()

app = FastAPI(title="Scientific Rag Assistant")
app.state.start_time = START_TIME

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router, prefix="/api")
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(ingest_router)