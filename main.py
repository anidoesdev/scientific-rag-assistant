#domain:AI/ML + LLM system papers from arXiv
#question answering and synthesis over papers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router

app = FastAPI(title="Scientific Rag Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router, prefix="/api")