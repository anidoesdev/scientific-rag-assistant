#domain:AI/ML + LLM system papers from arXiv
#question answering and synthesis over papers
from fastapi import FastAPI
from app.api.ask import router as ask_router

app = FastAPI(title="Scientific Rag Assistant")

app.include_router(ask_router, prefix="/api")