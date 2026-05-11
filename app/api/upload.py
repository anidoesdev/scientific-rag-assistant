import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import text as sql_text

from app.db.session import SessionLocal
from app.services.pipeline import RAW_DIR, ingest_single

router = APIRouter()


@router.get("/papers")
async def list_papers():
    with SessionLocal() as session:
        rows = session.execute(
            sql_text("SELECT DISTINCT paper_id, file_name FROM chunks ORDER BY paper_id")
        ).all()
    return [{"paper_id": r[0], "file_name": r[1]} for r in rows]


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / file.filename

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    with SessionLocal() as session:
        result = ingest_single(dest, session)

    if result.get("skipped"):
        return JSONResponse(
            status_code=200,
            content={
                "message": f"'{file.filename}' is already indexed — no changes made.",
                "result": result,
            },
        )

    return JSONResponse(
        status_code=201,
        content={
            "message": f"'{file.filename}' uploaded and ingested successfully.",
            "result": result,
        },
    )
