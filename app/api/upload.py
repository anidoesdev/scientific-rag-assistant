import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.db.session import SessionLocal
from app.services.pipeline import RAW_DIR, ingest_single

router = APIRouter()


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
