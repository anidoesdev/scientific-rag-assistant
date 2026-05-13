import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import text as sql_text

from app.db.session import SessionLocal
from app.services.pipeline import UPLOADS_DIR, ingest_single

router = APIRouter()


def _uploads_source_patterns() -> tuple[str, str]:
    """Return LIKE patterns for both Windows and Unix paths under UPLOADS_DIR."""
    base = str(UPLOADS_DIR)
    return base.replace("/", "\\") + "%", base.replace("\\", "/") + "%"


@router.get("/papers")
async def list_papers():
    with SessionLocal() as session:
        rows = session.execute(
            sql_text("SELECT DISTINCT paper_id, file_name, source FROM chunks ORDER BY paper_id")
        ).all()
    win_prefix, unix_prefix = _uploads_source_patterns()
    return [
        {
            "paper_id": r[0],
            "file_name": r[1],
            "is_session_upload": (r[2] or "").startswith(win_prefix[:-1])
                or (r[2] or "").startswith(unix_prefix[:-1]),
        }
        for r in rows
    ]


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / file.filename

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


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT source FROM chunks WHERE paper_id = :pid LIMIT 1"),
            {"pid": paper_id},
        ).first()

        if not row:
            raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

        source_path = row[0]

        session.execute(
            sql_text("DELETE FROM chunks WHERE paper_id = :pid"),
            {"pid": paper_id},
        )
        session.commit()

    deleted_file = None
    if source_path:
        path = Path(source_path)
        if path.exists():
            path.unlink(missing_ok=True)
            deleted_file = path.name

    return {"paper_id": paper_id, "deleted_file": deleted_file}


@router.delete("/uploads/cleanup")
async def cleanup_session_uploads():
    """Delete all session-uploaded PDFs and their indexed chunks."""
    deleted_files: list[str] = []

    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.glob("*.pdf"):
            deleted_files.append(f.name)
            f.unlink(missing_ok=True)

    win_pat, unix_pat = _uploads_source_patterns()
    with SessionLocal() as session:
        session.execute(
            sql_text(
                "DELETE FROM chunks WHERE source LIKE :win OR source LIKE :unix"
            ),
            {"win": win_pat, "unix": unix_pat},
        )
        session.commit()

    return {"deleted_files": deleted_files, "count": len(deleted_files)}
