import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text as sql_text

from app.auth.dependencies import get_current_user, get_optional_user
from app.db.session import SessionLocal
from app.services.pipeline import UPLOADS_DIR, ingest_single

router = APIRouter()


def _uploads_source_patterns() -> tuple[str, str]:
    """Return LIKE patterns for both Windows and Unix paths under UPLOADS_DIR."""
    base = str(UPLOADS_DIR)
    return base.replace("/", "\\") + "%", base.replace("\\", "/") + "%"


@router.get("/papers")
async def list_papers(user: dict | None = Depends(get_optional_user)):
    with SessionLocal() as session:
        rows = session.execute(
            sql_text(
                "SELECT DISTINCT paper_id, file_name, source, uploaded_by_user_id "
                "FROM chunks ORDER BY paper_id"
            )
        ).all()

        hidden: set[str] = set()
        if user:
            hidden_rows = session.execute(
                sql_text("SELECT paper_id FROM user_hidden_papers WHERE user_id = :uid"),
                {"uid": user["id"]},
            ).all()
            hidden = {r[0] for r in hidden_rows}

    win_prefix, unix_prefix = _uploads_source_patterns()
    result = []
    for paper_id, file_name, source, uploaded_by in rows:
        if paper_id in hidden:
            continue
        is_upload = (source or "").startswith(win_prefix[:-1]) or \
                    (source or "").startswith(unix_prefix[:-1])
        # Uploaded papers are only visible to their uploader
        if is_upload and (not user or uploaded_by != user["id"]):
            continue
        result.append({"paper_id": paper_id, "file_name": file_name, "is_session_upload": is_upload})
    return result


@router.get("/papers/{paper_id}/pdf")
async def get_paper_pdf(paper_id: str, _user: dict = Depends(get_current_user)):
    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT source FROM chunks WHERE paper_id = :pid LIMIT 1"),
            {"pid": paper_id},
        ).first()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

    path = Path(row[0])
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...), _user: dict = Depends(get_current_user)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / file.filename

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    with SessionLocal() as session:
        result = ingest_single(dest, session)
        if not result.get("skipped") and "paper_id" in result:
            session.execute(
                sql_text(
                    "UPDATE chunks SET uploaded_by_user_id = :uid WHERE paper_id = :pid"
                ),
                {"uid": user["id"], "pid": result["paper_id"]},
            )
            session.commit()

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
async def delete_paper(paper_id: str, user: dict = Depends(get_current_user)):
    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT source FROM chunks WHERE paper_id = :pid LIMIT 1"),
            {"pid": paper_id},
        ).first()

        if not row:
            raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

        source_path = row[0]
        win_prefix, unix_prefix = _uploads_source_patterns()
        is_upload = (source_path or "").startswith(win_prefix[:-1]) or \
                    (source_path or "").startswith(unix_prefix[:-1])

        if is_upload:
            session.execute(
                sql_text("DELETE FROM chunks WHERE paper_id = :pid"),
                {"pid": paper_id},
            )
            session.commit()

        else:
            session.execute(
                sql_text(
                    "INSERT INTO user_hidden_papers (user_id, paper_id) "
                    "VALUES (:uid, :pid) ON CONFLICT DO NOTHING"
                ),
                {"uid": user["id"], "pid": paper_id},
            )
            session.commit()
            return {"paper_id": paper_id, "hidden": True}

    deleted_file = None
    if source_path:
        path = Path(source_path)
        if path.exists():
            path.unlink(missing_ok=True)
            deleted_file = path.name

    return {"paper_id": paper_id, "deleted_file": deleted_file}


@router.delete("/uploads/cleanup")
async def cleanup_session_uploads(_user: dict = Depends(get_current_user)):
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
