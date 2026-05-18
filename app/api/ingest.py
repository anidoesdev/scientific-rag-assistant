from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.auth.dependencies import get_current_user
from app.db.session import SessionLocal
from app.services.pipeline import ingest_all_raw

router = APIRouter()


@router.post("/ingest")
async def ingest_papers(_user: dict = Depends(get_current_user)):
    with SessionLocal() as session:
        result = ingest_all_raw(session)

    ingested = result["ingested"]
    skipped = result["skipped"]
    failed = result["failed"]

    return JSONResponse(
        status_code=200,
        content={
            "message": (
                f"Ingested {len(ingested)} paper(s), "
                f"skipped {len(skipped)}, "
                f"failed {len(failed)}."
            ),
            "result": result,
        },
    )
