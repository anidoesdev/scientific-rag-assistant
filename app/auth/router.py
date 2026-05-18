import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from jose import jwt
from pydantic import BaseModel
from sqlalchemy import text as sql_text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth")


class GoogleAuthRequest(BaseModel):
    credential: str


def _make_jwt(user_id: int, email: str, secret: str, expire_hours: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=expire_hours)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "exp": expire},
        secret,
        algorithm="HS256",
    )


@router.post("/google")
async def google_auth(body: GoogleAuthRequest):
    settings = get_settings()

    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")

    try:
        idinfo = await asyncio.to_thread(
            id_token.verify_oauth2_token,
            body.credential,
            google_requests.Request(),
            settings.google_client_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {exc}")

    google_sub = idinfo["sub"]
    email = idinfo.get("email", "")
    name = idinfo.get("name", "")
    picture = idinfo.get("picture", "")

    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT id FROM users WHERE google_sub = :sub"),
            {"sub": google_sub},
        ).first()

        if row:
            user_id = row[0]
            session.execute(
                sql_text(
                    "UPDATE users SET email=:email, name=:name, picture=:picture "
                    "WHERE id=:id"
                ),
                {"email": email, "name": name, "picture": picture, "id": user_id},
            )
        else:
            result = session.execute(
                sql_text(
                    "INSERT INTO users (google_sub, email, name, picture) "
                    "VALUES (:sub, :email, :name, :picture) RETURNING id"
                ),
                {"sub": google_sub, "email": email, "name": name, "picture": picture},
            )
            user_id = result.scalar()

        session.commit()

    token = _make_jwt(user_id, email, settings.jwt_secret_key, settings.jwt_expire_hours)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"email": email, "name": name, "picture": picture},
    }


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user
