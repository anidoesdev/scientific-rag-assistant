from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy import text as sql_text

from app.core.config import get_settings
from app.db.session import SessionLocal

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    settings = get_settings()
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        user_id = int(payload["sub"])
        email: str = payload["email"]
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT id, email, name, picture FROM users WHERE id = :id"),
            {"id": user_id},
        ).first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return {"id": row[0], "email": row[1], "name": row[2], "picture": row[3]}


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
) -> Optional[dict]:
    if not credentials:
        return None
    settings = get_settings()
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None

    with SessionLocal() as session:
        row = session.execute(
            sql_text("SELECT id, email, name, picture FROM users WHERE id = :id"),
            {"id": user_id},
        ).first()

    if not row:
        return None
    return {"id": row[0], "email": row[1], "name": row[2], "picture": row[3]}
