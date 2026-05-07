import os
import pickle
import base64
import secrets

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy import text

from app.config import FRONTEND_URL, BACKEND_URL
from app.services.gmail_service import CREDENTIALS_PATH, SCOPES

router = APIRouter()

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

# In-memory cache: token → {"email", "creds"}
_cache:   dict = {}
_pending: dict = {}   # oauth state → Flow


def _db():
    from app.database import engine
    return engine


def _make_flow() -> Flow:
    return Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        redirect_uri=f"{BACKEND_URL}/auth/callback",
    )


def _creds_to_str(creds: Credentials) -> str:
    """Serialize credentials to base64 string (TEXT-safe for Postgres)."""
    return base64.b64encode(pickle.dumps(creds)).decode("utf-8")


def _creds_from_str(s: str) -> Credentials:
    return pickle.loads(base64.b64decode(s.encode("utf-8")))


def get_session(token: str) -> dict | None:
    """
    Return session dict for token.
    Checks memory first, then Supabase — sessions survive server restarts.
    """
    if not token:
        return None
    if token in _cache:
        return _cache[token]
    try:
        with _db().begin() as conn:
            row = conn.execute(
                text("SELECT email, creds_b64 FROM sessions WHERE token = :t"),
                {"t": token}
            ).fetchone()
        if row:
            session = {"email": row[0], "creds": _creds_from_str(row[1])}
            _cache[token] = session
            return session
    except Exception as e:
        print(f"[auth] session DB lookup error: {e}")
    return None


# ── /auth/login ───────────────────────────────────────────────────────────────

@router.get("/auth/login")
def auth_login():
    flow = _make_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    _pending[state] = flow
    return RedirectResponse(url=auth_url)


# ── /auth/callback ────────────────────────────────────────────────────────────

@router.get("/auth/callback")
def auth_callback(code: str = None, state: str = None, error: str = None):
    if error or not code or not state:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=auth_failed")

    flow = _pending.pop(state, None)
    if not flow:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=invalid_state")

    try:
        flow.fetch_token(code=code)
        creds = flow.credentials

        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = service.users().getProfile(userId="me").execute()
        email   = profile.get("emailAddress", "unknown@gmail.com")

        token    = secrets.token_urlsafe(32)
        creds_b64 = _creds_to_str(creds)

        # Persist to DB — sessions survive server restarts
        try:
            with _db().begin() as conn:
                conn.execute(text("""
                    INSERT INTO sessions (token, email, creds_b64, created_at)
                    VALUES (:t, :e, :c, NOW())
                    ON CONFLICT (token) DO UPDATE SET
                        email     = EXCLUDED.email,
                        creds_b64 = EXCLUDED.creds_b64,
                        created_at = NOW()
                """), {"t": token, "e": email, "c": creds_b64})
        except Exception as db_err:
            # DB persist failed — session still works in-memory for this run
            print(f"[auth] session DB persist failed (non-fatal): {db_err}")

        _cache[token] = {"email": email, "creds": creds}
        return RedirectResponse(url=f"{FRONTEND_URL}?session={token}&email={email}")

    except Exception as e:
        print(f"[auth/callback] error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=callback_failed")


# ── /auth/logout ──────────────────────────────────────────────────────────────

@router.get("/auth/logout")
def auth_logout(session_token: str = None):
    if session_token:
        _cache.pop(session_token, None)
        try:
            with _db().begin() as conn:
                conn.execute(
                    text("DELETE FROM sessions WHERE token = :t"),
                    {"t": session_token}
                )
        except Exception:
            pass
    return {"status": "logged out"}
