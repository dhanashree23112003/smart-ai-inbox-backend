import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL  = os.getenv("DATABASE_URL")
SECRET_KEY    = os.getenv("SECRET_KEY")
FRONTEND_URL  = os.getenv("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL   = os.getenv("BACKEND_URL",  "http://localhost:8000")

_required = {
    "DATABASE_URL": DATABASE_URL,
    "SECRET_KEY":   SECRET_KEY,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        f"Make sure your .env file is present and contains these keys."
    )
