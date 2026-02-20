"""
Конфигурация приложения.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API = os.getenv("QDRANT_API")
QDRANT_URL = os.getenv("QDRANT_URL")

DB_DSN = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

DB_HOST = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "127.0.0.1"
DB_PORT = _get_int("DB_PORT", _get_int("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
DB_USER = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")

DB_CONNECT_TIMEOUT = _get_int("DB_CONNECT_TIMEOUT", 10)
DB_POOL_MIN_CONN = _get_int("DB_POOL_MIN_CONN", 1)
DB_POOL_MAX_CONN = _get_int("DB_POOL_MAX_CONN", 10)

_raw = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGINS = [s.strip() for s in _raw.split(",") if s.strip()] or None