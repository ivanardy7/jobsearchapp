"""
Centralized configuration module.
Loads settings from .env file and provides config constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Helper to get configuration from environment or streamlit secrets
def _get_config(key: str, default: str = "") -> str:
    # 1. Try OS Environment
    val = os.getenv(key)
    if val:
        return val
    # 2. Try Streamlit Secrets fallback
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# ─── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY = _get_config("OPENAI_API_KEY", "")
OPENAI_MODEL = _get_config("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = _get_config("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ─── Database ─────────────────────────────────────────────
DATABASE_URL = _get_config("DATABASE_URL", "")

# ─── Vector Store ─────────────────────────────────────────
QDRANT_URL = _get_config("QDRANT_URL", "")
QDRANT_API_KEY = _get_config("QDRANT_API_KEY", "")
COLLECTION_NAME = "indonesian_jobs"


# ─── N8N ──────────────────────────────────────────────────
N8N_WEBHOOK_URL = _get_config("N8N_WEBHOOK_URL", "")
USE_N8N = _get_config("USE_N8N", "false").lower() == "true"


# ─── App Settings ─────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 100
SUPPORTED_CV_FORMATS = [".pdf", ".docx", ".doc"]
TOP_K_RESULTS = 10


def get_openai_api_key() -> str:
    """Get OpenAI API key, re-reading from st.secrets if needed."""
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    return _get_config("OPENAI_API_KEY", "")


def is_openai_configured() -> bool:
    """Check if OpenAI API key is set and valid-looking."""
    key = get_openai_api_key()
    return bool(key and key.startswith("sk-"))


def is_n8n_configured() -> bool:
    """Check if N8N webhook URL is set and USE_N8N is enabled."""
    use = USE_N8N
    if not use:
        use = _get_config("USE_N8N", "false").lower() == "true"
    url = N8N_WEBHOOK_URL
    if not url:
        url = _get_config("N8N_WEBHOOK_URL", "")
    return use and bool(url)

