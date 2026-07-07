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

# ─── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ─── Database ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ─── Vector Store ─────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "indonesian_jobs"


# ─── N8N ──────────────────────────────────────────────────
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
USE_N8N = os.getenv("USE_N8N", "false").lower() == "true"

# ─── Dataset ──────────────────────────────────────────────
DATASET_PATH = BASE_DIR / "Dataset" / "jobs.jsonl"
DATA_DIR = BASE_DIR / "data"

# ─── App Settings ─────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 100
SUPPORTED_CV_FORMATS = [".pdf", ".docx", ".doc"]
TOP_K_RESULTS = 10


def is_openai_configured() -> bool:
    """Check if OpenAI API key is set and valid-looking."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))


def is_n8n_configured() -> bool:
    """Check if N8N webhook URL is set and USE_N8N is enabled."""
    return USE_N8N and bool(N8N_WEBHOOK_URL)


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
