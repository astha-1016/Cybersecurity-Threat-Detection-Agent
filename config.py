"""
config.py — Central configuration for the Cybersecurity Threat Detection Agent.
All environment variables and constants are loaded here.
Never hardcode keys anywhere else — always import from this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
ABUSEIPDB_API_KEY     = os.getenv("ABUSEIPDB_API_KEY", "")
VIRUSTOTAL_API_KEY    = os.getenv("VIRUSTOTAL_API_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────────
MODEL_NAME            = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
MODEL_TEMPERATURE     = float(os.getenv("MODEL_TEMPERATURE", "0"))

# ── Embeddings & RAG ─────────────────────────────────────────────────────────
EMBED_MODEL           = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DOCS_PATH             = os.getenv("DOCS_PATH", "data/docs/")
TOP_K_RETRIEVAL       = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# ── Eval thresholds ───────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
MAX_EVAL_RETRIES       = int(os.getenv("MAX_EVAL_RETRIES", "2"))

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_WINDOW         = int(os.getenv("MEMORY_WINDOW", "6"))
DB_PATH               = os.getenv("DB_PATH", "data/memory.db")

# ── Validation ────────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file: GROQ_API_KEY=your_key_here"
    )
