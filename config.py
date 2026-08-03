"""
Central configuration for the RAG chatbot.
All values can be overridden via environment variables (or a .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# IMPORTANT: load_dotenv() with no path only checks the current working
# directory. If Streamlit (or python) is launched from anywhere other than
# this project folder, the .env file would silently fail to load and
# GROQ_API_KEY would end up as "" -> Groq API then returns 401 Unauthorized.
# Pointing explicitly at the .env next to this file avoids that.
ENV_PATH = Path(__file__).resolve().parent / ".env"

# override=True is important: by default python-dotenv will NOT replace a
# variable that's already set in your shell/OS environment. If GROQ_API_KEY
# was ever set directly (e.g. testing another app, or a system env var on
# Windows), that stale value silently wins over your .env file otherwise --
# your .env edits get ignored and you keep getting 401s that look unrelated
# to anything you just changed.
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ---- API Keys ----
# .strip() guards against a trailing newline/space pasted into the .env file,
# which also produces a malformed Authorization header -> 401.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")


def validate_groq_key():
    """Raise a clear, actionable error instead of letting a blank/bad key
    surface later as a confusing 401 from the Groq API."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            f"GROQ_API_KEY is empty. Checked for a .env file at: {ENV_PATH}\n"
            "Fix: make sure a file literally named '.env' (not '.env.example') "
            "exists in this same folder, contains a line like "
            "GROQ_API_KEY=gsk_xxxxxxxx with no quotes, and that you copied the "
            "full key with no extra spaces."
        )
    if not GROQ_API_KEY.startswith("gsk_"):
        # Not fatal (Groq could change key formats), but worth a warning.
        print(
            f"Warning: GROQ_API_KEY does not start with 'gsk_' as expected "
            f"(starts with {GROQ_API_KEY[:4]!r}). Double-check you copied the "
            "correct key from console.groq.com."
        )

# ---- Models ----
# Check https://console.groq.com/docs/models for the current list of
# supported model IDs -- Groq frequently adds/retires models.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---- Storage paths ----
DATA_DIR = os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.pkl")
BM25_PATH = os.path.join(VECTORSTORE_DIR, "bm25.pkl")
LOG_FILE = os.getenv("LOG_FILE", "rag_app.log")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))      # approx words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))  # word overlap between chunks

# ---- Retrieval ----
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 10))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 10))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", 4))

# Cross-encoder scores are sigmoid-normalized to [0, 1] before comparing
# against this threshold. If the best chunk scores below this, the system
# falls back to web search instead of answering from documents.
# Tune this value based on your own data -- start around 0.3-0.5 and
# adjust after observing real query results.
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", 0.35))

# ---- Conversation ----
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 6))

# ---- Web fallback ----
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 5))