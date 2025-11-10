"""Central configuration for the AI_UseCase project.

Secrets are read from environment variables or a local .env file.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore

# Load .env if present
if load_dotenv:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)  # type: ignore

# OpenAI
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_SEARCH_MODEL: str = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini-search-preview")

# RAG
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a helpful assistant. Use uploaded docs and cite filenames when useful.",
)
