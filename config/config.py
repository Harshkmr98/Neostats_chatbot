"""Central configuration for the AI_UseCase project.

All secrets (API keys) are read from environment variables or from a local
`.env` file so that they are not hard-coded in source control.
"""

import os
from pathlib import Path

try:
    # Optional dependency; if missing, .env loading is simply skipped
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# Try to load .env from project root: <project_root>/.env
# OS-level environment variables still take precedence.
if load_dotenv is not None:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        # Do not override existing OS env vars
        load_dotenv(dotenv_path=env_path, override=False)

# Core OpenAI configuration
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Default chat model for the assistant
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding model for RAG
OPENAI_EMBEDDING_MODEL: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# Dedicated web search model (GPT-4o Search Preview family)
# You can override this to "gpt-4o-mini-search-preview" via env if preferred.
OPENAI_SEARCH_MODEL: str = os.getenv(
    "OPENAI_SEARCH_MODEL",
    "gpt-4o-search-preview",
)

# RAG / retrieval configuration
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))  # characters per chunk

# Default system prompt used when the user does not provide one explicitly
DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    (
        "You are a helpful, domain-aware assistant. "
        "You answer using the provided knowledge base and web search results when relevant. "
        "If you are unsure or context is missing, you say so explicitly instead of guessing."
    ),
)
