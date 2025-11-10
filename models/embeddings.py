"""Embedding utilities used for Retrieval-Augmented Generation (RAG)."""

from typing import List

from openai import OpenAI

from config.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily initialise the OpenAI client for embeddings.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured.
    """
    global _client
    if _client is not None:
        return _client

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please configure it in your environment."
        )

    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts using the configured embedding model.

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    client = _get_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
