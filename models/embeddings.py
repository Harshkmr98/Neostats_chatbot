"""Embedding utilities used for Retrieval-Augmented Generation (RAG)."""
from __future__ import annotations
from typing import List
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def embed_texts(texts: List[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client()
    res = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in res.data]
