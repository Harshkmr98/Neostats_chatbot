"""LLM client and chat utilities using the OpenAI Chat Completions API."""
from __future__ import annotations
from typing import Any, Dict, List
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_MODEL

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def generate_chat_completion(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
