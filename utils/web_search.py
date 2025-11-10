"""Web search helper using OpenAI GPT-4o Search Preview models.

This module does *not* hit a third-party search API. Instead, it calls
OpenAI's specialised web-search model (e.g. `gpt-4o-search-preview` or
`gpt-4o-mini-search-preview`) via Chat Completions.

The model itself is trained to execute web searches and return answers
grounded in current web data.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from config.config import OPENAI_API_KEY, OPENAI_SEARCH_MODEL


_search_client: OpenAI | None = None


def _get_search_client() -> OpenAI:
    """Return a cached OpenAI client for web search.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured.
    """
    global _search_client
    if _search_client is not None:
        return _search_client

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please configure it in your environment."
        )

    _search_client = OpenAI(api_key=OPENAI_API_KEY)
    return _search_client


def search_web(query: str) -> str:
    """Execute a web search using a GPT-4o Search Preview model.

    The function sends the query to the specialised search model and
    returns the model's response text. This text is then used as context
    for the main assistant model.

    Args:
        query: User query or question.

    Returns:
        A string containing the search-based answer from the preview model.
        Returns an empty string if the call fails.
    """
    if not query.strip():
        return ""

    try:
        client = _get_search_client()

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a web search agent. Use live web data to answer the user query. "  # noqa: E501
                    "Return a short factual summary with key points and cite sources inline when possible."  # noqa: E501
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = client.chat.completions.create(
            model=OPENAI_SEARCH_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content or ""
    except Exception:
        # Fail softly and let the main model handle the query without web context.
        return ""
