"""Web search helper using OpenAI *search-preview* models."""
from __future__ import annotations
from typing import Any, Dict, List
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_SEARCH_MODEL

def search_web(query: str) -> str:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [
            {"role": "system", "content": "Search the web and summarise briefly with sources."},
            {"role": "user", "content": query},
        ]
        resp = client.chat.completions.create(model=OPENAI_SEARCH_MODEL, messages=messages)
        return resp.choices[0].message.content or ""
    except Exception:
        return ""
