"""LLM client and chat utilities using the OpenAI Chat Completions API."""

from typing import Any, Dict, List

from openai import OpenAI

from config.config import OPENAI_API_KEY, OPENAI_MODEL


_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Lazily initialise and cache the OpenAI client.

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


def generate_chat_completion(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float | None = 0.3,
    max_tokens: int | None = 512,
) -> str:
    """Call the OpenAI Chat Completions API and return the assistant text.

    Args:
        messages: List of dict messages in the OpenAI chat format:
            [{"role": "system" | "user" | "assistant", "content": "..."}]
        model: Optional override for the model name. Defaults to OPENAI_MODEL.
        temperature: Sampling temperature. Some models may ignore this.
        max_tokens: Maximum tokens to generate. Some models may ignore this.

    Returns:
        The assistant's message content as a string.
    """
    client = get_openai_client()
    model_name = model or OPENAI_MODEL

    # Use only parameters broadly supported by Chat Completions
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def get_chatgroq_model():
    """Backwards-compatible wrapper.

    The original template expected a `get_chatgroq_model` function returning
    an object with an `.invoke(messages)` method.

    Here we return a lightweight adapter that forwards to `generate_chat_completion`.
    """

    class _OpenAIChatAdapter:
        def invoke(self, messages: List[Dict[str, Any]]) -> Any:  # type: ignore[override]
            """Adapter method to mimic a LangChain ChatModel interface."""
            content = generate_chat_completion(messages)

            class _Result:
                def __init__(self, text: str) -> None:
                    self.content = text

            return _Result(content)

    return _OpenAIChatAdapter()
