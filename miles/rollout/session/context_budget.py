"""Limit completion tokens without ever removing tokens from the prompt."""

from typing import Any


def fit_completion_budget(request: dict[str, Any], *, prompt_length: int, context_length: int | None) -> dict[str, Any]:
    if context_length is None:
        return request
    remaining = context_length - prompt_length
    if remaining <= 0:
        raise ValueError("Input already fills the configured context window")
    result = dict(request)
    key = "max_completion_tokens" if request.get("max_completion_tokens") is not None else "max_tokens"
    requested = request.get(key)
    result[key] = min(requested, remaining) if requested is not None else remaining
    return result
