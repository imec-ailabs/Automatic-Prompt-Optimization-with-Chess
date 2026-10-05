"""Normalize token and cost metadata from DSPy language-model history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


class LMUsageSummary(BaseModel):
    """Summarize one DSPy LM's history independently."""

    calls: int = 0
    calls_with_usage: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    complete: bool = False


def summarize_lm_history(history: Sequence[dict[str, Any]]) -> LMUsageSummary:
    """Normalize DSPy/LiteLLM usage variants from LM history."""
    prompt_tokens = completion_tokens = 0
    cost_usd = 0.0
    calls_with_usage = 0
    prompt_calls = completion_calls = cost_calls = 0
    for entry in history:
        response = entry.get("response")
        usage = _field(response, "usage", {}) or entry.get("usage", {}) or {}
        prompt = _field(usage, "prompt_tokens")
        completion = _field(usage, "completion_tokens")
        cost = _field(usage, "cost")
        if cost is None:
            cost = entry.get("cost")
        if cost is None:
            cost = entry.get("response_cost")
        if cost is None:
            hidden = _field(response, "_hidden_params", {}) or {}
            cost = _field(hidden, "response_cost")
        if prompt is None and completion is None and cost is None:
            continue
        calls_with_usage += 1
        prompt_calls += prompt is not None
        completion_calls += completion is not None
        cost_calls += cost is not None
        prompt_tokens += int(prompt or 0)
        completion_tokens += int(completion or 0)
        cost_usd += float(cost or 0.0)
    return LMUsageSummary(
        calls=len(history),
        calls_with_usage=calls_with_usage,
        prompt_tokens=(
            prompt_tokens if history and prompt_calls == len(history) else None
        ),
        completion_tokens=(
            completion_tokens if history and completion_calls == len(history) else None
        ),
        cost_usd=cost_usd if history and cost_calls == len(history) else None,
        complete=bool(history)
        and prompt_calls == completion_calls == cost_calls == len(history),
    )


def lm_usage_fields(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return JSON-ready usage fields while preserving missing metadata."""
    usage = summarize_lm_history(history)
    total_tokens = (
        usage.prompt_tokens + usage.completion_tokens
        if usage.prompt_tokens is not None and usage.completion_tokens is not None
        else None
    )
    return {
        "num_calls": usage.calls,
        "calls_with_usage": usage.calls_with_usage,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(usage.cost_usd, 9) if usage.cost_usd is not None else None,
        "usage_complete": usage.complete,
    }


def _field(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from mapping-like or attribute-like values."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
