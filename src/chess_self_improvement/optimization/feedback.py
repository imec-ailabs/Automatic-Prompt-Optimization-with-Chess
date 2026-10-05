"""Shared adapters for optimizer-specific textual feedback support."""

from __future__ import annotations

from typing import Any


def simba_feedback(score: float, feedback: str) -> Any:
    """Return SIMBA's structured reward while keeping DSPy import optional."""
    import dspy

    return dspy.Prediction(score=score, feedback=feedback)
