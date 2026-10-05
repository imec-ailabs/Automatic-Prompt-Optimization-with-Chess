"""Shared DSPy optimizer construction."""

from chess_self_improvement.optimization.factory import (
    OptimizerMetrics,
    build_optimizer,
)

__all__ = ["OptimizerMetrics", "build_optimizer"]
