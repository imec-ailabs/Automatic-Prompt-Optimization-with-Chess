"""Task-neutral construction of the supported DSPy optimizers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from chess_self_improvement.optimization.parameters import (
    BootstrapParams,
    BootstrapRandomParams,
    CoproParams,
    GepaParams,
    MiproV2Params,
    SimbaParams,
    resolve_optimizer_params,
)

Metric = Callable[..., Any]


@dataclass(frozen=True)
class OptimizerMetrics:
    """Provide task-specific scalar and textual-feedback metrics."""

    scalar: Metric
    simba: Metric
    gepa: Metric


def build_optimizer(
    name: str,
    task_lm: Any,
    prompt_lm: Any,
    metrics: OptimizerMetrics,
    *,
    log_dir: Path | None,
    max_metric_calls: int,
    seed: int,
    optimizer_params: dict[str, Any] | None = None,
    include_ground_truth_feedback: bool = False,
    default_num_threads: int | None = None,
    default_reflection_minibatch_size: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a DSPy optimizer from shared parameters and task metrics."""
    import dspy

    params = resolve_optimizer_params(name, optimizer_params)

    def configured_int(field: str, configured: int) -> int:
        if optimizer_params and field in optimizer_params:
            return configured
        return default_num_threads if default_num_threads is not None else configured

    if name == "simba":
        assert isinstance(params, SimbaParams)
        return dspy.SIMBA(
            metric=partial(
                metrics.simba,
                include_ground_truth=include_ground_truth_feedback,
            ),
            bsize=params.bsize,
            num_candidates=params.num_candidates,
            max_steps=params.max_steps,
            max_demos=params.max_demos,
            prompt_model=prompt_lm,
            num_threads=configured_int("num_threads", params.num_threads),
        ), {"seed": seed}
    if name == "copro":
        assert isinstance(params, CoproParams)
        return dspy.COPRO(
            prompt_model=prompt_lm,
            metric=metrics.scalar,
            breadth=params.breadth,
            depth=params.depth,
            track_stats=params.track_stats,
        ), {
            "eval_kwargs": {
                "num_threads": configured_int(
                    "eval_num_threads", params.eval_num_threads
                ),
                "display_progress": params.eval_display_progress,
                "display_table": params.eval_display_table,
            }
        }
    if name == "bootstrap_random_search":
        assert isinstance(params, BootstrapRandomParams)
        return dspy.BootstrapFewShotWithRandomSearch(
            metric=metrics.scalar,
            max_bootstrapped_demos=params.max_bootstrapped_demos,
            max_labeled_demos=params.max_labeled_demos,
            max_rounds=params.max_rounds,
            num_candidate_programs=params.num_candidate_programs,
            num_threads=configured_int("num_threads", params.num_threads),
            max_errors=params.max_errors,
        ), {"valset": None}
    if name == "gepa":
        assert isinstance(params, GepaParams)
        minibatch_size = params.reflection_minibatch_size
        if not (optimizer_params and "reflection_minibatch_size" in optimizer_params):
            minibatch_size = (
                default_reflection_minibatch_size
                if default_reflection_minibatch_size is not None
                else minibatch_size
            )
        kwargs: dict[str, Any] = {
            "metric": partial(
                metrics.gepa,
                include_ground_truth=include_ground_truth_feedback,
            ),
            "reflection_minibatch_size": minibatch_size,
            "candidate_selection_strategy": params.candidate_selection_strategy,
            "reflection_lm": prompt_lm,
            "skip_perfect_score": params.skip_perfect_score,
            "use_merge": params.use_merge,
            "num_threads": configured_int("num_threads", params.num_threads),
            "track_stats": params.track_stats,
            "seed": seed,
            "log_dir": str(log_dir) if log_dir else None,
        }
        if params.auto is not None:
            kwargs["auto"] = params.auto
        else:
            kwargs["max_metric_calls"] = max_metric_calls
        return dspy.GEPA(**kwargs), {"valset": None}
    if name == "mipro_v2":
        assert isinstance(params, MiproV2Params)
        kwargs = {
            "metric": metrics.scalar,
            "prompt_model": prompt_lm,
            "task_model": task_lm,
            "max_bootstrapped_demos": params.max_bootstrapped_demos,
            "max_labeled_demos": params.max_labeled_demos,
            "num_threads": configured_int("num_threads", params.num_threads),
            "max_errors": params.max_errors,
            "seed": seed,
            "verbose": params.verbose,
            "log_dir": str(log_dir) if log_dir else None,
        }
        compile_kwargs: dict[str, Any] = {"valset": None}
        if params.auto is not None:
            kwargs["auto"] = params.auto
        else:
            kwargs["auto"] = None
            kwargs["num_candidates"] = params.num_candidates
            compile_kwargs["num_trials"] = params.num_trials
        compile_kwargs.update(
            {
                "minibatch": params.minibatch,
                "program_aware_proposer": params.program_aware_proposer,
                "data_aware_proposer": params.data_aware_proposer,
                "tip_aware_proposer": params.tip_aware_proposer,
                "fewshot_aware_proposer": params.fewshot_aware_proposer,
                "requires_permission_to_run": params.requires_permission_to_run,
            }
        )
        return dspy.MIPROv2(**kwargs), compile_kwargs
    if name == "bootstrap_few_shot":
        assert isinstance(params, BootstrapParams)
        return dspy.BootstrapFewShot(
            metric=metrics.scalar,
            max_bootstrapped_demos=params.max_bootstrapped_demos,
            max_labeled_demos=params.max_labeled_demos,
            max_rounds=params.max_rounds,
            max_errors=params.max_errors,
        ), {}
    raise ValueError(f"Unknown optimizer: {name}")
