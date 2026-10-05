"""Strict parameter models for the supported DSPy optimizers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizerParams(BaseModel):
    """Base for every per-optimizer parameter block (rejects extra keys).

    Defaults are the values the paper ran (Appendix B.6, Table 11), which are
    DSPy's own defaults with ``auto: heavy`` for GEPA and MIPROv2. Omitting the
    ``optimizer`` block therefore runs an optimizer at paper scale; the bounded
    example configs override these explicitly.
    """

    model_config = ConfigDict(extra="forbid")


class SimbaParams(OptimizerParams):
    bsize: int = Field(default=32, ge=1)
    num_candidates: int = Field(default=6, ge=1)
    max_steps: int = Field(default=8, ge=1)
    max_demos: int = Field(default=4, ge=0)
    num_threads: int = Field(default=16, ge=1)


class CoproParams(OptimizerParams):
    breadth: int = Field(default=10, ge=1)
    depth: int = Field(default=3, ge=1)
    track_stats: bool = True
    eval_num_threads: int = Field(default=16, ge=1)
    eval_display_progress: bool = False
    eval_display_table: bool = False


class BootstrapRandomParams(OptimizerParams):
    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=16, ge=0)
    max_rounds: int = Field(default=1, ge=1)
    num_candidate_programs: int = Field(default=16, ge=1)
    num_threads: int = Field(default=16, ge=1)
    max_errors: int = Field(default=5, ge=0)


class GepaParams(OptimizerParams):
    # ``auto: heavy`` resolved to ~3,436 metric calls in the paper. Set
    # ``auto: null`` to use the run's ``max_metric_calls`` instead.
    auto: Literal["light", "medium", "heavy"] | None = "heavy"
    reflection_minibatch_size: int = Field(default=3, ge=1)
    candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto"
    skip_perfect_score: bool = True
    use_merge: bool = True
    num_threads: int = Field(default=16, ge=1)
    track_stats: bool = True


class MiproV2Params(OptimizerParams):
    # ``auto: heavy`` resolved to ~18 candidates and ~27 trials in the paper.
    auto: Literal["light", "medium", "heavy"] | None = "heavy"
    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=4, ge=0)
    num_candidates: int | None = Field(default=None, ge=1)
    num_threads: int = Field(default=16, ge=1)
    max_errors: int = Field(default=5, ge=0)
    verbose: bool = False
    num_trials: int | None = Field(default=None, ge=1)
    minibatch: bool = False
    program_aware_proposer: bool = True
    data_aware_proposer: bool = True
    tip_aware_proposer: bool = True
    fewshot_aware_proposer: bool = True
    requires_permission_to_run: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_auto_with_manual_budgets(cls, data: Any) -> Any:
        """Fail loudly when ``auto`` is combined with explicit manual budgets.

        DSPy's ``auto`` mode resolves ``num_candidates`` and ``num_trials``
        itself, so persisting manual budgets alongside ``auto`` silently
        discards them. Rather than let a config quietly unset ``auto``'s
        resolved budgets, reject the conflicting combination so an ``auto=heavy``
        run cannot be silently downgraded. Callers that want manual budgets must
        omit ``auto`` (or set it to ``null``).
        """
        if not isinstance(data, dict):
            return data
        auto = data.get("auto")
        if auto is None:
            return data
        conflicting = [
            field
            for field in ("num_candidates", "num_trials")
            if data.get(field) is not None
        ]
        if conflicting:
            joined = ", ".join(sorted(conflicting))
            raise ValueError(
                f"MIPROv2 auto={auto!r} resolves the trial budget itself; "
                f"remove the manual {joined} (or set auto=null) so auto is not "
                "silently unset."
            )
        return data

    @model_validator(mode="after")
    def clear_manual_auto_budgets(self) -> MiproV2Params:
        """Drop the *defaulted* manual budgets DSPy auto mode does not use."""
        if self.auto is not None:
            self.num_candidates = None
            self.num_trials = None
        return self


class BootstrapParams(OptimizerParams):
    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=16, ge=0)
    max_rounds: int = Field(default=1, ge=1)
    max_errors: int = Field(default=5, ge=0)


_PARAM_MODELS: dict[str, type[OptimizerParams]] = {
    "simba": SimbaParams,
    "copro": CoproParams,
    "bootstrap_random_search": BootstrapRandomParams,
    "gepa": GepaParams,
    "mipro_v2": MiproV2Params,
    "bootstrap_few_shot": BootstrapParams,
}


def optimizer_param_model(name: str) -> type[OptimizerParams]:
    """Return the shared parameter model for an optimizer name."""
    try:
        return _PARAM_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer: {name}") from exc


def resolve_optimizer_params(
    name: str, params: dict[str, Any] | None = None
) -> OptimizerParams:
    """Validate and fill the shared optimizer parameters for one optimizer.

    Raises ``ValueError`` for an unknown optimizer and ``ValidationError`` for
    unknown, misspelled, or cross-optimizer fields.
    """
    return optimizer_param_model(name).model_validate(params or {})
