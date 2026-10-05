"""Single-process runner for the independent-position benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

import chess
import dspy
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chess_self_improvement.benchmarks.independent_positions import (
    exact_move_metric,
    gepa_move_metric,
    simba_move_metric,
)
from chess_self_improvement.datasets import PuzzleFilter, iter_lichess_puzzles
from chess_self_improvement.dspy_program import ChessMoveProgram, puzzle_task_examples
from chess_self_improvement.dspy_usage import lm_usage_fields
from chess_self_improvement.jev import JEV_MODEL, JevAdapter, JevLM, JevTransport
from chess_self_improvement.moves import resolve_move
from chess_self_improvement.optimization import OptimizerMetrics, build_optimizer
from chess_self_improvement.optimization.parameters import resolve_optimizer_params

BenchmarkName = Literal["independent_positions"]
MethodName = Literal[
    "baseline",
    "bootstrap_few_shot",
    "bootstrap_random_search",
    "copro",
    "gepa",
    "mipro_v2",
    "simba",
]


class ModelSettings(BaseModel):
    """Configure one DSPy language model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int | None = Field(default=1024, ge=1)
    retries: int = Field(default=3, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_jev(self) -> ModelSettings:
        """Reject generative settings that Decisions cannot apply."""
        if self.name == JEV_MODEL:
            if self.temperature != 0:
                raise ValueError("Jev requires temperature=0")
            if self.extra:
                raise ValueError("Jev does not support generative extra settings")
            if "max_tokens" in self.model_fields_set and self.max_tokens is not None:
                raise ValueError("Jev does not support max_tokens; omit it or use null")
            self.max_tokens = None
        return self


class RunConfig(BaseModel):
    """Configure one local baseline or prompt-optimization run."""

    model_config = ConfigDict(extra="forbid")

    benchmark: BenchmarkName
    method: MethodName = "baseline"
    train_path: Path = Path("data/puzzle_train.csv")
    test_path: Path = Path("data/puzzle_test.csv")
    output_dir: Path = Path("runs/example")
    task_model: ModelSettings
    prompt_model: ModelSettings | None = None
    seed: int = Field(default=42, ge=0)
    train_limit: int = Field(default=0, ge=0)
    validation_size: int = Field(default=140, ge=1)
    test_limit: int = Field(default=0, ge=0)
    max_metric_calls: int = Field(default=2000, ge=1)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    use_analysis: bool = False
    include_ground_truth_feedback: bool = False
    continue_on_error: bool = False

    @model_validator(mode="after")
    def validate_prompt_model(self) -> RunConfig:
        """Require a proposal model only when compilation needs one."""
        if self.prompt_model is not None and self.prompt_model.name == JEV_MODEL:
            raise ValueError("prompt_model must be a generative model, not Jev")
        if self.task_model.name == JEV_MODEL:
            if self.benchmark != "independent_positions" or self.use_analysis:
                raise ValueError(
                    "Jev requires independent_positions with use_analysis=false"
                )
            if self.method == "simba":
                raise ValueError("Jev does not support SIMBA sampling")
            if self.method != "baseline" and self.prompt_model is None:
                raise ValueError(
                    "Jev optimization requires an explicit generative prompt_model"
                )
            if self.method in {"bootstrap_few_shot", "bootstrap_random_search"}:
                if self.optimizer.get("max_rounds", 1) != 1:
                    raise ValueError("Jev bootstrapping requires max_rounds=1")
                self.optimizer = {**self.optimizer, "max_rounds": 1}
            if self.method != "baseline":
                resolve_optimizer_params(self.method, self.optimizer)
        if self.method != "baseline" and self.prompt_model is None:
            self.prompt_model = self.task_model.model_copy(update={"temperature": 1.0})
        return self


def load_config(path: Path) -> RunConfig:
    """Load a strict YAML run configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def run(config: RunConfig) -> dict[str, Any]:
    """Compile when requested, evaluate locally, and write reproducible artifacts."""
    # Revalidate before filesystem or network I/O, including mutated model instances.
    config = RunConfig.model_validate(config.model_dump())
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainset, valset, testset = _load_splits(config)
    task_lm = _build_lm(config.task_model)
    prompt_lm = _build_lm(config.prompt_model or config.task_model)
    program, metrics = _benchmark(config.benchmark, config.use_analysis)
    adapter = (
        JevAdapter(use_json_adapter_fallback=False)
        if isinstance(task_lm, JevLM)
        else dspy.JSONAdapter()
    )
    if isinstance(task_lm, JevLM):
        program.set_lm(task_lm)

    compile_seconds = 0.0
    compile_usage: dict[str, Any] | None = None
    if config.method != "baseline":
        optimizer, compile_kwargs = build_optimizer(
            config.method,
            task_lm,
            prompt_lm,
            metrics,
            log_dir=config.output_dir / "optimizer_logs",
            max_metric_calls=config.max_metric_calls,
            seed=config.seed,
            optimizer_params=config.optimizer,
            include_ground_truth_feedback=config.include_ground_truth_feedback,
        )
        if "valset" in compile_kwargs:
            compile_kwargs["valset"] = valset
        started = time.perf_counter()
        with dspy.context(lm=task_lm, adapter=adapter, track_usage=True):
            program = optimizer.compile(program, trainset=trainset, **compile_kwargs)
        compile_seconds = time.perf_counter() - started
        if isinstance(task_lm, JevLM):
            # DSPy restores saved LMs as generative LMs; persist only the program.
            program.set_lm(None)
            try:
                program.save(str(config.output_dir / "compiled.json"))
            finally:
                program.set_lm(task_lm)
        else:
            program.save(str(config.output_dir / "compiled.json"))
        compile_usage = {
            "task_model": _drain_usage(task_lm),
            "prompt_model": _drain_usage(prompt_lm),
        }

    results = _evaluate(
        config.benchmark,
        program,
        testset,
        task_lm,
        continue_on_error=config.continue_on_error,
    )
    summary = {
        "benchmark": config.benchmark,
        "method": config.method,
        "train_examples": len(trainset),
        "validation_examples": len(valset),
        "test_examples": len(testset),
        "compile_seconds": compile_seconds,
        "metrics": _aggregate(config.benchmark, results),
        "compile_usage": compile_usage,
        "evaluation_usage": _drain_usage(task_lm),
    }
    provenance = {
        "config": config.model_dump(mode="json"),
        "datasets": {
            "train_sha256": _sha256(config.train_path),
            "test_sha256": _sha256(config.test_path),
        },
        "dspy_version": getattr(dspy, "__version__", None),
    }
    _write_json(config.output_dir / "results.json", results)
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "run_config.json", provenance)
    return summary


def _drain_usage(lm: Any) -> dict[str, Any]:
    """Summarize then clear ``lm.history`` so each phase is counted from zero.

    DSPy keeps at most ``dspy.settings.max_history_size`` entries per LM and
    silently drops the oldest beyond that, so slicing the history by index
    would under-count long compilations. Draining per phase keeps every phase
    under the cap and flags the rare case where a single phase still exceeds it.
    """
    history = list(lm.history)
    lm.history.clear()
    usage = lm_usage_fields(history)
    cap = dspy.settings.max_history_size
    if cap and len(history) >= cap:
        usage["usage_complete"] = False
        usage["history_truncated"] = True
    if isinstance(lm, JevLM):
        usage.update(
            available=False,
            num_calls=None,
            calls_with_usage=None,
            reason="Jev usage is not recorded by this adapter",
        )
    return usage


def _build_lm(settings: ModelSettings) -> Any:
    if settings.name == JEV_MODEL:
        return JevLM(
            request=JevTransport(
                timeout_seconds=settings.timeout_seconds or 30,
                retries=settings.retries,
            )
        )
    kwargs = dict(settings.extra)
    if settings.timeout_seconds is not None:
        kwargs["timeout"] = settings.timeout_seconds
    return dspy.LM(
        settings.name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        num_retries=settings.retries,
        cache=False,
        **kwargs,
    )


def _benchmark(name: BenchmarkName, use_analysis: bool) -> tuple[Any, OptimizerMetrics]:
    return ChessMoveProgram(use_analysis=use_analysis), OptimizerMetrics(
        scalar=exact_move_metric,
        simba=simba_move_metric,
        gepa=gepa_move_metric,
    )


def _load_splits(config: RunConfig) -> tuple[list[Any], list[Any], list[Any]]:
    if config.method == "baseline":
        return (
            [],
            [],
            _load_examples(config.benchmark, config.test_path, config.seed)[
                : config.test_limit or None
            ],
        )
    tasks = list(
        iter_lichess_puzzles(config.train_path, PuzzleFilter(seed=config.seed))
    )
    random.Random(config.seed).shuffle(tasks)
    if config.validation_size >= len(tasks):
        raise ValueError("validation_size must be smaller than the training set")
    validation_tasks = tasks[: config.validation_size]
    training_tasks = tasks[config.validation_size :]
    if config.train_limit:
        training_tasks = training_tasks[: config.train_limit]
    trainset = [
        example for task in training_tasks for example in puzzle_task_examples(task)
    ]
    valset = [
        example for task in validation_tasks for example in puzzle_task_examples(task)
    ]
    test_tasks = list(
        iter_lichess_puzzles(config.test_path, PuzzleFilter(seed=config.seed))
    )
    testset = [example for task in test_tasks for example in puzzle_task_examples(task)]
    if config.test_limit:
        testset = testset[: config.test_limit]
    return trainset, valset, testset


def _load_examples(name: BenchmarkName, path: Path, seed: int) -> list[Any]:
    """Load examples for one benchmark from a Lichess-format CSV."""
    tasks = list(iter_lichess_puzzles(path, PuzzleFilter(seed=seed)))
    return [example for task in tasks for example in puzzle_task_examples(task)]


def _evaluate(
    name: BenchmarkName,
    program: Any,
    examples: list[Any],
    lm: Any,
    *,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    adapter = (
        JevAdapter(use_json_adapter_fallback=False)
        if isinstance(lm, JevLM)
        else dspy.JSONAdapter()
    )
    with dspy.context(lm=lm, adapter=adapter, track_usage=True):
        for example in examples:
            started = time.perf_counter()
            try:
                prediction = program(**example.inputs().toDict())
                result = _score(name, example, prediction)
                result["error"] = None
            except Exception as exc:
                if not continue_on_error:
                    raise
                result = _failed_result(name)
                result["error"] = f"{type(exc).__name__}: {exc}"
            # Field names follow the published observation schema in
            # results/paper/ so local runs can be joined with the paper's data.
            result.update(
                {
                    "source_id": getattr(example, "puzzle_id", None),
                    "rating": getattr(example, "rating", None),
                    "move_index": getattr(example, "position_index", None),
                    "solver_length": getattr(example, "position_count", None),
                    "position_fen": str(example.position_fen),
                    "latency_s": time.perf_counter() - started,
                }
            )
            results.append(result)
    return results


def _score(name: BenchmarkName, example: Any, prediction: Any) -> dict[str, Any]:
    raw = str(getattr(prediction, "move", "") or "")
    resolution = resolve_move(raw, chess.Board(str(example.position_fen)))
    return {
        "prediction": raw,
        "expected": str(example.move),
        "resolved": resolution.move_uci,
        "correct": resolution.move_uci == str(example.move),
    }


def _failed_result(name: BenchmarkName) -> dict[str, Any]:
    return {"prediction": "", "expected": None, "resolved": None, "correct": False}


def _aggregate(name: BenchmarkName, results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if not count:
        return {}
    # The paper's Table 3 scores a source record (one Lichess puzzle) as correct
    # only when every one of its solver positions is correct. That is only
    # meaningful for records whose positions were all evaluated, which
    # ``test_limit`` can cut short.
    by_record: dict[Any, list[dict[str, Any]]] = {}
    for result in results:
        by_record.setdefault(result.get("source_id"), []).append(result)
    complete = [
        positions
        for positions in by_record.values()
        if positions[0].get("solver_length") is not None
        and {position.get("move_index") for position in positions}
        == set(range(positions[0]["solver_length"]))
    ]
    record_scores = [
        all(position["correct"] for position in positions) for positions in complete
    ]
    return {
        "accuracy": sum(result["correct"] for result in results) / count,
        "source_records": len(by_record),
        "source_records_complete": len(complete),
        "source_record_accuracy": (
            sum(record_scores) / len(record_scores) if record_scores else None
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
