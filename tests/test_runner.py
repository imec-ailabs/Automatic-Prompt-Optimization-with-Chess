"""Offline tests for strict local run configuration and optimizer setup."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import dspy
import pytest
from pydantic import ValidationError

from chess_self_improvement.jev import JEV_MODEL, JevLM, JevTransport
from chess_self_improvement.optimization import OptimizerMetrics, build_optimizer
from chess_self_improvement.runner import (
    ModelSettings,
    RunConfig,
    _aggregate,
    _build_lm,
    _drain_usage,
    _load_splits,
)


def metric(*_: object, **__: object) -> float:
    return 0.0


@pytest.mark.parametrize(
    "name",
    [
        "bootstrap_few_shot",
        "bootstrap_random_search",
        "copro",
        "gepa",
        "mipro_v2",
        "simba",
    ],
)
def test_all_paper_optimizers_build(tmp_path: Path, name: str) -> None:
    metrics = OptimizerMetrics(metric, metric, metric)

    optimizer, _ = build_optimizer(
        name,
        SimpleNamespace(),
        SimpleNamespace(),
        metrics,
        log_dir=tmp_path,
        max_metric_calls=10,
        seed=42,
    )

    assert optimizer is not None


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "benchmark": "independent_positions",
                "task_model": {"name": "test"},
                "unknown_setting": True,
            }
        )


@pytest.mark.parametrize("benchmark", ["legal_moves", "next_fen"])
def test_config_rejects_removed_benchmarks(benchmark: str) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {"benchmark": benchmark, "task_model": {"name": "test"}}
        )


def test_baseline_does_not_require_training_data(tmp_path: Path) -> None:
    test_path = tmp_path / "test.csv"
    test_path.write_text(
        "PuzzleId,FEN,Moves,Rating,Themes\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,"
        "e2e4 e7e5,1000,opening\n",
        encoding="utf-8",
    )
    config = RunConfig(
        benchmark="independent_positions",
        method="baseline",
        train_path=tmp_path / "missing.csv",
        test_path=test_path,
        task_model=ModelSettings(name="test"),
    )

    trainset, valset, testset = _load_splits(config)

    assert not trainset
    assert not valset
    assert len(testset) == 1


def test_paper_splits_do_not_overlap() -> None:
    root = Path(__file__).parents[1]

    def identities(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
        import csv

        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        return (
            {row["PuzzleId"] for row in rows},
            {(row["FEN"], row["Moves"]) for row in rows},
        )

    train_ids, train_lines = identities(root / "data/puzzle_train.csv")
    test_ids, test_lines = identities(root / "data/puzzle_test.csv")

    assert train_ids.isdisjoint(test_ids)
    assert train_lines.isdisjoint(test_lines)


@pytest.mark.parametrize("name", [JEV_MODEL, "arbitrary/provider-model"])
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_model_timeout_must_be_positive_finite(name: str, timeout: float) -> None:
    with pytest.raises(ValidationError):
        ModelSettings(name=name, timeout_seconds=timeout)


def test_jev_model_settings_and_transport() -> None:
    settings = ModelSettings(name=JEV_MODEL, timeout_seconds=17, retries=1)
    assert settings.max_tokens is None
    assert ModelSettings.model_validate(settings.model_dump()).max_tokens is None
    lm = _build_lm(settings)
    assert isinstance(lm, JevLM)
    assert isinstance(lm.request, JevTransport)
    assert lm.request.timeout_seconds == 17
    assert lm.request.retries == 1
    assert lm.kwargs["max_tokens"] is None


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": 0.1},
        {"max_tokens": 1024},
        {"extra": {"top_p": 1}},
        {"extra": {"api_key": "SECRET"}},
    ],
)
def test_jev_rejects_generative_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelSettings(name=JEV_MODEL, **settings)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"benchmark": "next_fen"}, "independent_positions"),
        ({"benchmark": "legal_moves"}, "independent_positions"),
        ({"use_analysis": True}, "use_analysis=false"),
        ({"method": "simba"}, "SIMBA"),
        ({"method": "gepa"}, "explicit generative prompt_model"),
        ({"prompt_model": {"name": JEV_MODEL}}, "generative model"),
        (
            {
                "method": "bootstrap_few_shot",
                "optimizer": {"max_rounds": 2},
                "prompt_model": {"name": "prompt"},
            },
            "max_rounds=1",
        ),
        (
            {
                "method": "bootstrap_random_search",
                "optimizer": {"max_rounds": 2},
                "prompt_model": {"name": "prompt"},
            },
            "max_rounds=1",
        ),
        (
            {
                "method": "mipro_v2",
                "optimizer": {"max_rounds": 2},
                "prompt_model": {"name": "prompt"},
            },
            "Extra inputs",
        ),
    ],
)
def test_jev_invalid_run_config(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        RunConfig.model_validate(
            {
                "benchmark": "independent_positions",
                "task_model": {"name": JEV_MODEL},
                **overrides,
            }
        )


@pytest.mark.parametrize(
    "method",
    [
        "baseline",
        "bootstrap_few_shot",
        "bootstrap_random_search",
        "copro",
        "gepa",
        "mipro_v2",
    ],
)
def test_jev_compatible_methods_config_and_build(method: str, tmp_path: Path) -> None:
    config = RunConfig.model_validate(
        {
            "benchmark": "independent_positions",
            "method": method,
            "task_model": {"name": JEV_MODEL},
            "prompt_model": {
                "name": "unrestricted/prompt",
                "temperature": 1.2,
                "extra": {"drop_params": True},
            },
        }
    )
    if method == "baseline":
        return
    optimizer, _ = build_optimizer(
        method,
        JevLM(request=Mock()),
        Mock(),
        OptimizerMetrics(metric, metric, metric),
        log_dir=tmp_path,
        max_metric_calls=10,
        seed=42,
        optimizer_params=config.optimizer,
    )
    if method.startswith("bootstrap"):
        assert optimizer.max_rounds == 1
        assert config.optimizer["max_rounds"] == 1


@pytest.mark.parametrize("name", ["custom/arbitrary-task", "custom/arbitrary-prompt"])
def test_generative_kwargs_forwarded(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    constructor = Mock()
    monkeypatch.setattr(dspy, "LM", constructor)
    settings = ModelSettings(
        name=name,
        temperature=1.3,
        max_tokens=8192,
        retries=2,
        timeout_seconds=45,
        extra={"drop_params": True, "top_p": 0.9, "seed": 42},
    )
    assert _build_lm(settings) is constructor.return_value
    constructor.assert_called_once_with(
        name,
        temperature=1.3,
        max_tokens=8192,
        num_retries=2,
        cache=False,
        timeout=45,
        drop_params=True,
        top_p=0.9,
        seed=42,
    )


def test_generative_config_is_not_restricted() -> None:
    config = RunConfig(
        benchmark="independent_positions",
        method="simba",
        use_analysis=True,
        task_model=ModelSettings(name="any-model", extra={"top_p": 0.9}),
    )
    assert config.prompt_model is not None
    assert config.prompt_model.name == "any-model"
    assert config.prompt_model.temperature == 1
    assert config.prompt_model.extra == {"top_p": 0.9}


def test_generative_unset_timeout_preserves_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Mock()
    monkeypatch.setattr(dspy, "LM", constructor)
    _build_lm(ModelSettings(name="any-model", extra={"timeout": 13}))
    assert constructor.call_args.kwargs["timeout"] == 13


def _position(source_id: str, index: int, count: int, correct: bool) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "move_index": index,
        "solver_length": count,
        "correct": correct,
    }


def test_aggregate_scores_source_records_only_when_complete() -> None:
    results = [
        # Fully evaluated two-position record with one miss: record incorrect.
        _position("a", 0, 2, True),
        _position("a", 1, 2, False),
        # Fully evaluated one-position record: record correct.
        _position("b", 0, 1, True),
        # Truncated three-position record: excluded from the record-level metric.
        _position("c", 0, 3, True),
        _position("c", 1, 3, True),
    ]

    metrics = _aggregate("independent_positions", results)

    assert metrics["accuracy"] == pytest.approx(4 / 5)
    assert metrics["source_records"] == 3
    assert metrics["source_records_complete"] == 2
    assert metrics["source_record_accuracy"] == pytest.approx(1 / 2)


def test_aggregate_without_position_metadata_reports_no_record_accuracy() -> None:
    metrics = _aggregate("independent_positions", [{"source_id": "a", "correct": True}])

    assert metrics["accuracy"] == 1.0
    assert metrics["source_records_complete"] == 0
    assert metrics["source_record_accuracy"] is None


def test_puzzle_examples_carry_position_count(tmp_path: Path) -> None:
    from chess_self_improvement.datasets import PuzzleFilter, iter_lichess_puzzles
    from chess_self_improvement.dspy_program import puzzle_task_examples

    path = tmp_path / "puzzles.csv"
    path.write_text(
        "PuzzleId,FEN,Moves,Rating,Themes\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,"
        "e2e4 e7e5 g1f3 b8c6,1000,opening\n",
        encoding="utf-8",
    )
    examples = puzzle_task_examples(next(iter_lichess_puzzles(path, PuzzleFilter())))

    assert [example.position_index for example in examples] == [0, 1]
    assert {example.position_count for example in examples} == {2}


def _usage_entry(prompt: int, completion: int, cost: float) -> dict[str, Any]:
    return {
        "response": {
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            "_hidden_params": {"response_cost": cost},
        }
    }


def test_drain_usage_counts_then_clears_history() -> None:
    history = [_usage_entry(10, 2, 0.001), _usage_entry(5, 1, 0.002)]
    lm = SimpleNamespace(history=history)

    usage = _drain_usage(lm)

    assert usage["num_calls"] == 2
    assert usage["total_tokens"] == 18
    assert usage["cost_usd"] == pytest.approx(0.003)
    assert usage["usage_complete"] is True
    assert "history_truncated" not in usage
    assert lm.history == []


def test_drain_usage_flags_dspy_history_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dspy.settings, "max_history_size", 2)
    lm = SimpleNamespace(history=[_usage_entry(1, 1, 0.0), _usage_entry(1, 1, 0.0)])

    usage = _drain_usage(lm)

    assert usage["history_truncated"] is True
    assert usage["usage_complete"] is False
