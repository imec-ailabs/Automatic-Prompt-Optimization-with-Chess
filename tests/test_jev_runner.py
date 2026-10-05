"""Run the local Jev compile/save/evaluate path without provider calls."""

import io
import json
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import dspy
import litellm
import pytest
from dspy.utils import DummyLM

from chess_self_improvement import runner
from chess_self_improvement.dspy_program import ChessMoveProgram
from chess_self_improvement.jev import JEV_MODEL, JevAdapter, JevLM
from chess_self_improvement.runner import ModelSettings, RunConfig

INPUTS = {
    "position_fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
    "side_to_move": "white",
    "legal_moves_uci": "e2e3, e2d3",
}


@pytest.fixture(autouse=True)
def no_paid_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = Mock(side_effect=AssertionError("No paid calls in tests"))
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(litellm, "completion", blocked)
    monkeypatch.setattr(litellm, "acompletion", blocked)


@pytest.fixture
def offline_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[RunConfig, list[dict[str, Any]]]:
    example = dspy.Example(**INPUTS, move="e2e3", puzzle_id="test").with_inputs(*INPUTS)
    monkeypatch.setattr(
        runner, "_load_splits", lambda _: ([example], [example], [example])
    )
    monkeypatch.setattr(runner, "_sha256", lambda _: "offline-hash")
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET_NEVER_PERSIST")
    payloads = []

    def http(request: Any, *, timeout: float) -> io.BytesIO:
        assert timeout == 9
        payload = json.loads(request.data)
        payloads.append(payload)
        return io.BytesIO(
            json.dumps(
                {
                    "model": JEV_MODEL,
                    "answers": {
                        "move": {
                            "type": "choice",
                            "choice": "e2e3",
                            "probabilities": {"e2e3": 0.7, "e2d3": 0.3},
                        }
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", http)
    config = RunConfig(
        benchmark="independent_positions",
        output_dir=tmp_path / "run",
        task_model=ModelSettings(name=JEV_MODEL, timeout_seconds=9, retries=0),
    )
    return config, payloads


def test_jev_baseline(offline_run: tuple[RunConfig, list[dict[str, Any]]]) -> None:
    config, payloads = offline_run
    summary = runner.run(config)
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["compile_usage"] is None
    assert len(payloads) == 1
    assert payloads[0]["state"] == INPUTS
    usage = summary["evaluation_usage"]
    assert usage["available"] is False
    assert usage["num_calls"] is None
    assert usage["cost_usd"] is None
    assert usage["total_tokens"] is None
    assert usage["usage_complete"] is False
    persisted = json.loads((config.output_dir / "run_config.json").read_text())
    assert persisted["config"]["task_model"]["max_tokens"] is None
    assert not (config.output_dir / "compiled.json").exists()
    for path in config.output_dir.iterdir():
        assert "SECRET_NEVER_PERSIST" not in path.read_text()


@pytest.mark.parametrize(
    "method",
    ["bootstrap_few_shot", "bootstrap_random_search", "copro", "gepa", "mipro_v2"],
)
def test_jev_compile_save_reload_and_correct_eval_lm(
    offline_run: tuple[RunConfig, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    config, payloads = offline_run
    config = RunConfig.model_validate(
        {
            **config.model_dump(),
            "method": method,
            "prompt_model": {"name": "arbitrary/generative", "temperature": 1.0},
        }
    )
    prompt_lm = DummyLM([{"proposal": "Prefer activity."}])
    monkeypatch.setattr(dspy, "LM", Mock(return_value=prompt_lm))

    def factory(
        name: str, task: Any, prompt: Any, *_: Any, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        assert name == method
        assert isinstance(task, JevLM)
        assert prompt is prompt_lm
        if method.startswith("bootstrap"):
            assert kwargs["optimizer_params"]["max_rounds"] == 1

        def compile_program(program: Any, *, trainset: Any, valset: Any) -> Any:
            assert isinstance(dspy.settings.adapter, JevAdapter)
            assert program.get_lm() is task
            with dspy.context(lm=prompt):
                proposal = dspy.Predict("task -> proposal")(task="Improve").proposal
            program.choose_move.signature = (
                program.choose_move.signature.with_instructions(proposal)
            )
            program.choose_move.demos = trainset
            assert program(**INPUTS).move == "e2e3"
            # Optimizers can return a deep-copied, LM-bound candidate.
            return program.deepcopy()

        return Mock(compile=compile_program), {"valset": None}

    monkeypatch.setattr(runner, "build_optimizer", factory)
    evaluate = runner._evaluate

    def evaluate_bound(
        name: Any, program: Any, examples: Any, lm: Any, **kwargs: Any
    ) -> Any:
        assert program.get_lm() is lm
        return evaluate(name, program, examples, lm, **kwargs)

    monkeypatch.setattr(runner, "_evaluate", evaluate_bound)
    summary = runner.run(config)
    assert summary["metrics"]["accuracy"] == 1
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert "Prefer activity." in payloads[-1]["questions"]["move"]["instructions"]
    assert "Labeled example 1:" in payloads[-1]["questions"]["move"]["instructions"]
    compiled_path = config.output_dir / "compiled.json"
    saved = json.loads(compiled_path.read_text())
    assert saved["choose_move"]["lm"] is None
    for path in config.output_dir.iterdir():
        assert "SECRET_NEVER_PERSIST" not in path.read_text()
        assert "JevTransport" not in path.read_text()
    loaded = ChessMoveProgram()
    loaded.load(str(compiled_path))
    task_lm = runner._build_lm(config.task_model)
    loaded.set_lm(task_lm)
    with dspy.context(adapter=JevAdapter()):
        assert loaded(**INPUTS).move == "e2e3"
    assert payloads[-1] == payloads[0]


def test_actual_single_round_bootstrap(
    offline_run: tuple[RunConfig, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, payloads = offline_run
    config = RunConfig.model_validate(
        {
            **config.model_dump(),
            "method": "bootstrap_few_shot",
            "prompt_model": {"name": "arbitrary/generative"},
            "optimizer": {"max_bootstrapped_demos": 1, "max_labeled_demos": 0},
        }
    )
    monkeypatch.setattr(dspy, "LM", Mock(return_value=DummyLM([])))
    assert runner.run(config)["metrics"]["accuracy"] == 1
    assert len(payloads) == 2
    assert "Labeled example 1:" in payloads[-1]["questions"]["move"]["instructions"]


def test_mutated_invalid_config_fails_before_io(
    offline_run: tuple[RunConfig, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, payloads = offline_run
    config.use_analysis = True
    io_mock = Mock(side_effect=AssertionError("Must validate first"))
    monkeypatch.setattr(runner, "_load_splits", io_mock)
    with pytest.raises(ValueError, match="use_analysis=false"):
        runner.run(config)
    assert not config.output_dir.exists()
    assert not payloads


def test_generative_runner_keeps_json_adapter(
    offline_run: tuple[RunConfig, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, payloads = offline_run
    config.task_model = ModelSettings(name="arbitrary/generative")
    lm = DummyLM([{"move": "e2e3"}], adapter=dspy.JSONAdapter())
    monkeypatch.setattr(runner, "_build_lm", lambda _: lm)
    original = runner._score

    def score(*args: Any) -> dict[str, Any]:
        assert type(dspy.settings.adapter) is dspy.JSONAdapter
        return original(*args)

    monkeypatch.setattr(runner, "_score", score)
    assert runner.run(config)["metrics"]["accuracy"] == 1
    assert not payloads
