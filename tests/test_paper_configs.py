"""Validate the standalone paper examples without constructing or calling LMs."""

from pathlib import Path

import pytest
import yaml

from chess_self_improvement.optimization.parameters import resolve_optimizer_params
from chess_self_improvement.runner import load_config

ROOT = Path(__file__).parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
META_MODEL = "openrouter/google/gemini-3.5-flash"
JEV_MODEL = "typesafe/jev-1.13-20260917"
COHORT = {
    "openrouter/google/gemini-3.5-flash-lite": (1024, 30, "low", True),
    "openrouter/openai/gpt-4o-mini": (1024, 30, "none", False),
    # None: no reasoning flag, so the provider's default reasoning applies.
    "openrouter/anthropic/claude-haiku-4.5": (1024, 30, None, None),
    "openrouter/qwen/qwen3.8-27b": (512, 120, None, None),
    "openrouter/openai/gpt-5.6-luna": (4096, 300, "low", True),
    "openrouter/deepseek/deepseek-v4-pro-0813": (512, 120, None, None),
    "openrouter/meta/muse-spark-1.2-contributor": (16384, 600, "low", True),
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_example_schema_and_roles(path: Path) -> None:
    """Every example is bounded, strict, and explicit about its model roles."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_config(path)

    assert config.task_model.name in {*COHORT, JEV_MODEL}
    assert config.task_model.retries == 1
    assert config.task_model.timeout_seconds is not None
    assert config.seed == 42
    assert config.test_limit == 20
    assert not config.use_analysis
    assert not config.include_ground_truth_feedback
    assert not config.continue_on_error
    assert (ROOT / config.train_path).is_file()
    assert (ROOT / config.test_path).is_file()

    if config.method == "baseline":
        assert config.prompt_model is None
        assert config.optimizer == {}
    else:
        assert "prompt_model" in raw  # Do not accept the runner's target fallback.
        assert config.prompt_model is not None
        assert config.prompt_model.name == META_MODEL
        assert config.prompt_model.temperature == 1
        assert config.prompt_model.max_tokens == 8192
        assert config.prompt_model.timeout_seconds == 30
        assert config.prompt_model.retries == 1
        assert config.prompt_model.extra == {"drop_params": True}
        assert config.train_limit == 20
        assert config.validation_size == 10
        resolve_optimizer_params(config.method, config.optimizer)

    if config.task_model.name == JEV_MODEL:
        assert config.benchmark == "independent_positions"
        assert set(raw["task_model"]) == {"name", "retries", "timeout_seconds"}
        assert config.task_model.timeout_seconds == 30
        assert config.task_model.extra == {}
    else:
        tokens, timeout, effort, include = COHORT[config.task_model.name]
        assert config.task_model.temperature == 0
        assert config.task_model.timeout_seconds == timeout
        assert config.task_model.max_tokens == tokens
        expected_extra: dict[str, object] = {"drop_params": True}
        if effort is not None:
            expected_extra |= {"reasoning_effort": effort, "include_reasoning": include}
        assert config.task_model.extra == expected_extra


def test_complete_cohort_and_supported_task() -> None:
    """The eight small baselines cover the independent-position panel."""
    configs = [load_config(path) for path in CONFIGS]
    targets = [
        config.task_model.name
        for config in configs
        if config.benchmark == "independent_positions" and config.method == "baseline"
    ]

    assert len(targets) == 8
    assert set(targets) == {*COHORT, JEV_MODEL}
    assert {config.benchmark for config in configs} == {"independent_positions"}
    assert len({config.output_dir for config in configs}) == len(configs)
