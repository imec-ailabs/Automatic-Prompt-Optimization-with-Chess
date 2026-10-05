"""Verify the paper explorer's scientific joins and missing-data behavior."""

import csv
from pathlib import Path
from typing import Any

import pytest

from scripts.build_results_site import build_dataset, recorded_number


@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    """Rebuild the site from the hash-verified published observations."""
    return build_dataset(Path("results/paper"))


def test_full_main_panel_and_exact_prompt_joins(dataset: dict[str, Any]) -> None:
    """Every observed condition retains all seeds and exact parent prompt text."""
    runs = dataset["runs"]
    assert len(runs) == 165
    assert len({(run["model"], run["method"]) for run in runs}) == 55
    assert all(run["puzzles"] == 559 for run in runs)
    optimized = [run for run in runs if run["method"] != "Baseline"]
    assert len(optimized) == len(dataset["prompts"]) == 141
    assert len({run["prompt_id"] for run in optimized}) == 141
    with Path("results/paper/prompts.csv").open(newline="") as stream:
        parents = {
            row["prompt_id"]: row["text"]
            for row in csv.DictReader(stream)
            if row["row_kind"] == "prompt"
        }
    for run in optimized:
        assert dataset["prompts"][run["prompt_id"]]["text"] == parents[run["prompt_id"]]


def test_paper_headline_scores(dataset: dict[str, Any]) -> None:
    """All-correct puzzle aggregation reproduces the headline SIMBA gain."""
    means = {}
    for method in ("Baseline", "SIMBA"):
        values = [
            run["accuracy"]
            for run in dataset["runs"]
            if run["model"] == "Gemini 3.5 Flash Lite" and run["method"] == method
        ]
        assert len(values) == 3
        means[method] = sum(values) / len(values)
    assert round(means["Baseline"], 2) == 26.30
    assert round(means["SIMBA"], 2) == 33.87
    assert round(means["SIMBA"] - means["Baseline"], 2) == 7.57


def test_missing_budget_and_trace_are_not_invented(dataset: dict[str, Any]) -> None:
    """Unknown cost stays null; trace indices retain zero and original semantics."""
    for run in dataset["runs"]:
        if run["method"] == "Baseline":
            assert run["compile"] == 0
        elif run["model"] in ("Claude Haiku 4.5", "Jev 1.13"):
            assert run["compile"] is None
            assert run["cumulative"] is None
        if run["compile"] is not None and run["baseline_reference"] is not None:
            assert run["cumulative"] == pytest.approx(
                run["compile"] + run["baseline_reference"]
            )
    trace = dataset["prompts"]["prompt/jev-1-13/copro/43"]["trace"]
    assert trace["best_iteration"] == 0
    assert dataset["prompts"]["prompt/claude-haiku-4-5/gepa/42"]["trace"] is None
    assert recorded_number("") is None
    assert recorded_number(None) is None
    assert recorded_number("0") == 0
    with pytest.raises(ValueError):
        recorded_number("nan")


def test_cumulative_cost_uses_model_baseline_without_seed_pairing(
    dataset: dict[str, Any],
) -> None:
    """Add the baseline reference once, excluding optimized-prompt evaluation."""
    runs = dataset["runs"]
    baselines = [
        run
        for run in runs
        if run["model"] == "Gemini 3.5 Flash Lite" and run["method"] == "Baseline"
    ]
    assert {run["seed"] for run in baselines} == {"42", "123", "456"}
    reference = sum(run["evaluation"] for run in baselines) / 3
    for run in baselines:
        assert run["cumulative"] == run["evaluation"]
        assert run["prompt_status"] == "baseline"
    optimized = [
        run
        for run in runs
        if run["model"] == "Gemini 3.5 Flash Lite" and run["method"] == "SIMBA"
    ]
    for run in optimized:
        assert run["baseline_reference"] == pytest.approx(reference)
        assert set(run["baseline_reference_ids"]) == {r["id"] for r in baselines}
        assert run["cumulative"] == pytest.approx(reference + run["compile"])
        assert run["cumulative"] != pytest.approx(run["compile"] + run["evaluation"])


def test_unchanged_prompts_retain_optimization_spending(
    dataset: dict[str, Any],
) -> None:
    """No-update outcomes remain distinct from zero-compilation baselines."""
    unchanged = [run for run in dataset["runs"] if run["prompt_status"] == "unchanged"]
    assert len(unchanged) == 21
    for run in unchanged:
        assert dataset["prompts"][run["prompt_id"]]["unchanged"]
        if run["compile"] is not None:
            assert run["compile"] > 0
            assert run["cumulative"] > run["baseline_reference"]
    # BFS retains the original instruction but adds demonstrations: it is updated.
    assert all(
        run["prompt_status"] == "updated"
        for run in dataset["runs"]
        if run["method"] == "BFS"
    )
