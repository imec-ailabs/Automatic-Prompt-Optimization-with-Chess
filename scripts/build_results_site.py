"""Build a compact, provenance-linked dataset for the static paper explorer."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

MODELS: dict[str, str] = {
    "gpt-4o-mini": "GPT-4o Mini",
    "jev": "Jev 1.13",
    "qwen38-27b": "Qwen3.8 27B",
    "claude-haiku-45": "Claude Haiku 4.5",
    "deepseek-v4-pro-0813": "DeepSeek V4 Pro 0813",
    "gemini-35-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt56-luna-low": "GPT-5.6 Luna",
    "muse-spark-contributor-low": "Muse Spark 1.2 Contributor",
}
METHODS: dict[str, str] = {
    "baseline": "Baseline",
    "bootstrap-few-shot": "BFS",
    "bootstrap-random-search": "BRS",
    "copro": "COPRO",
    "gepa": "GEPA",
    "mipro-v2": "MIPROv2",
    "simba": "SIMBA",
}
BASELINE: str = "Solve the chess puzzle by finding the single best move."


def recorded_number(value: Any) -> float | None:
    """Keep absent values distinct from recorded zero; reject invalid costs."""
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid nonnegative value: {value!r}")
    return number


def build_dataset(root: Path) -> dict[str, Any]:
    """Verify source hashes, aggregate main-panel puzzles, and join artifacts."""
    with gzip.open(root / "public_manifest.json.gz", "rt") as stream:
        manifest = json.load(stream)
    for name in ("prompts.csv", "independent_positions.csv.gz"):
        with (root / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")

    dictionaries = manifest["dictionaries"]
    evaluations = {
        key: value
        for key, value in dictionaries["evaluations"].items()
        if value["cohort"] == "main_panel"
    }
    prompts: dict[str, dict[str, Any]] = {}
    prompt_index: dict[tuple[str, str, str], str] = {}
    with (root / "prompts.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["row_kind"] != "prompt":
                continue
            key = row["prompt_id"]
            meta = dictionaries["prompts"][key]
            identity = (meta["model"], meta["optimizer"], meta["seed"])
            if identity in prompt_index:
                raise ValueError(f"Ambiguous prompt identity: {identity}")
            prompt_index[identity] = key
            prompts[key] = {
                "text": row["text"],
                "sha256": meta["source_program_sha256"],
                "unchanged": row["prompt_unchanged"] == "True",
                "condition_no_update": row["condition_no_update"] == "True",
                "trace": meta["compile_metadata"].get("training_trace_summary"),
                "whole_run_costs": meta["compile_metadata"].get(
                    "whole_run_reported_cost_usd"
                ),
                "cost_gaps": [gap for gap in meta["gaps"] if "cost" in gap],
            }

    sources: dict[str, dict[str, list[bool]]] = defaultdict(dict)
    seen: set[tuple[str, str, str]] = set()
    row_costs: dict[str, list[float | None]] = defaultdict(list)
    with gzip.open(root / "independent_positions.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            memberships = (
                set(json.loads(row["evaluation_ids_json"])) & evaluations.keys()
            )
            for key in memberships:
                identity = (key, row["source_id"], row["move_index"])
                if identity in seen:
                    raise ValueError(f"Duplicate main-panel position: {identity}")
                seen.add(identity)
                sources[key].setdefault(row["source_id"], []).append(
                    row["correct"].lower() == "true"
                )
                row_costs[key].append(recorded_number(row["cost_usd"]))

    runs: list[dict[str, Any]] = []
    for key, evaluation in evaluations.items():
        puzzles = sources[key]
        correct = sum(all(positions) for positions in puzzles.values())
        if (
            len(puzzles) != evaluation["expected_sources"]
            or len(row_costs[key]) != evaluation["expected_positions"]
            or correct != evaluation["found_correct_sources"]
            or correct != evaluation.get("expected_correct_sources", correct)
        ):
            raise ValueError(f"Main-panel coverage/accuracy mismatch: {key}")
        model = MODELS[evaluation["model"]]
        method = METHODS[evaluation["optimizer"]]
        seed = str(evaluation["seed"])
        prompt_id = None
        if method != "Baseline":
            prompt_id = prompt_index[(model, method, seed)]

        ledger = evaluation.get("run_cost_ledger", {})
        compile_cost = recorded_number(ledger.get("compile_cost_usd"))
        # Zero optimization expense is meaningful only for an unoptimized baseline.
        if method == "Baseline":
            compile_cost = 0.0
        elif compile_cost == 0:
            compile_cost = None
        evaluation_cost = recorded_number(ledger.get("evaluation_cost_usd"))
        cost_source: str | None = "run_cost_ledger"
        if evaluation_cost in (None, 0):
            costs = row_costs[key]
            evaluation_cost = (
                sum(cost for cost in costs if cost is not None)
                if costs and all(cost is not None for cost in costs)
                else None
            )
            if evaluation_cost == 0:
                evaluation_cost = None
            cost_source = "complete_position_costs" if evaluation_cost else None
        runs.append(
            {
                "id": key,
                "model": model,
                "method": method,
                "seed": seed,
                "accuracy": 100 * correct / len(puzzles),
                "correct": correct,
                "puzzles": len(puzzles),
                "compile": compile_cost,
                "evaluation": evaluation_cost,
                "evaluation_cost_source": cost_source,
                "prompt_id": prompt_id,
                "task_calls": recorded_number(ledger.get("compile_task_lm_calls")),
                "meta_calls": recorded_number(ledger.get("compile_prompt_lm_calls")),
            }
        )
    for model in MODELS.values():
        baselines = [
            run for run in runs if run["model"] == model and run["method"] == "Baseline"
        ]
        if len(baselines) != 3:
            raise ValueError(f"Expected three baseline evaluations for {model}")
        baseline_mean = (
            mean(run["evaluation"] for run in baselines)
            if all(run["evaluation"] is not None for run in baselines)
            else None
        )
        for run in runs:
            if run["model"] != model:
                continue
            # Baseline labels are not paired optimization seeds. Use one declared
            # per-model reference cost for every optimized artifact instead.
            reference_cost = (
                run["evaluation"] if run["method"] == "Baseline" else baseline_mean
            )
            run["baseline_reference"] = reference_cost
            run["baseline_reference_ids"] = (
                [run["id"]]
                if run["method"] == "Baseline"
                else [baseline["id"] for baseline in baselines]
            )
            run["cumulative"] = (
                reference_cost + run["compile"]
                if reference_cost is not None and run["compile"] is not None
                else None
            )
            run["prompt_status"] = (
                "baseline"
                if run["method"] == "Baseline"
                else "unchanged"
                if prompts[run["prompt_id"]]["unchanged"]
                else "updated"
            )
    return {
        "schema_version": 2,
        "cumulative_cost_definition": (
            "Optimization cost plus model mean baseline evaluation cost. Baseline "
            "runs use their own evaluation cost. No seed pairing is assumed. "
            "Selected-prompt evaluation cost is excluded."
        ),
        "models": list(MODELS.values()),
        "methods": list(METHODS.values()),
        "runs": runs,
        "prompts": prompts,
        "baseline_prompt": BASELINE,
        "sources": {
            name: manifest["files"][name]["sha256"]
            for name in ("prompts.csv", "independent_positions.csv.gz")
        },
    }


def main() -> None:
    """Regenerate the browser dataset without APIs or application dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/paper"))
    parser.add_argument("--output", type=Path, default=Path("site/results.json"))
    args = parser.parse_args()
    dataset = build_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    logging.basicConfig(level=logging.INFO)
    logging.info("Exported %s runs to %s", len(dataset["runs"]), args.output)


if __name__ == "__main__":
    main()
