"""Offline tests for benchmark parsing, loading, and scoring."""

from pathlib import Path
from types import SimpleNamespace

import chess

from chess_self_improvement.benchmarks.independent_positions import exact_move_metric
from chess_self_improvement.datasets import PuzzleFilter, iter_lichess_puzzles
from chess_self_improvement.dspy_program import puzzle_task_examples


def _csv(path: Path) -> None:
    path.write_text(
        "PuzzleId,FEN,Moves,Rating,Themes\n"
        f"p1,{chess.STARTING_FEN},e2e4 e7e5,1000,opening\n",
        encoding="utf-8",
    )


def test_independent_position_expansion_and_metric(tmp_path: Path) -> None:
    path = tmp_path / "puzzles.csv"
    _csv(path)
    task = next(iter_lichess_puzzles(path, PuzzleFilter(seed=42)))
    example = puzzle_task_examples(task)[0]

    assert example.move == "e7e5"
    assert exact_move_metric(example, SimpleNamespace(move="e5")) == 1.0
