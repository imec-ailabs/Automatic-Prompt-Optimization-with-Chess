"""Load Lichess puzzle data without internal repository dependencies."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from chess_self_improvement.benchmarks.independent_positions import PuzzleTask


class PuzzleFilter(BaseModel):
    """Select a deterministic subset of Lichess puzzles."""

    limit: int | None = Field(default=None, ge=1)
    min_rating: int | None = Field(default=None, ge=0)
    max_rating: int | None = Field(default=None, ge=0)
    solver_moves: int | None = Field(default=None, ge=1)
    themes: tuple[str, ...] = ()
    seed: int = 0


def iter_lichess_puzzles(
    path: Path, puzzle_filter: PuzzleFilter | None = None
) -> Iterator[PuzzleTask]:
    """Yield filtered puzzles from the official Lichess CSV format."""
    selected = puzzle_filter or PuzzleFilter()
    tasks: list[PuzzleTask] = []
    matched = 0
    rng = random.Random(selected.seed)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rating = int(row["Rating"])
            themes = tuple(row.get("Themes", "").split())
            moves = tuple(row["Moves"].split())
            solver_moves = len(moves[1::2])
            if selected.min_rating is not None and rating < selected.min_rating:
                continue
            if selected.max_rating is not None and rating > selected.max_rating:
                continue
            if (
                selected.solver_moves is not None
                and solver_moves != selected.solver_moves
            ):
                continue
            if selected.themes and not set(selected.themes).issubset(themes):
                continue
            task = PuzzleTask(
                puzzle_id=row["PuzzleId"],
                initial_fen=row["FEN"],
                moves_uci=moves,
                rating=rating,
                themes=themes,
            )
            matched += 1
            if selected.limit is None or len(tasks) < selected.limit:
                tasks.append(task)
            else:
                replacement = rng.randrange(matched)
                if replacement < selected.limit:
                    tasks[replacement] = task
    rng.shuffle(tasks)
    yield from tasks
