"""Offline tests for Lichess dataset generation."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
import zstandard

from scripts.generate_puzzle_datasets import main, open_snapshot, sample_rows

HEADER = "PuzzleId,FEN,Moves,Rating,Themes\n"


def source_csv(count: int = 8) -> str:
    """Build valid, distinct source rows."""
    rows = [
        f"p{index},8/8/8/8/8/8/P6k/K7 w - - 0 1,a2a3 h2g2,"
        f"{1000 + index},endgame quietMove"
        for index in range(count)
    ]
    return HEADER + "\n".join(rows) + "\n"


def test_sample_rows_is_deterministic_and_filters() -> None:
    _, first, matched = sample_rows(
        io.StringIO(source_csv()),
        count=3,
        seed=7,
        min_rating=1002,
        max_rating=1006,
        solver_moves=1,
        themes={"endgame"},
    )
    _, second, _ = sample_rows(
        io.StringIO(source_csv()),
        count=3,
        seed=7,
        min_rating=1002,
        max_rating=1006,
        solver_moves=1,
        themes={"endgame"},
    )

    assert first == second
    assert matched == 5
    assert len(first) == 3


def test_sample_rows_rejects_invalid_source() -> None:
    source = io.StringIO(HEADER + "bad,not-a-fen,a2a3,1000,endgame\n")

    with pytest.raises(ValueError, match="only 0 rows matched"):
        sample_rows(source, count=1, seed=42)


def test_sample_rows_rejects_one_move_lines() -> None:
    source = io.StringIO(
        HEADER + "short,8/8/8/8/8/8/P6k/K7 w - - 0 1,a2a3,1000,endgame\n"
    )

    with pytest.raises(ValueError, match="only 0 rows matched"):
        sample_rows(source, count=1, seed=42)


def test_sample_rows_keeps_duplicate_ids_out_of_sample() -> None:
    duplicate = source_csv(3).replace("p1,", "p0,")

    _, rows, _ = sample_rows(io.StringIO(duplicate), count=2, seed=42)

    assert len({row["PuzzleId"] for row in rows}) == 2


def test_open_snapshot_decompresses_zstandard(tmp_path: Path) -> None:
    source = tmp_path / "puzzles.csv.zst"
    source.write_bytes(zstandard.ZstdCompressor().compress(source_csv().encode()))

    with open_snapshot(source) as stream:
        _, rows, matched = sample_rows(stream, count=2, seed=42)

    assert len(rows) == 2
    assert matched == 8


def test_main_writes_disjoint_datasets(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    source.write_text(source_csv(), encoding="utf-8")

    result = main(
        [
            "--source",
            str(source),
            "--train-output",
            str(train),
            "--test-output",
            str(test),
            "--train-size",
            "3",
            "--test-size",
            "2",
        ]
    )

    train_rows = list(csv.DictReader(train.open(encoding="utf-8")))
    test_rows = list(csv.DictReader(test.open(encoding="utf-8")))
    assert result == 0
    assert len(train_rows) == 3
    assert len(test_rows) == 2
    assert {row["PuzzleId"] for row in train_rows}.isdisjoint(
        row["PuzzleId"] for row in test_rows
    )
