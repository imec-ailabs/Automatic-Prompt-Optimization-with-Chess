"""Generate deterministic train and test datasets from a Lichess puzzle snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import chess
import zstandard

DEFAULT_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
REQUIRED_COLUMNS = {"PuzzleId", "FEN", "Moves", "Rating", "Themes"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Local Lichess .csv or .csv.zst snapshot (downloads latest if omitted)",
    )
    parser.add_argument(
        "--download-path",
        type=Path,
        default=Path("data/lichess_db_puzzle.csv.zst"),
        help="Where to cache a downloaded snapshot",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Download the latest snapshot even when the cache exists",
    )
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/puzzle_train.csv")
    )
    parser.add_argument(
        "--test-output", type=Path, default=Path("data/puzzle_test.csv")
    )
    parser.add_argument("--train-size", type=positive_int, default=559)
    parser.add_argument("--test-size", type=positive_int, default=559)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--max-rating", type=int)
    parser.add_argument("--solver-moves", type=positive_int)
    parser.add_argument(
        "--theme",
        action="append",
        default=[],
        help="Required theme; repeat to require multiple themes",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output datasets"
    )
    args = parser.parse_args(argv)
    if args.min_rating is not None and args.min_rating < 0:
        parser.error("--min-rating must be non-negative")
    if args.max_rating is not None and args.max_rating < 0:
        parser.error("--max-rating must be non-negative")
    if (
        args.min_rating is not None
        and args.max_rating is not None
        and args.min_rating > args.max_rating
    ):
        parser.error("--min-rating cannot exceed --max-rating")
    return args


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def download_snapshot(url: str, destination: Path, *, refresh: bool = False) -> Path:
    """Download a snapshot atomically unless it is already cached."""
    if destination.exists() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, temporary)
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return destination


@contextmanager
def open_snapshot(path: Path) -> Iterator[TextIO]:
    """Yield a text stream for a plain CSV or Zstandard-compressed CSV."""
    if path.suffix != ".zst":
        with path.open(newline="", encoding="utf-8") as stream:
            yield stream
        return
    with (
        path.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as raw,
        io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream,
    ):
        yield stream


def valid_row(
    row: dict[str, str],
    *,
    min_rating: int | None,
    max_rating: int | None,
    solver_moves: int | None,
    themes: set[str],
) -> bool:
    """Return whether a source row is valid and satisfies all filters."""
    try:
        rating = int(row["Rating"])
        moves = row["Moves"].split()
        board = chess.Board(row["FEN"])
        if len(moves) < 2:
            return False
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return False
            board.push(move)
    except (KeyError, TypeError, ValueError, chess.InvalidMoveError):
        return False
    return not (
        (min_rating is not None and rating < min_rating)
        or (max_rating is not None and rating > max_rating)
        or (solver_moves is not None and len(moves[1::2]) != solver_moves)
        or not themes.issubset(row["Themes"].split())
    )


def sample_rows(
    stream: TextIO,
    *,
    count: int,
    seed: int,
    min_rating: int | None = None,
    max_rating: int | None = None,
    solver_moves: int | None = None,
    themes: set[str] | None = None,
) -> tuple[list[str], list[dict[str, str]], int]:
    """Select ``count`` matching rows using bounded-memory reservoir sampling."""
    reader = csv.DictReader(stream)
    if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        raise ValueError(
            f"source CSV is missing required columns: {', '.join(missing)}"
        )
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    matched = 0
    required_themes = themes or set()
    for row in reader:
        if not valid_row(
            row,
            min_rating=min_rating,
            max_rating=max_rating,
            solver_moves=solver_moves,
            themes=required_themes,
        ):
            continue
        puzzle_id = row["PuzzleId"]
        if not puzzle_id or puzzle_id in selected_ids:
            continue
        matched += 1
        if len(reservoir) < count:
            reservoir.append(row)
            selected_ids.add(puzzle_id)
        else:
            replacement = rng.randrange(matched)
            if replacement < count:
                selected_ids.remove(reservoir[replacement]["PuzzleId"])
                reservoir[replacement] = row
                selected_ids.add(puzzle_id)
    if matched < count:
        raise ValueError(f"only {matched} rows matched; {count} are required")
    rng.shuffle(reservoir)
    return list(reader.fieldnames), reservoir, matched


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write source rows atomically in their original schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, newline="", encoding="utf-8", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Generate train and test datasets and report their provenance."""
    args = parse_args(argv)
    outputs = (args.train_output, args.test_output)
    if args.train_output == args.test_output:
        raise ValueError("train and test outputs must differ")
    if args.source is not None and args.source in outputs:
        raise ValueError("source and output paths must differ")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output exists (pass --overwrite): {names}")
    source = args.source or download_snapshot(
        DEFAULT_URL, args.download_path, refresh=args.refresh_source
    )
    if source in outputs:
        raise ValueError("download and output paths must differ")
    total = args.train_size + args.test_size
    with open_snapshot(source) as stream:
        fieldnames, rows, matched = sample_rows(
            stream,
            count=total,
            seed=args.seed,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            solver_moves=args.solver_moves,
            themes=set(args.theme),
        )
    write_rows(args.train_output, fieldnames, rows[: args.train_size])
    write_rows(args.test_output, fieldnames, rows[args.train_size :])
    print(f"source: {source}")
    print(f"source_sha256: {sha256(source)}")
    print(f"matching_rows: {matched}")
    print(f"train_rows: {args.train_size} -> {args.train_output}")
    print(f"test_rows: {args.test_size} -> {args.test_output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
