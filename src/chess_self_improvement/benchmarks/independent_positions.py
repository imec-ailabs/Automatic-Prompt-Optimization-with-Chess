"""Benchmark for choosing the reference move in an independent position."""

from __future__ import annotations

from typing import Any

import chess
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from pydantic import BaseModel, Field, model_validator

from chess_self_improvement.moves import resolve_move


class PuzzleTask(BaseModel):
    """A Lichess puzzle line used to derive independent solver positions."""

    puzzle_id: str
    initial_fen: str
    moves_uci: tuple[str, ...] = Field(min_length=2)
    rating: int | None = Field(default=None, ge=0)
    themes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_moves(self) -> PuzzleTask:
        """Validate that the complete reference line is legal."""
        board = chess.Board(self.initial_fen)
        for move_text in self.moves_uci:
            try:
                move = chess.Move.from_uci(move_text)
            except ValueError as exc:
                raise ValueError(f"Invalid UCI move: {move_text}") from exc
            if move not in board.legal_moves:
                raise ValueError(f"Illegal reference move {move_text} in {board.fen()}")
            board.push(move)
        return self


def solver_positions(task: PuzzleTask) -> list[tuple[chess.Board, str, str]]:
    """Build each solver position by replaying the authoritative line."""
    board = chess.Board(task.initial_fen)
    positions: list[tuple[chess.Board, str, str]] = []
    for index, move_uci in enumerate(task.moves_uci):
        if index % 2 == 0:
            board.push_uci(move_uci)
            continue
        positions.append((board.copy(), move_uci, task.moves_uci[index - 1]))
        board.push_uci(move_uci)
    return positions


def exact_move_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score exact legal UCI equivalence for one independent position."""
    del trace
    move_text = str(getattr(prediction, "move", prediction) or "")
    resolution = resolve_move(move_text, chess.Board(str(example.position_fen)))
    return float(resolution.move_uci == str(example.move))


def gepa_move_metric(
    gold: Any,
    prediction: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
    *,
    include_ground_truth: bool = False,
) -> ScoreWithFeedback:
    """Return classified move feedback in GEPA's metric format."""
    del trace, pred_name, pred_trace
    score = exact_move_metric(gold, prediction)
    move_text = str(getattr(prediction, "move", prediction) or "")
    resolution = resolve_move(move_text, chess.Board(str(gold.position_fen)))
    if score == 1.0:
        feedback = "Correct legal move. Preserve concise single-move output."
    elif resolution.error == "no_candidate":
        feedback = (
            "Failure type: parse failure. Return exactly one legal move in SAN or "
            "UCI notation."
        )
    elif not resolution.legal:
        feedback = (
            "Failure type: illegal or ambiguous move. Return exactly one move from "
            "the supplied legal UCI list."
        )
    else:
        feedback = (
            "Failure type: wrong legal move. Reassess forcing checks, captures, "
            "threats, and replies."
        )
    if include_ground_truth and score == 0.0:
        feedback += f" Expected move: {gold.move}."
    return ScoreWithFeedback(score=score, feedback=feedback)


def simba_move_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
    *,
    include_ground_truth: bool = False,
) -> Any:
    """Return move feedback through SIMBA's structured reward contract."""
    from chess_self_improvement.optimization.feedback import simba_feedback

    result = gepa_move_metric(
        example,
        prediction,
        trace,
        include_ground_truth=include_ground_truth,
    )
    return simba_feedback(result.score, result.feedback)
