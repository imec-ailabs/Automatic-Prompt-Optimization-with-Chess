"""DSPy-required chess program and training example conversion."""

from __future__ import annotations

from typing import Any

import dspy

from chess_self_improvement.benchmarks.independent_positions import (
    PuzzleTask,
    solver_positions,
)


class ChessMoveSignature(dspy.Signature):  # type: ignore[misc]
    """Solve the chess puzzle by finding the single best move."""

    position_fen: str = dspy.InputField(desc="Current position in FEN")
    side_to_move: str = dspy.InputField(desc="white or black")
    legal_moves_uci: str = dspy.InputField(desc="Comma-separated legal UCI moves")
    move: str = dspy.OutputField(
        desc="Exactly one legal SAN or UCI move, without reasoning or labels"
    )


class ChessMoveAnalysisSignature(dspy.Signature):  # type: ignore[misc]
    """Solve the chess puzzle by finding the single best move."""

    position_fen: str = dspy.InputField(desc="Current position in FEN")
    side_to_move: str = dspy.InputField(desc="white or black")
    legal_moves_uci: str = dspy.InputField(desc="Comma-separated legal UCI moves")
    analysis: str = dspy.OutputField(
        desc=(
            "Briefly analyse the position in at most 3 short sentences: name the "
            "key checks, captures, pins, forks, or threats and the one or two "
            "candidate moves they point to. Be concise; do not enumerate every "
            "line or re-derive the board."
        )
    )
    move: str = dspy.OutputField(desc="Exactly one legal SAN or UCI move")


class ChessMoveProgram(dspy.Module):  # type: ignore[misc]
    """DSPy program whose predictor can be compiled by GEPA, MIPRO, or SIMBA."""

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        use_analysis: bool = False,
    ) -> None:
        super().__init__()
        if predictor is not None:
            self.choose_move = predictor
        else:
            sig = ChessMoveAnalysisSignature if use_analysis else ChessMoveSignature
            self.choose_move = dspy.Predict(sig)

    def forward(
        self,
        *,
        position_fen: str,
        side_to_move: str,
        legal_moves_uci: str,
    ) -> Any:
        """Return a DSPy prediction containing one move."""
        return self.choose_move(
            position_fen=position_fen,
            side_to_move=side_to_move,
            legal_moves_uci=legal_moves_uci,
        )


def puzzle_example(
    *,
    position_fen: str,
    side_to_move: str,
    legal_moves_uci: tuple[str, ...],
    expected_move_uci: str,
    puzzle_id: str | None = None,
    rating: int | None = None,
    themes: tuple[str, ...] = (),
    position_index: int = 0,
) -> Any:
    """Build a DSPy training example without placing the target in its inputs."""
    return dspy.Example(
        position_fen=position_fen,
        side_to_move=side_to_move,
        legal_moves_uci=", ".join(legal_moves_uci),
        move=expected_move_uci,
        puzzle_id=puzzle_id,
        rating=rating,
        themes=themes,
        position_index=position_index,
    ).with_inputs(
        "position_fen",
        "side_to_move",
        "legal_moves_uci",
    )


def puzzle_task_examples(task: PuzzleTask) -> list[Any]:
    """Convert every solver position into an independent DSPy QA example."""
    positions = solver_positions(task)
    examples: list[Any] = []
    for position_index, (board, expected, _) in enumerate(positions):
        example = puzzle_example(
            position_fen=board.fen(),
            side_to_move="white" if board.turn else "black",
            legal_moves_uci=tuple(move.uci() for move in board.legal_moves),
            expected_move_uci=expected,
            puzzle_id=task.puzzle_id,
            rating=task.rating,
            themes=task.themes,
            position_index=position_index,
        )
        # Lets consumers of results.json tell a fully evaluated source record
        # from one truncated by ``test_limit``.
        example.position_count = len(positions)
        examples.append(example)
    return examples
