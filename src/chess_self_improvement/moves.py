"""Extract and resolve lenient model output against legal chess moves."""

from __future__ import annotations

import re
from collections.abc import Iterable

import chess

from chess_self_improvement.domain import MoveCandidate, MoveResolution

_UCI_PATTERN = re.compile(
    r"(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])", re.I
)
_SAN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])((?:O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?)(?![A-Za-z0-9])",
    re.I,
)
_FINAL_MOVE_PATTERN = re.compile(
    r"(?im)^[^\S\r\n]*final[ \t]+move[ \t]*:[ \t]*(.*?)[ \t]*$"
)


def _fragments(text: str) -> Iterable[tuple[str, str]]:
    stripped = text.strip()
    if stripped:
        yield stripped.strip("`[](){} \t\r\n.,;:"), "whole_output"
    for match in _UCI_PATTERN.finditer(text):
        yield match.group(1), "uci_from_text"
    for match in _SAN_PATTERN.finditer(text):
        yield match.group(1), "san_from_text"


def resolve_move(raw_output: str, board: chess.Board) -> MoveResolution:
    """Resolve exactly one legal SAN or UCI move from potentially noisy output."""
    final_moves = _FINAL_MOVE_PATTERN.findall(raw_output)
    parse_text = final_moves[-1] if final_moves else raw_output
    candidates: list[MoveCandidate] = []
    seen_fragments: set[tuple[str, str]] = set()
    resolved: dict[str, tuple[str, str]] = {}

    fragments: Iterable[tuple[str, str]]
    if final_moves:
        fragments = ((parse_text.strip("`[](){} \t\r\n.,;"), "final_move"),)
    else:
        fragments = _fragments(parse_text)
    for fragment, source in fragments:
        key = (fragment.casefold(), source)
        if not fragment or key in seen_fragments:
            continue
        seen_fragments.add(key)
        move: chess.Move | None = None
        normalized = fragment.replace("0", "O")
        try:
            move = chess.Move.from_uci(fragment.lower())
            if move not in board.legal_moves:
                move = None
        except ValueError:
            pass
        if move is None:
            try:
                move = board.parse_san(normalized)
            except ValueError:
                move = None
        candidate = MoveCandidate(text=fragment, source=source)
        if move is not None:
            uci = move.uci()
            san = board.san(move)
            candidate = candidate.model_copy(
                update={"resolved_uci": uci, "resolved_san": san, "legal": True}
            )
            resolved.setdefault(uci, (san, source))
        candidates.append(candidate)

    # Bare square notation in descriptive prose is not a committed move answer.
    if not final_moves and re.search(
        r"\b(?:square|controls?|targets?|weak|strong)\b", parse_text, re.I
    ):
        explicit = re.search(
            r"\b(?:move|play|choose|answer|prediction)\s*(?::|is)?\s*"
            r"(?:`|\[)?(?:[a-h][1-8][a-h][1-8][qrbn]?|"
            r"(?:O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]"
            r"(?:=[QRBN])?)[+#]?)",
            parse_text,
            re.I,
        )
        legal_candidates = [candidate for candidate in candidates if candidate.legal]
        begins_with_legal_move = bool(
            legal_candidates
            and parse_text.strip()
            .casefold()
            .startswith(legal_candidates[0].text.casefold())
        )
        if explicit is None and not begins_with_legal_move:
            return MoveResolution(
                raw_output=raw_output,
                candidates=tuple(candidates),
                error="no_legal_candidate",
            )

    if not candidates:
        return MoveResolution(
            raw_output=raw_output, candidates=(), error="no_candidate"
        )
    if not resolved:
        return MoveResolution(
            raw_output=raw_output,
            candidates=tuple(candidates),
            error="no_legal_candidate",
        )
    if len(resolved) > 1:
        return MoveResolution(
            raw_output=raw_output, candidates=tuple(candidates), error="ambiguous"
        )
    uci, (san, mode) = next(iter(resolved.items()))
    return MoveResolution(
        raw_output=raw_output,
        candidates=tuple(candidates),
        move_uci=uci,
        move_san=san,
        parser_mode=mode,
    )
