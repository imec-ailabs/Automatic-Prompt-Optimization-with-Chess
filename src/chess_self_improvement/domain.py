"""Models used to audit parsing of model-generated chess moves."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MoveCandidate(BaseModel):
    """Record one move-like fragment extracted from model output."""

    text: str
    source: str
    resolved_uci: str | None = None
    resolved_san: str | None = None
    legal: bool = False


class MoveResolution(BaseModel):
    """Describe the auditable result of parsing a model response."""

    raw_output: str
    candidates: tuple[MoveCandidate, ...]
    move_uci: str | None = None
    move_san: str | None = None
    parser_mode: str | None = None
    error: Literal["no_candidate", "no_legal_candidate", "ambiguous"] | None = None
    parser_version: str = "1.1"

    @property
    def legal(self) -> bool:
        """Return whether parsing identified exactly one legal move."""
        return self.move_uci is not None
