"""Command-line interface for local benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from chess_self_improvement.runner import load_config, run

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run chess self-improvement benchmarks locally."""


@app.command()
def evaluate(config: Path) -> None:
    """Run one benchmark configuration on this machine."""
    typer.echo(json.dumps(run(load_config(config)), indent=2))
