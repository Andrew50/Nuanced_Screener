from __future__ import annotations

from pathlib import Path

import typer

from ..dotenv import load_dotenv

app = typer.Typer(add_completion=False, no_args_is_help=True)
models_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(models_app, name="models")
candidates_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(candidates_app, name="candidates")
weak_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(weak_app, name="weak")
setups_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(setups_app, name="setups")


def main() -> None:
    # Load `.env` from the current working directory (repo root in typical usage).
    # Values in the shell environment take precedence over `.env`.
    load_dotenv(Path(".") / ".env", override=False)
    app()
