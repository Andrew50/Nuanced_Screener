"""Typer CLI package for the `ns` entry point."""

from __future__ import annotations

from .root import app, main
from . import data as _data  # noqa: F401
from . import candidates as _candidates  # noqa: F401
from . import weak as _weak  # noqa: F401
from . import models as _models  # noqa: F401

__all__ = ["app", "main"]
