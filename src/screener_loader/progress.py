from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@contextmanager
def progress_ctx(*, transient: bool = False) -> Iterator[Progress]:
    """
    Shared progress-bar style for long-running tasks (training, pretraining, etc.).

    Notes:
    - Rich auto-disables nicely when not attached to a TTY.
    - `transient=False` keeps the final bar visible (useful for logs).
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=bool(transient),
        refresh_per_second=10,
    ) as progress:
        yield progress

