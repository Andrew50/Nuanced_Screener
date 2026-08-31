from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .common import _config
from .root import candidates_app


@candidates_app.command("latest")
def candidates_latest(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    window_size: int = typer.Option(100, "--window-size"),
    end_lookback_bars: int = typer.Option(7, "--end-lookback-bars"),
    limit: int = typer.Option(200, "--limit", help="Print top-N rows (0=print none)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional output parquet path."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    """
    Generate high-recall heuristic candidates over the latest window per ticker.
    """
    from ..candidates import CandidateSpec, propose_latest_candidates

    cfg = _config(repo_root=repo_root, window_size=int(window_size), duckdb_threads=duckdb_threads)
    spec = CandidateSpec(window_size=int(window_size), end_lookback_bars=int(end_lookback_bars))
    df = propose_latest_candidates(cfg, spec=spec)
    if df.empty:
        print("[yellow]No candidates emitted[/yellow]")
        return

    out_path = out or (cfg.paths.derived_dir / "candidates_latest.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[green]Wrote[/green] {out_path} rows={len(df)} tickers={df['ticker'].nunique()}")
    if int(limit) > 0:
        print(df.head(int(limit)))

