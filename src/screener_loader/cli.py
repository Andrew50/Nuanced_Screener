from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import LoaderConfig
from .derived import rebuild_last_n_bars_from_polygon_date_partitions
from .paths import ensure_dirs
from .screening import run_named_query
from .universe import build_universe
from .update import update_market_data

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _parse_ymd(value: str | None, flag_name: str) -> date | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(f"{flag_name} must be YYYY-MM-DD; got {value!r}") from e


def _config(
    repo_root: Path = Path("."),
    ticker_source: str = "nasdaq_trader",
    ohlcv_vendor: str = "polygon_grouped",
    exclude_test_issues: bool = True,
    exclude_etfs: bool = False,
    include_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    full_refresh: bool = False,
    interval: str = "1d",
    batch_size: int = 50,
    processes: int = 8,
    executor: str = "threads",
    pause_seconds: float = 0.0,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    fail_fast: bool = False,
    window_size: int = 100,
    feature_columns: Optional[list[str]] = None,
    duckdb_threads: int = 4,
    lookback_years: int = 2,
    refresh_tail_days: int = 3,
    polygon_adjusted: bool = True,
    polygon_include_otc: bool = False,
    calls_per_minute: int = 5,
) -> LoaderConfig:
    start_d = _parse_ymd(start_date, "--start-date")
    end_d = _parse_ymd(end_date, "--end-date")
    cfg = LoaderConfig(
        repo_root=repo_root,
        ticker_source=ticker_source,
        ohlcv_vendor=ohlcv_vendor,
        exclude_test_issues=exclude_test_issues,
        exclude_etfs=exclude_etfs,
        include_exchanges=tuple(x.upper() for x in include_exchanges),
        start_date=start_d,
        end_date=end_d,
        full_refresh=full_refresh,
        interval=interval,
        batch_size=batch_size,
        processes=processes,
        executor="processes" if executor == "processes" else "threads",
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        window_size=window_size,
        feature_columns=tuple(feature_columns or []),
        duckdb_threads=duckdb_threads,
        lookback_years=int(lookback_years),
        refresh_tail_days=int(refresh_tail_days),
        polygon_adjusted=bool(polygon_adjusted),
        polygon_include_otc=bool(polygon_include_otc),
        calls_per_minute=int(calls_per_minute),
    )
    ensure_dirs(cfg.paths)
    return cfg


@app.command()
def universe(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ticker_source: str = typer.Option("nasdaq_trader", "--ticker-source"),
    exclude_test_issues: bool = typer.Option(True, "--exclude-test-issues/--include-test-issues"),
    exclude_etfs: bool = typer.Option(False, "--exclude-etfs/--include-etfs"),
    include_exchanges: tuple[str, str, str] = typer.Option(("NASDAQ", "NYSE", "AMEX"), "--include-exchanges"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ticker_source=ticker_source,
        exclude_test_issues=exclude_test_issues,
        exclude_etfs=exclude_etfs,
        include_exchanges=include_exchanges,
    )
    out = build_universe(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def update(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ohlcv_vendor: str = typer.Option("polygon_grouped", "--ohlcv-vendor"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    full_refresh: bool = typer.Option(False, "--full-refresh"),
    batch_size: int = typer.Option(50, "--batch-size"),
    processes: int = typer.Option(8, "--processes"),
    executor: str = typer.Option("threads", "--executor"),
    pause_seconds: float = typer.Option(0.0, "--pause-seconds"),
    max_retries: int = typer.Option(3, "--max-retries"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
    lookback_years: int = typer.Option(2, "--lookback-years", help="Polygon date-mode: years to backfill."),
    refresh_tail_days: int = typer.Option(3, "--refresh-tail-days", help="Polygon date-mode: re-fetch last N trading days."),
    adjusted: bool = typer.Option(True, "--adjusted/--unadjusted", help="Polygon: adjusted prices."),
    include_otc: bool = typer.Option(False, "--include-otc/--exclude-otc", help="Polygon: include OTC tickers."),
    calls_per_minute: int = typer.Option(5, "--calls-per-minute", help="Polygon free tier: max calls per minute."),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ohlcv_vendor=ohlcv_vendor,
        start_date=start_date,
        end_date=end_date,
        full_refresh=full_refresh,
        batch_size=batch_size,
        processes=processes,
        executor=executor,
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
        lookback_years=lookback_years,
        refresh_tail_days=refresh_tail_days,
        polygon_adjusted=adjusted,
        polygon_include_otc=include_otc,
        calls_per_minute=calls_per_minute,
    )
    update_market_data(cfg)


@app.command("rebuild-last100")
def rebuild_last100(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
    )
    out = rebuild_last_n_bars_from_polygon_date_partitions(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def screen(
    query: str = typer.Option(..., "--query", help="Named query to run (see screener_loader/screening.py)."),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    cfg = _config(repo_root=repo_root)
    df = run_named_query(cfg, query=query, limit=limit)
    print(df)


def main() -> None:
    app()

