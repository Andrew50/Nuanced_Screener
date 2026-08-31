from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Optional

import typer

from ..config import LoaderConfig
from ..paths import ensure_dirs


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha1_json(payload: dict) -> str:
    b = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


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

