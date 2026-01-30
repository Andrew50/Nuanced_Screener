from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

from .paths import DataPaths


ExecutorType = Literal["threads", "processes"]


@dataclass(frozen=True)
class LoaderConfig:
    repo_root: Path = Path(".")

    # Swap-in sources/vendors without changing storage/query layers
    ticker_source: str = "nasdaq_trader"
    ohlcv_vendor: str = "polygon_grouped"

    # Universe filters
    exclude_test_issues: bool = True
    exclude_etfs: bool = False
    include_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")

    # History pull
    start_date: date | None = None
    end_date: date | None = None
    full_refresh: bool = False
    interval: str = "1d"

    # Polygon grouped-daily date-mode
    lookback_years: int = 2
    refresh_tail_days: int = 3
    polygon_adjusted: bool = True
    polygon_include_otc: bool = False
    calls_per_minute: int = 5

    # Parallelism + throttling
    batch_size: int = 50
    processes: int = 8
    executor: ExecutorType = "threads"
    pause_seconds: float = 0.0

    # Reliability
    max_retries: int = 3
    timeout_seconds: float = 30.0
    fail_fast: bool = False

    # Derived dataset
    window_size: int = 100
    feature_columns: tuple[str, ...] = field(default_factory=tuple)

    # DuckDB runtime
    duckdb_threads: int = 4

    @property
    def paths(self) -> DataPaths:
        return DataPaths(self.repo_root)

    def with_exchanges(self, exchanges: Iterable[str]) -> "LoaderConfig":
        ex = tuple(str(x).upper() for x in exchanges)
        return LoaderConfig(**{**self.__dict__, "include_exchanges": ex})

