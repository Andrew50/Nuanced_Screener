from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .config import LoaderConfig
from .duckdb_utils import connect, parquet_max_date
from .paths import atomic_replace


RAW_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "source",
    "asof_ts",
]


@dataclass(frozen=True)
class TickerTask:
    ticker: str
    start: Optional[date]
    end: Optional[date]
    full_refresh: bool


@dataclass(frozen=True)
class TickerResult:
    ticker: str
    status: str  # updated|no_change|no_data|failed
    rows_written: int = 0
    last_date: Optional[date] = None
    error: Optional[str] = None


def plan_task_for_ticker(config: LoaderConfig, ticker: str) -> TickerTask | None:
    raw_path = config.paths.raw_ticker_parquet(ticker)

    end = config.end_date or date.today()
    if config.full_refresh:
        start = config.start_date
        return TickerTask(ticker=ticker, start=start, end=end, full_refresh=True)

    # Fast path: initial runs often have no per-ticker files yet. Avoid creating
    # a DuckDB connection per ticker unless the file exists.
    if not raw_path.exists():
        start = config.start_date
    else:
        con = connect(config)
        max_date = parquet_max_date(con, raw_path)
        if max_date is not None:
            start = max_date + timedelta(days=1)
        else:
            start = config.start_date

    if start is not None and start > end:
        return None
    return TickerTask(ticker=ticker, start=start, end=end, full_refresh=False)


def _ensure_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure optional columns exist with stable ordering.
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA
    if "source" not in df.columns:
        df["source"] = pd.NA
    if "asof_ts" not in df.columns:
        df["asof_ts"] = datetime.now(timezone.utc)

    missing = {"ticker", "date", "open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df[RAW_COLUMNS].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def merge_write_ticker_parquet(
    config: LoaderConfig,
    ticker: str,
    new_bars: pd.DataFrame,
    full_refresh: bool,
) -> TickerResult:
    raw_path = config.paths.raw_ticker_parquet(ticker)

    if new_bars is None or new_bars.empty:
        if raw_path.exists() and not full_refresh:
            return TickerResult(ticker=ticker, status="no_change", rows_written=0, last_date=None)
        return TickerResult(ticker=ticker, status="no_data", rows_written=0, last_date=None)

    new_bars = _ensure_raw_columns(new_bars)

    con = connect(config)
    con.register("new_bars", new_bars)

    tmp_path = Path(str(raw_path) + ".tmp")

    if full_refresh or not raw_path.exists():
        con.execute(
            f"""
            COPY (
              SELECT {", ".join(RAW_COLUMNS)}
              FROM new_bars
              ORDER BY date
            )
            TO '{tmp_path.as_posix()}'
            (FORMAT PARQUET, CODEC 'ZSTD');
            """
        )
        atomic_replace(tmp_path, raw_path)
        last_date = new_bars["date"].max()
        return TickerResult(ticker=ticker, status="updated", rows_written=int(len(new_bars)), last_date=last_date)

    # Merge + dedupe by (ticker, date), preferring new rows.
    con.execute(
        f"""
        CREATE TEMP VIEW merged AS
        WITH all_rows AS (
          SELECT {", ".join(RAW_COLUMNS)}, 0 AS is_new
          FROM read_parquet('{raw_path.as_posix()}')
          UNION ALL
          SELECT {", ".join(RAW_COLUMNS)}, 1 AS is_new
          FROM new_bars
        ),
        ranked AS (
          SELECT
            {", ".join(RAW_COLUMNS)},
            ROW_NUMBER() OVER (
              PARTITION BY ticker, date
              ORDER BY is_new DESC, asof_ts DESC
            ) AS rn
          FROM all_rows
        )
        SELECT {", ".join(RAW_COLUMNS)}
        FROM ranked
        WHERE rn = 1
        ORDER BY date
        """
    )
    con.execute(
        f"""
        COPY (SELECT {", ".join(RAW_COLUMNS)} FROM merged)
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, raw_path)
    last_date = con.execute("SELECT max(date) FROM merged").fetchone()[0]
    rows_written = int(con.execute("SELECT count(*) FROM merged").fetchone()[0])
    return TickerResult(ticker=ticker, status="updated", rows_written=rows_written, last_date=last_date)

