from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import LoaderConfig
from ..duckdb_utils import connect


def _sql_quote_path(p: Path) -> str:
    return "'" + str(p).replace("'", "''") + "'"


@dataclass(frozen=True)
class CandidatePoolSpec:
    setups: tuple[str, ...]
    start_date: date | None = None
    end_date: date | None = None
    min_close: float = 1.0
    min_dollar_vol: float = 0.0
    max_candidates: int | None = None
    seed: int = 1337


def build_candidate_pool(config: LoaderConfig, *, spec: CandidatePoolSpec) -> pd.DataFrame:
    """
    Build a reusable candidate pool: (ticker, asof_date, setup, source, ...).
    """
    setups = tuple(str(s).strip() for s in (spec.setups or ()))
    setups = tuple(s for s in setups if s)
    if not setups:
        raise ValueError("CandidatePoolSpec.setups must be non-empty")

    parts = config.paths.list_polygon_grouped_daily_partitions()
    parquet_files: list[Path] = []
    source = "per_ticker_raw"
    if parts:
        # Restrict to date range if requested.
        ds = sorted(parts.keys())
        if spec.start_date is not None:
            ds = [d for d in ds if d >= spec.start_date]
        if spec.end_date is not None:
            ds = [d for d in ds if d <= spec.end_date]
        parquet_files = [parts[d] for d in ds if d in parts]
        if parquet_files:
            source = "polygon_grouped_daily"

    con = connect(config)

    if parquet_files:
        files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"
        raw_rel = f"read_parquet({files_sql})"
    else:
        # DuckDB supports globbing.
        raw_glob = (config.paths.raw_dir / "*.parquet").as_posix()
        raw_rel = f"read_parquet('{raw_glob}')"

    # Base selection.
    where = []
    params: list[object] = []
    if spec.start_date is not None:
        where.append("CAST(date AS DATE) >= ?")
        params.append(spec.start_date)
    if spec.end_date is not None:
        where.append("CAST(date AS DATE) <= ?")
        params.append(spec.end_date)
    where.append("close IS NOT NULL")
    where.append("volume IS NOT NULL")
    where.append("close >= ?")
    params.append(float(spec.min_close))
    if float(spec.min_dollar_vol) > 0:
        where.append("(close * volume) >= ?")
        params.append(float(spec.min_dollar_vol))
    where_sql = " AND ".join(where) if where else "TRUE"

    base = con.execute(
        f"""
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS asof_date,
          CAST(close AS DOUBLE) AS close,
          CAST(volume AS DOUBLE) AS volume,
          CAST(close * volume AS DOUBLE) AS dollar_vol
        FROM {raw_rel}
        WHERE {where_sql}
        """,
        params,
    ).df()
    if base.empty:
        return pd.DataFrame(columns=["ticker", "asof_date", "setup", "source", "close", "volume", "dollar_vol"])

    # Optional sampling/limiting (favor liquidity and recency by default).
    out = base.sort_values(["asof_date", "dollar_vol"], ascending=[False, False]).reset_index(drop=True)
    if spec.max_candidates is not None and int(spec.max_candidates) > 0 and len(out) > int(spec.max_candidates):
        out = out.head(int(spec.max_candidates)).reset_index(drop=True)

    # Expand to setups (Cartesian product).
    rows = []
    for setup in setups:
        tmp = out.copy()
        tmp["setup"] = str(setup)
        tmp["source"] = source
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)

