from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .calendar_utils import TradingCalendar, last_n_trading_days_ending_at
from .config import LoaderConfig
from .duckdb_utils import connect
from .paths import DataPaths, atomic_replace, ensure_dirs


# Keep these consistent with `derived.py` (duplicated intentionally to avoid importing private constants).
_FEATURE_SQL: dict[str, str] = {
    # Returns (based on close)
    "ret_1d": "(close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_1d",
    "ret_5d": "(close / LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_5d",
    "ret_21d": "(close / LAG(close, 21) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_21d",
    # Liquidity
    "vol_avg_20": "AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_avg_20",
    "dollar_vol_avg_20": "AVG(volume * close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS dollar_vol_avg_20",
    # Volatility-ish
    "range_pct": "((high - low) / NULLIF(close, 0.0)) AS range_pct",
}

_FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    "ret_1d": 1,
    "ret_5d": 5,
    "ret_21d": 21,
    "vol_avg_20": 19,
    "dollar_vol_avg_20": 19,
    "range_pct": 0,
}


def _max_feature_lookback_days(feature_columns: Iterable[str]) -> int:
    m = 0
    for f in feature_columns:
        key = str(f).strip()
        if not key:
            continue
        m = max(m, _FEATURE_LOOKBACK_DAYS.get(key, 0))
    return m


def _sql_quote_path(p: Path) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def stable_sample_id(ticker: str, asof_date: date, setup: str) -> str:
    """
    Stable 64-bit-ish id encoded as 16 hex chars.
    """
    s = f"{ticker.upper()}|{asof_date.isoformat()}|{setup}"
    h = hashlib.sha1(s.encode("utf-8")).digest()[:8]
    return h.hex()


@dataclass(frozen=True)
class WindowedBuildSpec:
    window_size: int
    feature_columns: tuple[str, ...] = ()
    # Masking semantics: decision at D open => only keep open at asof_date.
    mask_current_day_to_open_only: bool = True
    require_full_window: bool = True


@dataclass(frozen=True)
class WindowedBuildMeta:
    source_csv: str | None
    source_csv_mtime_ns: int | None
    source_csv_size: int | None
    window_size: int
    feature_columns: list[str]
    mask_current_day_to_open_only: bool
    rows: int
    samples: int


def _read_existing_meta(meta_path: Path) -> dict | None:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(meta_path: Path, meta: WindowedBuildMeta) -> None:
    meta_path.write_text(json.dumps(meta.__dict__, indent=2, default=str) + "\n", encoding="utf-8")


def build_windowed_bars(
    labels_df: pd.DataFrame,
    *,
    config: LoaderConfig,
    spec: WindowedBuildSpec,
    out_path: Path | None = None,
    source_csv: Path | None = None,
    cal: TradingCalendar | None = None,
    reuse_if_unchanged: bool = True,
) -> Path:
    """
    Materialize long-form windowed bars for each labeled sample.

    Output schema (long):
    - sample_id, ticker, asof_date, setup, label
    - bar_date, t
    - open, high, low, close, volume, adj_close (+ optional derived features)

    Leakage rule (decision at D open):
    - if bar_date == asof_date: keep open but mask high/low/close/volume/adj_close.
    """
    ensure_dirs(config.paths)
    out_path = out_path or config.paths.windowed_bars_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if reuse_if_unchanged and out_path.exists() and meta_path.exists() and source_csv is not None:
        st = source_csv.stat()
        prev = _read_existing_meta(meta_path)
        if (
            prev
            and prev.get("source_csv") == str(source_csv)
            and int(prev.get("source_csv_mtime_ns") or -1) == int(st.st_mtime_ns)
            and int(prev.get("source_csv_size") or -1) == int(st.st_size)
            and int(prev.get("window_size") or -1) == int(spec.window_size)
            and list(prev.get("feature_columns") or []) == list(spec.feature_columns)
            and bool(prev.get("mask_current_day_to_open_only")) == bool(spec.mask_current_day_to_open_only)
        ):
            return out_path

    if labels_df.empty:
        raise ValueError("labels_df is empty")
    required = {"ticker", "asof_date", "setup", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_df missing required columns: {sorted(missing)}")

    cal = cal or TradingCalendar("NYSE")
    window_size = int(spec.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    # One row per sample.
    samples = labels_df.copy()
    samples["ticker"] = samples["ticker"].astype(str).str.upper()
    samples["setup"] = samples["setup"].astype(str)
    samples["sample_id"] = samples.apply(lambda r: stable_sample_id(r["ticker"], r["asof_date"], r["setup"]), axis=1)

    # Ensure (ticker, asof_date, setup) is unique in the working set.
    dup = samples.duplicated(subset=["ticker", "asof_date", "setup"], keep=False)
    if dup.any():
        ex = samples.loc[dup, ["ticker", "asof_date", "setup"]].head(10).to_dict(orient="records")
        raise ValueError(f"labels_df contains duplicate (ticker, asof_date, setup) rows; examples: {ex}")

    # Build the per-sample trading-day window table.
    unique_asof = sorted(set(samples["asof_date"].tolist()))
    windows_by_asof: dict[date, list[date]] = {}
    for d in unique_asof:
        w = last_n_trading_days_ending_at(d, n=window_size, cal=cal)
        if spec.require_full_window and len(w) != window_size:
            windows_by_asof[d] = []
        else:
            windows_by_asof[d] = w

    # Drop samples that don't have a full window (avoids ragged sequences).
    if spec.require_full_window:
        ok_mask = samples["asof_date"].map(lambda d: len(windows_by_asof.get(d, [])) == window_size)
        samples = samples.loc[ok_mask].reset_index(drop=True)
        if samples.empty:
            raise ValueError("No samples have a full trading-day window_size; check your labels date range/window_size.")

    window_rows: list[dict] = []
    for sid, d in zip(samples["sample_id"].tolist(), samples["asof_date"].tolist(), strict=True):
        dates = windows_by_asof.get(d) or []
        # Chronological ordering: t=0 oldest ... t=window_size-1 newest (asof_date at end).
        for t, bar_d in enumerate(dates):
            window_rows.append({"sample_id": sid, "bar_date": bar_d, "t": int(t)})
    windows = pd.DataFrame(window_rows)
    if windows.empty:
        raise ValueError("Window planning produced no rows")

    # Determine which raw data files to read.
    needed_dates = sorted(set(windows["bar_date"].tolist()))
    if not needed_dates:
        raise ValueError("No bar dates required")

    lookback = _max_feature_lookback_days(spec.feature_columns)
    if lookback > 0:
        # Add extra preceding trading days to compute lag/rolling features.
        earliest = needed_dates[0]
        extra = last_n_trading_days_ending_at(earliest, n=lookback, cal=cal)
        needed_dates = sorted(set(needed_dates) | set(extra))

    parts = config.paths.list_polygon_grouped_daily_partitions()
    parquet_files: list[Path]
    parquet_files = []
    if parts:
        parquet_files = [parts[d] for d in needed_dates if d in parts]

    # Fallback: if partitions exist but don't cover this label date range,
    # use per-ticker raw files instead.
    if not parquet_files:
        tickers = sorted(set(samples["ticker"].astype(str).str.upper().tolist()))
        parquet_files = [config.paths.raw_ticker_parquet(t) for t in tickers if config.paths.raw_ticker_parquet(t).exists()]

    tmp_path = Path(str(out_path) + ".tmp")
    con = connect(config)
    con.register("samples", samples)
    con.register("windows", windows)

    if not parquet_files:
        raise ValueError("No raw OHLCV parquet files found for requested samples/dates")

    for f in spec.feature_columns:
        key = str(f).strip()
        if not key:
            continue
        if key not in _FEATURE_SQL:
            raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")

    feature_exprs = []
    for f in spec.feature_columns:
        key = str(f).strip()
        if key:
            feature_exprs.append(_FEATURE_SQL[key])
    feature_sql = ""
    if feature_exprs:
        feature_sql = ",\n          " + ",\n          ".join(feature_exprs)

    files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"

    # Build and write the long windowed dataset.
    # Note: `windows` drives which (sample_id, bar_date, t) rows exist; raw is left-joined so
    # missing ticker/date bars become NULLs (caller can decide to drop/impute).
    if spec.mask_current_day_to_open_only:
        mask_high = "CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.high END AS high"
        mask_low = "CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.low END AS low"
        mask_close = "CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.close END AS close"
        mask_volume = "CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.volume END AS volume"
        mask_adj = "CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.adj_close END AS adj_close"
    else:
        mask_high = "raw.high AS high"
        mask_low = "raw.low AS low"
        mask_close = "raw.close AS close"
        mask_volume = "raw.volume AS volume"
        mask_adj = "raw.adj_close AS adj_close"

    feature_select_sql = ""
    if spec.feature_columns:
        parts = []
        for c in spec.feature_columns:
            key = str(c).strip()
            if not key:
                continue
            if spec.mask_current_day_to_open_only:
                parts.append(
                    f"CASE WHEN windows.bar_date = samples.asof_date THEN NULL ELSE raw.{key} END AS {key}"
                )
            else:
                parts.append(f"raw.{key} AS {key}")
        if parts:
            feature_select_sql = ",\n              " + ",\n              ".join(parts)

    con.execute(
        f"""
        COPY (
          WITH raw AS (
            SELECT
              ticker,
              CAST(date AS DATE) AS date,
              open,
              high,
              low,
              close,
              volume,
              adj_close
              {feature_sql}
            FROM read_parquet({files_sql})
          ),
          joined AS (
            SELECT
              samples.sample_id,
              samples.ticker,
              CAST(samples.asof_date AS DATE) AS asof_date,
              samples.setup,
              samples.label,
              CAST(windows.bar_date AS DATE) AS bar_date,
              CAST(windows.t AS INTEGER) AS t,
              raw.open AS open,
              {mask_high},
              {mask_low},
              {mask_close},
              {mask_volume},
              {mask_adj}
              {feature_select_sql}
            FROM samples
            JOIN windows ON windows.sample_id = samples.sample_id
            LEFT JOIN raw
              ON raw.ticker = samples.ticker
             AND raw.date = windows.bar_date
          )
          SELECT *
          FROM joined
          ORDER BY sample_id, t
        )
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )

    atomic_replace(tmp_path, out_path)

    st = source_csv.stat() if source_csv is not None and source_csv.exists() else None
    meta = WindowedBuildMeta(
        source_csv=str(source_csv) if source_csv is not None else None,
        source_csv_mtime_ns=int(st.st_mtime_ns) if st is not None else None,
        source_csv_size=int(st.st_size) if st is not None else None,
        window_size=int(spec.window_size),
        feature_columns=[str(x) for x in spec.feature_columns],
        mask_current_day_to_open_only=bool(spec.mask_current_day_to_open_only),
        rows=int(len(windows)),
        samples=int(len(samples)),
    )
    _write_meta(meta_path, meta)
    return out_path

