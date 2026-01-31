from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .calendar_utils import TradingCalendar, last_n_trading_days_ending_at, next_n_trading_days_starting_after
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
    # Optional per-sample columns copied from input labels_df into output (repeated per bar row).
    # Use for candidate-level meta like cand_t_start/cand_t_end.
    sample_meta_columns: tuple[str, ...] = ()
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
    sample_meta_columns: list[str]
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
    - plus any sample_meta_columns (repeated per bar row)
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
            and list(prev.get("sample_meta_columns") or []) == list(spec.sample_meta_columns)
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
    samples["asof_date"] = pd.to_datetime(samples["asof_date"], errors="coerce").dt.date
    samples["setup"] = samples["setup"].astype(str)
    if samples["asof_date"].isna().any():
        bad = labels_df.loc[samples["asof_date"].isna(), "asof_date"].head(10).tolist()
        raise ValueError(f"Could not parse some asof_date values; examples: {bad}")
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

    # Validate sample meta columns exist and are safe identifiers for DuckDB SELECT.
    # (We avoid quoting here; keep the contract simple and explicit.)
    meta_cols: list[str] = []
    for c in spec.sample_meta_columns:
        key = str(c).strip()
        if not key:
            continue
        if key not in samples.columns:
            raise ValueError(f"Requested sample_meta_column not present in labels_df: {key!r}")
        if not all((ch.isalnum() or ch == "_") for ch in key):
            raise ValueError(
                f"sample_meta_column {key!r} contains unsupported characters; "
                "use only letters/numbers/underscore."
            )
        meta_cols.append(key)

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

    sample_meta_select_sql = ""
    if meta_cols:
        sample_meta_select_sql = ",\n              " + ",\n              ".join(f"samples.{c} AS {c}" for c in meta_cols)

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
              samples.label
              {sample_meta_select_sql}
              , CAST(windows.bar_date AS DATE) AS bar_date,
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
        sample_meta_columns=[str(x) for x in spec.sample_meta_columns if str(x).strip()],
        mask_current_day_to_open_only=bool(spec.mask_current_day_to_open_only),
        rows=int(len(windows)),
        samples=int(len(samples)),
    )
    _write_meta(meta_path, meta)
    return out_path


@dataclass(frozen=True)
class ContextWindowBuildSpec:
    """
    Build windows with BOTH past and future bars around asof_date (teacher/diagnostics only).

    Time indexing:
    - past window includes asof_date, length=past_window, with t_rel in [-past_window+1 .. 0]
    - future window excludes asof_date, length=future_window, with t_rel in [1 .. future_window]
    - t is a dense 0..(past+future-1) index with asof_date at t=past_window-1
    """

    past_window: int
    future_window: int
    feature_columns: tuple[str, ...] = ()
    sample_meta_columns: tuple[str, ...] = ()
    require_full_window: bool = True


@dataclass(frozen=True)
class ContextWindowBuildMeta:
    source_pool: str | None
    source_pool_mtime_ns: int | None
    source_pool_size: int | None
    past_window: int
    future_window: int
    feature_columns: list[str]
    sample_meta_columns: list[str]
    require_full_window: bool
    rows: int
    samples: int
    lf_config_hash: str | None = None


def build_context_window_bars(
    candidates_df: pd.DataFrame,
    *,
    config: LoaderConfig,
    spec: ContextWindowBuildSpec,
    out_path: Path | None = None,
    source_pool: Path | None = None,
    cal: TradingCalendar | None = None,
    reuse_if_unchanged: bool = True,
    lf_config_hash: str | None = None,
) -> Path:
    """
    Materialize long-form *context* bars for each candidate sample.

    Output schema (long):
    - sample_id, ticker, asof_date, setup
    - label (nullable; preserved if present on candidates_df else NULL)
    - plus any sample_meta_columns (repeated per bar row)
    - bar_date, t, t_rel
    - open, high, low, close, volume, adj_close (+ optional derived features)

    This is intentionally separate from the core `build_windowed_bars()` cache.
    """
    ensure_dirs(config.paths)
    out_path = out_path or (config.paths.derived_dir / "context_windows.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if reuse_if_unchanged and out_path.exists() and meta_path.exists() and source_pool is not None and source_pool.exists():
        st = source_pool.stat()
        prev = _read_existing_meta(meta_path)
        if (
            prev
            and prev.get("source_pool") == str(source_pool)
            and int(prev.get("source_pool_mtime_ns") or -1) == int(st.st_mtime_ns)
            and int(prev.get("source_pool_size") or -1) == int(st.st_size)
            and int(prev.get("past_window") or -1) == int(spec.past_window)
            and int(prev.get("future_window") or -1) == int(spec.future_window)
            and list(prev.get("feature_columns") or []) == list(spec.feature_columns)
            and list(prev.get("sample_meta_columns") or []) == list(spec.sample_meta_columns)
            and bool(prev.get("require_full_window")) == bool(spec.require_full_window)
            and (prev.get("lf_config_hash") or None) == (lf_config_hash or None)
        ):
            return out_path

    if candidates_df.empty:
        raise ValueError("candidates_df is empty")
    required = {"ticker", "asof_date", "setup"}
    missing = required - set(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df missing required columns: {sorted(missing)}")

    cal = cal or TradingCalendar("NYSE")
    past = int(spec.past_window)
    future = int(spec.future_window)
    if past <= 0:
        raise ValueError("past_window must be > 0")
    if future < 0:
        raise ValueError("future_window must be >= 0")

    # One row per sample.
    samples = candidates_df.copy()
    samples["ticker"] = samples["ticker"].astype(str).str.upper()
    samples["asof_date"] = pd.to_datetime(samples["asof_date"], errors="coerce").dt.date
    samples["setup"] = samples["setup"].astype(str)
    if samples["asof_date"].isna().any():
        bad = candidates_df.loc[samples["asof_date"].isna(), "asof_date"].head(10).tolist()
        raise ValueError(f"Could not parse some asof_date values; examples: {bad}")
    if "label" not in samples.columns:
        samples["label"] = pd.NA
    samples["sample_id"] = samples.apply(lambda r: stable_sample_id(r["ticker"], r["asof_date"], r["setup"]), axis=1)

    # De-dupe exact keys (candidate pools can be unioned from multiple sources).
    samples = samples.drop_duplicates(subset=["ticker", "asof_date", "setup"], keep="first").reset_index(drop=True)
    if samples.empty:
        raise ValueError("No samples remain after de-duplication")

    # Build per-asof planned dates (past+future).
    unique_asof = sorted(set(samples["asof_date"].tolist()))
    past_by_asof: dict[date, list[date]] = {}
    future_by_asof: dict[date, list[date]] = {}
    for d in unique_asof:
        pw = last_n_trading_days_ending_at(d, n=past, cal=cal)
        fw = next_n_trading_days_starting_after(d, n=future, cal=cal) if future > 0 else []
        if spec.require_full_window and (len(pw) != past or len(fw) != future):
            past_by_asof[d] = []
            future_by_asof[d] = []
        else:
            past_by_asof[d] = pw
            future_by_asof[d] = fw

    if spec.require_full_window:
        ok_mask = samples["asof_date"].map(
            lambda d: len(past_by_asof.get(d, [])) == past and len(future_by_asof.get(d, [])) == future
        )
        samples = samples.loc[ok_mask].reset_index(drop=True)
        if samples.empty:
            raise ValueError("No samples have a full (past_window, future_window) context window; widen date range.")

    window_rows: list[dict] = []
    for sid, d in zip(samples["sample_id"].tolist(), samples["asof_date"].tolist(), strict=True):
        pw = past_by_asof.get(d) or []
        fw = future_by_asof.get(d) or []
        # Past: chronological, ending at asof_date with t_rel in [-past+1..0]
        for i, bar_d in enumerate(pw):
            t_rel = int(i - (past - 1))
            t = int(i)
            window_rows.append({"sample_id": sid, "bar_date": bar_d, "t": t, "t_rel": t_rel})
        # Future: chronological after asof_date with t_rel in [1..future]
        for j, bar_d in enumerate(fw, start=1):
            t_rel = int(j)
            t = int((past - 1) + j)
            window_rows.append({"sample_id": sid, "bar_date": bar_d, "t": t, "t_rel": t_rel})

    windows = pd.DataFrame(window_rows)
    if windows.empty:
        raise ValueError("Context window planning produced no rows")

    # Determine which raw data files to read.
    needed_dates = sorted(set(windows["bar_date"].tolist()))
    if not needed_dates:
        raise ValueError("No bar dates required")

    lookback = _max_feature_lookback_days(spec.feature_columns)
    if lookback > 0:
        earliest = needed_dates[0]
        extra = last_n_trading_days_ending_at(earliest, n=lookback, cal=cal)
        needed_dates = sorted(set(needed_dates) | set(extra))

    parts = config.paths.list_polygon_grouped_daily_partitions()
    parquet_files: list[Path] = []
    if parts:
        parquet_files = [parts[d] for d in needed_dates if d in parts]
    if not parquet_files:
        tickers = sorted(set(samples["ticker"].astype(str).str.upper().tolist()))
        parquet_files = [config.paths.raw_ticker_parquet(t) for t in tickers if config.paths.raw_ticker_parquet(t).exists()]
    if not parquet_files:
        raise ValueError("No raw OHLCV parquet files found for requested samples/dates")

    for f in spec.feature_columns:
        key = str(f).strip()
        if not key:
            continue
        if key not in _FEATURE_SQL:
            raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")

    # Validate sample meta columns exist and are safe identifiers for DuckDB SELECT.
    meta_cols: list[str] = []
    for c in spec.sample_meta_columns:
        key = str(c).strip()
        if not key:
            continue
        if key not in samples.columns:
            raise ValueError(f"Requested sample_meta_column not present in candidates_df: {key!r}")
        if not all((ch.isalnum() or ch == "_") for ch in key):
            raise ValueError(
                f"sample_meta_column {key!r} contains unsupported characters; use only letters/numbers/underscore."
            )
        meta_cols.append(key)

    feature_exprs = []
    for f in spec.feature_columns:
        key = str(f).strip()
        if key:
            feature_exprs.append(_FEATURE_SQL[key])
    feature_sql = ""
    if feature_exprs:
        feature_sql = ",\n          " + ",\n          ".join(feature_exprs)

    files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"
    tmp_path = Path(str(out_path) + ".tmp")

    con = connect(config)
    con.register("samples", samples)
    con.register("windows", windows)

    feature_select_sql = ""
    if spec.feature_columns:
        parts_sel = []
        for c in spec.feature_columns:
            key = str(c).strip()
            if key:
                parts_sel.append(f"raw.{key} AS {key}")
        if parts_sel:
            feature_select_sql = ",\n              " + ",\n              ".join(parts_sel)

    sample_meta_select_sql = ""
    if meta_cols:
        sample_meta_select_sql = ",\n              " + ",\n              ".join(f"samples.{c} AS {c}" for c in meta_cols)

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
              samples.label
              {sample_meta_select_sql}
              , CAST(windows.bar_date AS DATE) AS bar_date,
              CAST(windows.t AS INTEGER) AS t,
              CAST(windows.t_rel AS INTEGER) AS t_rel,
              raw.open AS open,
              raw.high AS high,
              raw.low AS low,
              raw.close AS close,
              raw.volume AS volume,
              raw.adj_close AS adj_close
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

    st = source_pool.stat() if source_pool is not None and source_pool.exists() else None
    meta = ContextWindowBuildMeta(
        source_pool=str(source_pool) if source_pool is not None else None,
        source_pool_mtime_ns=int(st.st_mtime_ns) if st is not None else None,
        source_pool_size=int(st.st_size) if st is not None else None,
        past_window=int(past),
        future_window=int(future),
        feature_columns=[str(x) for x in spec.feature_columns],
        sample_meta_columns=[str(x) for x in spec.sample_meta_columns if str(x).strip()],
        require_full_window=bool(spec.require_full_window),
        rows=int(len(windows)),
        samples=int(len(samples)),
        lf_config_hash=str(lf_config_hash) if lf_config_hash is not None else None,
    )
    # Write a separate meta struct (not WindowedBuildMeta).
    meta_path.write_text(json.dumps(meta.__dict__, indent=2, default=str) + "\n", encoding="utf-8")
    return out_path

