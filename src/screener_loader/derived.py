from __future__ import annotations

from pathlib import Path

from rich import print

from .config import LoaderConfig
from .duckdb_utils import connect
from .paths import atomic_replace, ensure_dirs


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
    # Max lag/window required for computing the feature at a given row.
    "ret_1d": 1,
    "ret_5d": 5,
    "ret_21d": 21,
    "vol_avg_20": 19,
    "dollar_vol_avg_20": 19,
    "range_pct": 0,
}


def _max_feature_lookback_days(feature_columns: tuple[str, ...]) -> int:
    m = 0
    for f in feature_columns:
        key = f.strip()
        if not key:
            continue
        m = max(m, _FEATURE_LOOKBACK_DAYS.get(key, 0))
    return m


def _sql_quote_path(p: Path) -> str:
    # DuckDB SQL single-quoted literal
    return "'" + str(p).replace("'", "''") + "'"


def rebuild_last_n_bars_from_files(config: LoaderConfig, parquet_files: list[Path]) -> Path:
    """
    Build last-N bars from an explicit list of Parquet files (e.g. date partitions).
    """
    ensure_dirs(config.paths)
    out_path = config.paths.last_100_bars_parquet
    tmp_path = Path(str(out_path) + ".tmp")

    con = connect(config)

    if not parquet_files:
        # Create an empty Parquet with a stable schema so downstream queries fail less often.
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        feature_cols = []
        for f in config.feature_columns:
            key = f.strip()
            if not key:
                continue
            if key not in _FEATURE_SQL:
                raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")
            feature_cols.append(key)

        typed_nulls = [
            "CAST(NULL AS VARCHAR) AS ticker",
            "CAST(NULL AS DATE) AS date",
            "CAST(NULL AS DOUBLE) AS open",
            "CAST(NULL AS DOUBLE) AS high",
            "CAST(NULL AS DOUBLE) AS low",
            "CAST(NULL AS DOUBLE) AS close",
            "CAST(NULL AS BIGINT) AS volume",
            "CAST(NULL AS DOUBLE) AS adj_close",
        ]
        for c in feature_cols:
            typed_nulls.append(f"CAST(NULL AS DOUBLE) AS {c}")
        typed_nulls.append("CAST(NULL AS BIGINT) AS rn")

        con.execute(
            f"""
            COPY (
              SELECT
                {", ".join(typed_nulls)}
              WHERE FALSE
            )
            TO '{tmp_path.as_posix()}'
            (FORMAT PARQUET, CODEC 'ZSTD');
            """
        )
        atomic_replace(tmp_path, out_path)
        print(f"[yellow]Derived[/yellow] wrote empty {out_path}")
        return out_path

    feature_exprs = []
    for f in config.feature_columns:
        key = f.strip()
        if not key:
            continue
        if key not in _FEATURE_SQL:
            raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")
        feature_exprs.append(_FEATURE_SQL[key])

    feature_sql = ""
    if feature_exprs:
        feature_sql = ",\n            " + ",\n            ".join(feature_exprs)

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"

    con.execute(
        f"""
        COPY (
          WITH base AS (
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
          ranked AS (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM base
          )
          SELECT *
          FROM ranked
          WHERE rn <= {window_size}
        )
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, out_path)
    print(f"[green]Derived[/green] wrote {out_path}")
    return out_path


def rebuild_last_n_bars_from_polygon_date_partitions(config: LoaderConfig) -> Path:
    """
    Efficient derived rebuild for Polygon date-partitioned raw data:
    read only the most recent K partitions where K ~= window_size + max feature lookback.
    """
    parts = config.paths.list_polygon_grouped_daily_partitions()
    if not parts:
        return rebuild_last_n_bars_from_files(config, [])

    dates_sorted = sorted(parts.keys())
    lookback = _max_feature_lookback_days(config.feature_columns)
    k = int(config.window_size) + int(lookback) + 2
    if k <= 0:
        k = 1
    recent_dates = dates_sorted[-k:]
    files = [parts[d] for d in recent_dates if d in parts]
    return rebuild_last_n_bars_from_files(config, files)


def rebuild_last_n_bars(config: LoaderConfig) -> Path:
    """
    Rebuild consolidated derived dataset containing last `window_size` bars per ticker.
    This is the primary screener input for market-wide scans.
    """
    ensure_dirs(config.paths)
    raw_glob = (config.paths.raw_dir / "*.parquet").as_posix()
    out_path = config.paths.last_100_bars_parquet
    tmp_path = Path(str(out_path) + ".tmp")

    con = connect(config)

    raw_files = list(config.paths.raw_dir.glob("*.parquet"))
    if not raw_files:
        # Create an empty Parquet with a stable schema so downstream queries fail less often.
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        feature_cols = []
        for f in config.feature_columns:
            key = f.strip()
            if not key:
                continue
            if key not in _FEATURE_SQL:
                raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")
            feature_cols.append(key)

        # Build a typed empty relation.
        typed_nulls = [
            "CAST(NULL AS VARCHAR) AS ticker",
            "CAST(NULL AS DATE) AS date",
            "CAST(NULL AS DOUBLE) AS open",
            "CAST(NULL AS DOUBLE) AS high",
            "CAST(NULL AS DOUBLE) AS low",
            "CAST(NULL AS DOUBLE) AS close",
            "CAST(NULL AS BIGINT) AS volume",
            "CAST(NULL AS DOUBLE) AS adj_close",
        ]
        for c in feature_cols:
            typed_nulls.append(f"CAST(NULL AS DOUBLE) AS {c}")
        typed_nulls.append("CAST(NULL AS BIGINT) AS rn")

        con.execute(
            f"""
            COPY (
              SELECT
                {", ".join(typed_nulls)}
              WHERE FALSE
            )
            TO '{tmp_path.as_posix()}'
            (FORMAT PARQUET, CODEC 'ZSTD');
            """
        )
        atomic_replace(tmp_path, out_path)
        print(f"[yellow]Derived[/yellow] no raw files; wrote empty {out_path}")
        return out_path

    feature_exprs = []
    for f in config.feature_columns:
        key = f.strip()
        if not key:
            continue
        if key not in _FEATURE_SQL:
            raise ValueError(f"Unknown feature column: {key}. Known: {sorted(_FEATURE_SQL)}")
        feature_exprs.append(_FEATURE_SQL[key])

    feature_sql = ""
    if feature_exprs:
        feature_sql = ",\n            " + ",\n            ".join(feature_exprs)

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    # Build a compact last-N table for scans.
    con.execute(
        f"""
        COPY (
          WITH base AS (
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
            FROM read_parquet('{raw_glob}')
          ),
          ranked AS (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM base
          )
          SELECT *
          FROM ranked
          WHERE rn <= {window_size}
        )
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, out_path)
    print(f"[green]Derived[/green] wrote {out_path}")
    return out_path

