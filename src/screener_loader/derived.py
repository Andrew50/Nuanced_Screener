from __future__ import annotations

from pathlib import Path

from rich import print

from .config import LoaderConfig
from .duckdb_utils import connect
from .feature_sql import (
    DAILY_RANGE_PCT_SQL,
    feature_sql_fragments,
    max_feature_lookback_days,
    merge_filter_features,
)
from .paths import atomic_replace, ensure_dirs


def _resolved_feature_columns(config: LoaderConfig) -> tuple[str, ...]:
    return merge_filter_features(config.feature_columns)


def _feature_sql_clause(feature_columns: tuple[str, ...]) -> str:
    exprs = feature_sql_fragments(feature_columns)
    if not exprs:
        return ""
    return ",\n            " + ",\n            ".join(exprs)


def _empty_typed_nulls(feature_columns: tuple[str, ...]) -> list[str]:
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
    for c in feature_columns:
        typed_nulls.append(f"CAST(NULL AS DOUBLE) AS {c}")
    typed_nulls.append("CAST(NULL AS BIGINT) AS rn")
    return typed_nulls


def _copy_last_n_sql(source_rel: str, *, window_size: int, feature_columns: tuple[str, ...]) -> str:
    feature_sql = _feature_sql_clause(feature_columns)
    return f"""
        COPY (
          WITH src AS (
            SELECT
              ticker,
              CAST(date AS DATE) AS date,
              open,
              high,
              low,
              close,
              volume,
              adj_close
            FROM {source_rel}
          ),
          staged AS (
            SELECT
              *,
              {DAILY_RANGE_PCT_SQL}
            FROM src
          ),
          base AS (
            SELECT
              ticker,
              date,
              open,
              high,
              low,
              close,
              volume,
              adj_close
              {feature_sql}
            FROM staged
          ),
          ranked AS (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM base
          )
          SELECT *
          FROM ranked
          WHERE rn <= {int(window_size)}
        )
    """


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

    feature_columns = _resolved_feature_columns(config)
    if not parquet_files:
        # Create an empty Parquet with a stable schema so downstream queries fail less often.
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        typed_nulls = _empty_typed_nulls(feature_columns)

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

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"
    sql = _copy_last_n_sql(f"read_parquet({files_sql})", window_size=window_size, feature_columns=feature_columns)
    con.execute(
        f"""
        {sql}
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
    feature_columns = _resolved_feature_columns(config)
    lookback = max_feature_lookback_days(feature_columns)
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
    feature_columns = _resolved_feature_columns(config)
    raw_files = list(config.paths.raw_dir.glob("*.parquet"))
    if not raw_files:
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        typed_nulls = _empty_typed_nulls(feature_columns)
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

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    sql = _copy_last_n_sql(f"read_parquet('{raw_glob}')", window_size=window_size, feature_columns=feature_columns)
    con.execute(
        f"""
        {sql}
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, out_path)
    print(f"[green]Derived[/green] wrote {out_path}")
    return out_path

