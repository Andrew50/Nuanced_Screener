from __future__ import annotations

from pathlib import Path

import duckdb

from .config import LoaderConfig


def connect(config: LoaderConfig) -> duckdb.DuckDBPyConnection:
    # In-memory engine is fine; we’re using Parquet as durable storage.
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false;")
    con.execute(f"PRAGMA threads={int(config.duckdb_threads)};")
    return con


def parquet_max_date(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> object | None:
    # Returns a Python date (duckdb maps DATE to datetime.date) or None.
    if not parquet_path.exists():
        return None
    return con.execute(
        "SELECT max(date) AS max_date FROM read_parquet(?)",
        [str(parquet_path)],
    ).fetchone()[0]

