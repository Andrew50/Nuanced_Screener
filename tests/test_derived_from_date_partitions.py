from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.derived import rebuild_last_n_bars_from_polygon_date_partitions
from screener_loader.paths import ensure_dirs


def _write_partition(root: Path, d: date, rows: list[dict]) -> None:
    cfg = LoaderConfig(repo_root=root)
    ensure_dirs(cfg.paths)
    out = cfg.paths.polygon_grouped_daily_parquet(d)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)


def test_rebuild_last_n_from_polygon_date_partitions(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=2, feature_columns=("ret_1d",))
    ensure_dirs(cfg.paths)

    d1 = date(2026, 1, 1)
    d2 = date(2026, 1, 2)
    d3 = date(2026, 1, 3)

    for d, close_a, close_b in [(d1, 10.0, 20.0), (d2, 11.0, 19.0), (d3, 12.0, 18.0)]:
        _write_partition(
            tmp_path,
            d,
            [
                {"ticker": "AAA", "date": d, "open": close_a, "high": close_a, "low": close_a, "close": close_a, "volume": 100, "adj_close": None},
                {"ticker": "BBB", "date": d, "open": close_b, "high": close_b, "low": close_b, "close": close_b, "volume": 200, "adj_close": None},
            ],
        )

    out = rebuild_last_n_bars_from_polygon_date_partitions(cfg)
    assert out.exists()

    con = duckdb.connect(database=":memory:")
    df = con.execute(
        "SELECT ticker, count(*) AS n, max(rn) AS max_rn, min(rn) AS min_rn FROM read_parquet(?) GROUP BY ticker",
        [str(out)],
    ).df()
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert set(df["n"]) == {2}
    assert set(df["min_rn"]) == {1}
    assert set(df["max_rn"]) == {2}

