from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.derived import rebuild_last_n_bars
from screener_loader.paths import ensure_dirs


def _make_raw(ticker: str, start: date, days: int) -> pd.DataFrame:
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        close = float(100 + i)
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + i,
                "adj_close": close,
                "source": "test",
                "asof_ts": pd.Timestamp("2026-01-29T00:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def test_rebuild_last_100_per_ticker(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=100, feature_columns=("ret_1d", "ret_21d"))
    ensure_dirs(cfg.paths)

    start = date(2020, 1, 1)
    _make_raw("AAA", start, 150).to_parquet(cfg.paths.raw_ticker_parquet("AAA"), index=False)
    _make_raw("BBB", start, 120).to_parquet(cfg.paths.raw_ticker_parquet("BBB"), index=False)

    out = rebuild_last_n_bars(cfg)
    assert out.exists()

    con = duckdb.connect(database=":memory:")
    df = con.execute("SELECT ticker, count(*) AS n, max(rn) AS max_rn, min(rn) AS min_rn FROM read_parquet(?) GROUP BY ticker", [str(out)]).df()
    counts = dict(zip(df["ticker"], df["n"]))
    assert counts["AAA"] == 100
    assert counts["BBB"] == 100
    assert int(df.set_index("ticker").loc["AAA", "min_rn"]) == 1
    assert int(df.set_index("ticker").loc["AAA", "max_rn"]) == 100

    latest = con.execute("SELECT ticker, max(date) AS max_date FROM read_parquet(?) WHERE rn = 1 GROUP BY ticker", [str(out)]).df()
    assert latest.set_index("ticker").loc["AAA", "max_date"].date() == start + timedelta(days=149)
    assert latest.set_index("ticker").loc["BBB", "max_date"].date() == start + timedelta(days=119)


def test_rebuild_empty_raw_writes_empty_parquet(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=100, feature_columns=("ret_1d",))
    ensure_dirs(cfg.paths)
    out = rebuild_last_n_bars(cfg)
    assert out.exists()

