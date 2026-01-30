from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.raw_update import merge_write_ticker_parquet


def _raw_df(ticker: str, rows: list[tuple[date, float]]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    out = []
    for d, close in rows:
        out.append(
            {
                "ticker": ticker,
                "date": d,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100,
                "adj_close": close,
                "source": "test",
                "asof_ts": now,
            }
        )
    return pd.DataFrame(out)


def test_merge_dedup_prefers_new(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)

    ticker = "AAA"
    raw_path = cfg.paths.raw_ticker_parquet(ticker)

    existing = _raw_df(
        ticker,
        [
            (date(2020, 1, 1), 10.0),
            (date(2020, 1, 2), 11.0),
            (date(2020, 1, 3), 12.0),
        ],
    )
    existing.to_parquet(raw_path, index=False)

    new = _raw_df(
        ticker,
        [
            (date(2020, 1, 3), 99.0),  # overlap should replace
            (date(2020, 1, 4), 13.0),
        ],
    )

    res = merge_write_ticker_parquet(cfg, ticker=ticker, new_bars=new, full_refresh=False)
    assert res.status == "updated"
    assert raw_path.exists()

    merged = pd.read_parquet(raw_path).sort_values("date").reset_index(drop=True)
    assert len(merged) == 4
    assert float(merged.loc[merged["date"] == date(2020, 1, 3), "close"].iloc[0]) == 99.0

