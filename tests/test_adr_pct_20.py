from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.config import LoaderConfig
from screener_loader.derived import rebuild_last_n_bars
from screener_loader.paths import ensure_dirs


def _raw_bars(n: int = 40) -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    close = 100.0
    for i in range(n):
        d = start + timedelta(days=i)
        rows.append(
            {
                "ticker": "AAA",
                "date": d,
                "open": close,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000,
                "adj_close": close,
            }
        )
        close += 1.0
    return pd.DataFrame(rows)


def test_adr_pct_20_matches_frozen_formula(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=40, feature_columns=())
    ensure_dirs(cfg.paths)
    _raw_bars(40).to_parquet(cfg.paths.raw_ticker_parquet("AAA"), index=False)
    df = pd.read_parquet(rebuild_last_n_bars(cfg)).sort_values("date").reset_index(drop=True)
    assert "adr_pct_20" in df.columns
    assert "dollar_vol_avg_20" in df.columns

    prev = df["close"].shift(1)
    daily = (df["high"] - df["low"]) / prev
    expected = float(daily.rolling(20, min_periods=20).mean().iloc[-1])
    got = float(df.iloc[-1]["adr_pct_20"])
    assert got == pytest.approx(expected, rel=1e-9)
    # Need 20 defined (H-L)/prior_close terms, so early rows are null.
    assert df["adr_pct_20"].isna().sum() >= 1
