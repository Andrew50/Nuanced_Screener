from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.labels import load_labels_csv
from screener_loader.paths import ensure_dirs
from screener_loader.windowed_dataset import WindowedBuildSpec, build_windowed_bars, stable_sample_id


def _write_raw_ticker_parquet(repo_root: Path, ticker: str, dates: list[date]) -> None:
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
                "adj_close": 100.5 + i,
                "source": "test",
                "asof_ts": pd.Timestamp("2026-01-01T00:00:00Z"),
            }
        )
    df = pd.DataFrame(rows)
    out = repo_root / "data" / "raw" / f"{ticker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def test_window_builder_masks_current_day_to_open_only(tmp_path: Path) -> None:
    repo_root = tmp_path
    cfg = LoaderConfig(repo_root=repo_root, window_size=3, duckdb_threads=1)
    ensure_dirs(cfg.paths)

    # Choose trading days that avoid holidays.
    trading_days = [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5), date(2025, 3, 6), date(2025, 3, 7)]
    _write_raw_ticker_parquet(repo_root, "AAPL", trading_days)

    labels_csv = repo_root / "labels.csv"
    pd.DataFrame(
        [{"ticker": "AAPL", "date": "2025-03-06", "setup": "flag", "label": True}]
    ).to_csv(labels_csv, index=False)
    labels_df = load_labels_csv(labels_csv).df

    out_path = build_windowed_bars(
        labels_df,
        config=cfg,
        spec=WindowedBuildSpec(window_size=3, feature_columns=(), mask_current_day_to_open_only=True),
        out_path=repo_root / "windowed.parquet",
        source_csv=labels_csv,
        reuse_if_unchanged=False,
    )

    w = pd.read_parquet(out_path)
    sid = stable_sample_id("AAPL", date(2025, 3, 6), "flag")
    sub = w[w["sample_id"] == sid].sort_values("t")
    assert len(sub) == 3
    assert sub.iloc[-1]["bar_date"] == date(2025, 3, 6)
    # Open is available
    assert pd.notna(sub.iloc[-1]["open"])
    # But other OHLCV are masked
    assert pd.isna(sub.iloc[-1]["high"])
    assert pd.isna(sub.iloc[-1]["low"])
    assert pd.isna(sub.iloc[-1]["close"])
    assert pd.isna(sub.iloc[-1]["volume"])
    assert pd.isna(sub.iloc[-1]["adj_close"])

