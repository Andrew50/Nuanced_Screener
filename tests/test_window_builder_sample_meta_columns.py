from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
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
    pd.DataFrame(rows).to_parquet(repo_root / "data" / "raw" / f"{ticker}.parquet", index=False)


def test_window_builder_copies_sample_meta_columns(tmp_path: Path) -> None:
    repo_root = tmp_path
    cfg = LoaderConfig(repo_root=repo_root, window_size=3, duckdb_threads=1)
    ensure_dirs(cfg.paths)

    trading_days = [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5), date(2025, 3, 6), date(2025, 3, 7)]
    _write_raw_ticker_parquet(repo_root, "AAPL", trading_days)

    labels_df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "asof_date": date(2025, 3, 6),
                "setup": "flag:impulse_consolidation:L8:test",
                "label": False,
                "cand_t_start": 0,
                "cand_t_end": 2,
            }
        ]
    )

    out_path = build_windowed_bars(
        labels_df,
        config=cfg,
        spec=WindowedBuildSpec(
            window_size=3,
            feature_columns=(),
            sample_meta_columns=("cand_t_start", "cand_t_end"),
            mask_current_day_to_open_only=True,
            require_full_window=True,
        ),
        out_path=repo_root / "windowed_with_meta.parquet",
        source_csv=None,
        reuse_if_unchanged=False,
    )

    w = pd.read_parquet(out_path)
    assert "cand_t_start" in w.columns
    assert "cand_t_end" in w.columns

    sid = stable_sample_id("AAPL", date(2025, 3, 6), "flag:impulse_consolidation:L8:test")
    sub = w[w["sample_id"] == sid].sort_values("t")
    assert len(sub) == 3
    assert sub["cand_t_start"].nunique() == 1
    assert sub["cand_t_end"].nunique() == 1
    assert int(sub["cand_t_start"].iloc[0]) == 0
    assert int(sub["cand_t_end"].iloc[0]) == 2

