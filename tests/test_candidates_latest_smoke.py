from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from screener_loader.candidates import CandidateSpec, propose_latest_candidates
from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs


def test_propose_latest_candidates_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    cfg = LoaderConfig(repo_root=repo_root, window_size=20, duckdb_threads=1)
    ensure_dirs(cfg.paths)

    # Build a synthetic last_100_bars.parquet with exactly T rows for one ticker.
    T = int(cfg.window_size)
    start = date(2025, 1, 2)
    days = [start + timedelta(days=i) for i in range(T)]

    rows = []
    close = 100.0
    for i, d in enumerate(days):
        # Simple drift.
        close = close + (0.1 if i < T - 1 else 0.0)
        prev_close = close
        open_ = prev_close
        high = open_ + 1.0
        low = open_ - 1.0

        # Last bar: big gap open + volume spike.
        if i == T - 1:
            open_ = prev_close * 1.10
            high = open_ * 1.02
            low = open_ * 0.98
            close = open_ * 1.01
            vol = 5000
        else:
            vol = 1000

        rows.append(
            {
                "ticker": "AAPL",
                "date": d,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": int(vol),
                "adj_close": float(close),
                # rn is 1 for most recent in derived, but candidate generator sorts by date anyway.
                "rn": int(T - i),
            }
        )

    df = pd.DataFrame(rows)
    out_path = cfg.paths.last_100_bars_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    cand = propose_latest_candidates(cfg, spec=CandidateSpec(window_size=T, end_lookback_bars=5))
    assert not cand.empty
    assert {"ticker", "asof_date", "setup", "label", "cand_t_start", "cand_t_end"} <= set(cand.columns)

    # Should have no duplicates on the repo uniqueness key.
    assert not cand.duplicated(subset=["ticker", "asof_date", "setup"]).any()

    # Latest-only, but cand_t_end is within last k bars.
    assert (cand["cand_t_end"] >= (T - 1 - 5)).all()
    assert (cand["cand_t_end"] <= (T - 1)).all()

