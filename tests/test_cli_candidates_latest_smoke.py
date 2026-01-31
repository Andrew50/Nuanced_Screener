from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from screener_loader.cli import app
from screener_loader.paths import ensure_dirs
from screener_loader.config import LoaderConfig


def test_cli_candidates_latest_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    cfg = LoaderConfig(repo_root=repo_root, window_size=20, duckdb_threads=1)
    ensure_dirs(cfg.paths)

    # Minimal derived last_100_bars.parquet for one ticker.
    T = int(cfg.window_size)
    start = date(2025, 1, 2)
    days = [start + timedelta(days=i) for i in range(T)]
    rows = []
    for i, d in enumerate(days):
        open_ = 100.0 + i
        close = open_
        vol = 1000 if i < T - 1 else 5000
        # force a gap at the last bar
        if i == T - 1:
            open_ = 110.0
            close = 111.0
        rows.append(
            {
                "ticker": "AAPL",
                "date": d,
                "open": float(open_),
                "high": float(open_ + 1),
                "low": float(open_ - 1),
                "close": float(close),
                "volume": int(vol),
                "adj_close": float(close),
                "rn": int(T - i),
            }
        )
    pd.DataFrame(rows).to_parquet(cfg.paths.last_100_bars_parquet, index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "candidates",
            "latest",
            "--repo-root",
            str(repo_root),
            "--window-size",
            "20",
            "--end-lookback-bars",
            "3",
            "--limit",
            "0",
            "--duckdb-threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    out_path = repo_root / "data" / "derived" / "candidates_latest.parquet"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert not df.empty

