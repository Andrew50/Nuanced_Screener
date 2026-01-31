from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from screener_loader.cli import app


def _write_raw_ticker_parquet(repo_root: Path, ticker: str, dates: list[date]) -> None:
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "open": 10.0 + i,
                "high": 11.0 + i,
                "low": 9.0 + i,
                "close": 10.5 + i,
                "volume": 1000 + i,
                "adj_close": 10.5 + i,
                "source": "test",
                "asof_ts": pd.Timestamp("2026-01-01T00:00:00Z"),
            }
        )
    df = pd.DataFrame(rows)
    out = repo_root / "data" / "raw" / f"{ticker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def test_cli_weak_generate_pseudo_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    days = [
        date(2025, 3, 3),
        date(2025, 3, 4),
        date(2025, 3, 5),
        date(2025, 3, 6),
        date(2025, 3, 7),
        date(2025, 3, 10),
        date(2025, 3, 11),
        date(2025, 3, 12),
        date(2025, 3, 13),
        date(2025, 3, 14),
        date(2025, 3, 17),
    ]
    _write_raw_ticker_parquet(repo_root, "AAPL", days)

    runner = CliRunner()
    pool_path = repo_root / "candidate_pool.parquet"
    out_dir = repo_root / "weak_out"
    res1 = runner.invoke(
        app,
        [
            "weak",
            "build-candidate-pool",
            "--repo-root",
            str(repo_root),
            "--setup",
            "flag",
            "--start-date",
            "2025-03-05",
            "--end-date",
            "2025-03-12",
            "--max-candidates",
            "10",
            "--out",
            str(pool_path),
            "--duckdb-threads",
            "1",
        ],
    )
    assert res1.exit_code == 0, res1.output
    assert pool_path.exists()

    res2 = runner.invoke(
        app,
        [
            "weak",
            "generate-pseudo",
            "--repo-root",
            str(repo_root),
            "--candidate-pool",
            str(pool_path),
            "--setup",
            "flag",
            "--past-window",
            "5",
            "--future-window",
            "2",
            "--combine",
            "majority",
            "--out-dir",
            str(out_dir),
            "--duckdb-threads",
            "1",
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert (out_dir / "pseudo_labels.parquet").exists()
    assert (out_dir / "pseudo_labels.csv").exists()
    assert (out_dir / "lf_matrix.parquet").exists()

