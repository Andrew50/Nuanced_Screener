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


def test_cli_models_train_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    # Bars for March 2025 weekdays (no major holidays in this range).
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
    ]
    _write_raw_ticker_parquet(repo_root, "AAPL", days)

    labels_csv = repo_root / "labels.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2025-03-06", "setup": "flag", "label": True},
            {"ticker": "AAPL", "date": "2025-03-07", "setup": "flag", "label": False},
            {"ticker": "AAPL", "date": "2025-03-10", "setup": "flag", "label": True},
            {"ticker": "AAPL", "date": "2025-03-11", "setup": "flag", "label": False},
            {"ticker": "AAPL", "date": "2025-03-12", "setup": "flag", "label": True},
            {"ticker": "AAPL", "date": "2025-03-13", "setup": "flag", "label": False},
            {"ticker": "AAPL", "date": "2025-03-14", "setup": "flag", "label": True},
        ]
    ).to_csv(labels_csv, index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "models",
            "train",
            "--repo-root",
            str(repo_root),
            "--labels-csv",
            str(labels_csv),
            "--model-type",
            "dummy_constant_prior",
            "--setup",
            "flag",
            "--window-size",
            "3",
            "--split",
            "time",
            "--train-frac",
            "0.6",
            "--val-frac",
            "0.2",
            "--test-frac",
            "0.2",
            "--duckdb-threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output

    # Should have written at least one metrics.json under data/models.
    models_dir = repo_root / "data" / "models" / "dummy_constant_prior" / "flag"
    assert models_dir.exists()
    metrics = list(models_dir.glob("**/metrics.json"))
    assert metrics, f"expected metrics.json under {models_dir}, got none. output={res.output}"

