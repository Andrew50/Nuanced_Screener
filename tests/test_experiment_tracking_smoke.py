from __future__ import annotations

from datetime import date
from pathlib import Path

import json
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


def test_models_train_writes_experiment_manifest(tmp_path: Path) -> None:
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

    models_dir = repo_root / "data" / "models" / "dummy_constant_prior" / "flag"
    runs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    assert runs
    run_dir = runs[-1]

    exp_p = run_dir / "experiment.json"
    meta_p = run_dir / "run_meta.json"
    cfg_p = run_dir / "train_config.json"
    assert exp_p.exists(), "expected experiment.json"
    assert meta_p.exists(), "expected run_meta.json"
    assert cfg_p.exists(), "expected train_config.json"

    exp = json.loads(exp_p.read_text(encoding="utf-8"))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert exp.get("experiment_id")
    assert meta.get("experiment_id") == exp.get("experiment_id")


def test_models_index_includes_experiment_columns(tmp_path: Path) -> None:
    # Reuse the above test pattern quickly.
    repo_root = tmp_path
    _write_raw_ticker_parquet(
        repo_root,
        "AAPL",
        [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5), date(2025, 3, 6), date(2025, 3, 7)],
    )
    labels_csv = repo_root / "labels.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2025-03-06", "setup": "flag", "label": True},
            {"ticker": "AAPL", "date": "2025-03-07", "setup": "flag", "label": False},
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
            "2",
            "--split",
            "time",
            "--train-frac",
            "0.5",
            "--val-frac",
            "0.5",
            "--test-frac",
            "0.0",
            "--duckdb-threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output

    # Index and ensure experiment columns exist.
    out = repo_root / "data" / "models" / "_index.parquet"
    res2 = runner.invoke(app, ["models", "index", "--repo-root", str(repo_root), "--out", str(out)])
    assert res2.exit_code == 0, res2.output
    df = pd.read_parquet(out)
    assert "meta.experiment_id" in df.columns or "exp.experiment_id" in df.columns

