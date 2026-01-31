from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import importlib.util

import pandas as pd
import pytest
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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_cli_models_train_torch_ssl_head_student_smoke(tmp_path: Path) -> None:
    # Build minimal raw history.
    repo_root = tmp_path
    days = [date(2025, 3, 3) + timedelta(days=i) for i in range(0, 40)]
    # Filter to weekdays only (avoid non-trading date resolution headaches).
    days = [d for d in days if d.weekday() < 5]
    _write_raw_ticker_parquet(repo_root, "AAPL", days)

    # Create minimal encoder artifact.
    import torch

    from screener_loader.ssl.schema import SSLSchema, write_schema
    from screener_loader.ssl.tcn import TCNConfig, TCNEncoder

    encoder_dir = repo_root / "encoder_artifact"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    enc = TCNEncoder(TCNConfig(in_features=6, d_model=16, num_blocks=2, kernel_size=3, dropout=0.1))
    torch.save(enc.state_dict(), encoder_dir / "encoder.pt")
    schema = SSLSchema(
        feature_names=["r", "rr", "u", "l", "body", "logv"],
        normalization="per_window_zscore",
        window_max=8,
        crop_lengths=[8],
        encoder={"type": "tcn", "in_features": 6, "d_model": 16, "num_blocks": 2, "kernel_size": 3, "dropout": 0.1},
        pretrain_mask_current_day_to_open_only=False,
    )
    write_schema(encoder_dir / "schema.json", schema)

    # Labels with pseudo columns.
    labels_csv = repo_root / "labels.csv"
    rows = []
    for d in days[10:30]:
        rows.append(
            {
                "ticker": "AAPL",
                "date": d.isoformat(),
                "setup": "flag",
                "label": bool((d.day % 2) == 0),
                "p_label": 0.8 if (d.day % 2) == 0 else 0.2,
                "weight": 0.9,
            }
        )
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

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
            "torch_ssl_head_student",
            "--setup",
            "flag",
            "--window-size",
            "8",
            "--split",
            "time",
            "--train-frac",
            "0.7",
            "--val-frac",
            "0.15",
            "--test-frac",
            "0.15",
            "--duckdb-threads",
            "1",
            "--encoder-dir",
            str(encoder_dir),
            "--head-epochs",
            "1",
            "--head-batch-size",
            "16",
            "--sample-meta-column",
            "p_label",
            "--sample-meta-column",
            "weight",
            "--extra-meta-column",
            "p_label",
            "--extra-meta-column",
            "weight",
            "--calibration-labels-csv",
            str(labels_csv),
        ],
    )
    assert res.exit_code == 0, res.output

    # Should have written at least one metrics.json under data/models.
    models_dir = repo_root / "data" / "models" / "torch_ssl_head_student" / "flag"
    assert models_dir.exists()
    metrics = list(models_dir.glob("**/metrics.json"))
    assert metrics, f"expected metrics.json under {models_dir}, got none. output={res.output}"

