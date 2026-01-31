from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from screener_loader.calendar_utils import TradingCalendar
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


def test_ssl_pretrain_and_finetune_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    repo_root = tmp_path
    cal = TradingCalendar("NYSE")

    # Build enough real NYSE trading days for a 96-bar window.
    days = cal.valid_trading_days(date(2025, 1, 2), date(2025, 7, 31))
    assert len(days) >= 130
    days = days[:130]

    _write_raw_ticker_parquet(repo_root, "AAPL", days)

    # Labels CSV (also used as labels_only ticker source for pretraining).
    labels_csv = repo_root / "labels.csv"
    # Use a small set of asof dates late enough to have full windows.
    label_days = days[-12:]
    pd.DataFrame(
        [{"ticker": "AAPL", "date": d.isoformat(), "setup": "flag", "label": bool(i % 2 == 0)} for i, d in enumerate(label_days)]
    ).to_csv(labels_csv, index=False)

    # Pretrain: limit steps so the test stays fast.
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "models",
            "pretrain",
            "--repo-root",
            str(repo_root),
            "--ticker-source",
            "labels_only",
            "--labels-csv",
            str(labels_csv),
            "--num-samples",
            "16",
            "--window-max",
            "96",
            "--crop-length",
            "64",
            "--crop-length",
            "96",
            "--date-start",
            days[-40].isoformat(),
            "--date-end",
            days[-1].isoformat(),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--chunk-size",
            "16",
            "--max-steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output

    pretrain_root = repo_root / "data" / "models" / "ssl_tcn_masked_pretrain" / "_pretrain"
    encoder_dirs = list(pretrain_root.glob("*"))
    assert encoder_dirs, f"expected pretrain run dirs under {pretrain_root}"
    encoder_dir = sorted(encoder_dirs)[-1]
    assert (encoder_dir / "encoder.pt").exists()
    assert (encoder_dir / "schema.json").exists()

    # Forward-pass check for L=64 and L=96.
    from screener_loader.ssl.schema import read_schema
    from screener_loader.ssl.tcn import TCNConfig, TCNEncoder

    schema = read_schema(encoder_dir / "schema.json")
    enc_info = schema.encoder
    cfg = TCNConfig(
        in_features=int(enc_info["in_features"]),
        d_model=int(enc_info["d_model"]),
        num_blocks=int(enc_info["num_blocks"]),
        kernel_size=int(enc_info["kernel_size"]),
        dropout=float(enc_info["dropout"]),
    )
    enc = TCNEncoder(cfg)
    sd = torch.load(encoder_dir / "encoder.pt", map_location="cpu")
    enc.load_state_dict(sd)
    enc.eval()
    x64 = torch.randn(2, 64, cfg.in_features)
    x96 = torch.randn(2, 96, cfg.in_features)
    m64 = torch.ones(2, 64, dtype=torch.bool)
    m96 = torch.ones(2, 96, dtype=torch.bool)
    e64 = enc.encode(x64, mask_time=m64)
    e96 = enc.encode(x96, mask_time=m96)
    assert tuple(e64.shape) == (2, cfg.d_model)
    assert tuple(e96.shape) == (2, cfg.d_model)

    # Finetune: train a small head from labels using the pretrained encoder.
    res2 = runner.invoke(
        app,
        [
            "models",
            "train",
            "--repo-root",
            str(repo_root),
            "--labels-csv",
            str(labels_csv),
            "--model-type",
            "ssl_tcn_classifier",
            "--setup",
            "flag",
            "--window-size",
            "96",
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
            "--encoder-dir",
            str(encoder_dir),
            "--head-epochs",
            "1",
            "--head-hidden-dim",
            "16",
            "--head-lr",
            "0.01",
        ],
    )
    assert res2.exit_code == 0, res2.output

    out_dir = repo_root / "data" / "models" / "ssl_tcn_classifier" / "flag"
    metrics = list(out_dir.glob("**/metrics.json"))
    assert metrics, f"expected metrics.json under {out_dir}, got none. output={res2.output}"

