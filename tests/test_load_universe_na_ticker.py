from __future__ import annotations

from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.universe import load_universe


def test_load_universe_preserves_na_ticker(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)

    # Pandas read_csv defaults would treat "NA" as missing; ensure we preserve it.
    (cfg.paths.meta_dir / "tickers.csv").write_text(
        "ticker,name,exchange,is_etf,is_test_issue,source_file\n"
        "NA,NA Corp,NYSE,False,False,otherlisted.txt\n"
        "AAPL,Apple Inc,NASDAQ,False,False,nasdaqlisted.txt\n"
    )

    df = load_universe(cfg)
    assert "NA" in set(df["ticker"])
    assert "AAPL" in set(df["ticker"])

