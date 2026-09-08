from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.charts import load_ohlcv_window, render_chart_png
from screener_loader.setups.spec import ChartStyle


matplotlib = pytest.importorskip("matplotlib")


def test_chart_png_is_deterministic(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    start = date(2025, 1, 2)
    rows = []
    for i in range(30):
        d = start + timedelta(days=i)
        px = 100.0 + i
        rows.append(
            {
                "ticker": "NVDA",
                "date": d,
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.2,
                "volume": 1_000_000 + i,
                "adj_close": px,
            }
        )
    pd.DataFrame(rows).to_parquet(cfg.paths.raw_ticker_parquet("NVDA"), index=False)
    asof = start + timedelta(days=29)
    df = load_ohlcv_window(cfg, "NVDA", asof, 20)
    assert len(df) == 20
    a = render_chart_png(df, ticker="NVDA", asof_date=asof, style=ChartStyle())
    b = render_chart_png(df, ticker="NVDA", asof_date=asof, style=ChartStyle())
    assert a == b
    assert a[:8] == b"\x89PNG\r\n\x1a\n"
