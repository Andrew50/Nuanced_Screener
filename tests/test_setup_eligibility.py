from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.eligibility import compute_eligibility
from screener_loader.setups.service import SetupService, update_spec_fields
from screener_loader.setups.spec import MarketCapUnavailableError, SetupFilters


def _latest_row(ticker: str, close: float, dollar_vol: float, adr: float) -> dict:
    return {
        "ticker": ticker,
        "date": date(2026, 6, 18),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000_000,
        "adj_close": close,
        "dollar_vol_avg_20": dollar_vol,
        "adr_pct_20": adr,
        "rn": 1,
    }


def test_eligibility_union(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    svc = SetupService(cfg.paths)
    svc.save_global_filters(svc.load_global_filters())
    a = svc.create("flag", "Flag")
    svc.save(update_spec_fields(a, filters=SetupFilters(min_adr_pct_20=0.04)))
    b = svc.create("ep", "EP")
    svc.save(update_spec_fields(b, filters=SetupFilters(min_adr_pct_20=0.08)))
    svc.create("mr", "Mean Reversion")
    svc.set_enabled("mr", False)

    rows = [
        _latest_row("AAPL", 10.0, 8_000_000, 0.05),  # flag only
        _latest_row("NVDA", 20.0, 20_000_000, 0.10),  # flag + ep
        _latest_row("XYZ", 1.0, 8_000_000, 0.10),  # fails global min_price=2
        _latest_row("CHEAPVOL", 10.0, 100.0, 0.10),  # fails global dollar vol
    ]
    pd.DataFrame(rows).to_parquet(cfg.paths.last_100_bars_parquet, index=False)

    result = compute_eligibility(cfg, svc)
    tickers = set(result.tickers["ticker"].astype(str))
    assert tickers == {"AAPL", "NVDA"}
    assert result.eligible_setups["AAPL"] == ("flag",)
    assert result.eligible_setups["NVDA"] == ("ep", "flag")
    assert "mr" not in result.eligible_setups.get("NVDA", ())


def test_eligibility_market_cap_fail_closed(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    svc = SetupService(cfg.paths)
    svc.create("flag", "Flag")
    yaml_path = cfg.paths.setups_dir / "flag" / "setup.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("min_market_cap: null", "min_market_cap: 100000000")
    yaml_path.write_text(text, encoding="utf-8")
    pd.DataFrame([_latest_row("AAPL", 10.0, 8_000_000, 0.05)]).to_parquet(
        cfg.paths.last_100_bars_parquet, index=False
    )
    with pytest.raises(MarketCapUnavailableError):
        compute_eligibility(cfg, svc)
