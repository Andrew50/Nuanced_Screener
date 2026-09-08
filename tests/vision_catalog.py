"""Temporary YAML catalog + last-N + universe fixtures for Agent 3 tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import shutil

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.service import SetupService, update_spec_fields
from screener_loader.setups.spec import ChartStyle, GlobalFilters, SetupCriteria, SetupFilters

ASOF = date(2026, 6, 18)
LOOKBACK = 20
AFTER_CLOSE = datetime(2026, 6, 18, 21, 0, tzinfo=timezone.utc)
SYNTHETIC_PNG = Path(__file__).resolve().parent / "fixtures" / "vision" / "synthetic.png"


@dataclass
class CatalogHandles:
    cfg: LoaderConfig
    service: SetupService
    asof: date
    lookback: int


def after_close_clock():
    return AFTER_CLOSE


def write_universe(cfg: LoaderConfig, tickers: list[str]) -> None:
    cfg.paths.meta_dir.mkdir(parents=True, exist_ok=True)
    rows = ["ticker,name,exchange,is_etf,is_test_issue,source_file"]
    for t in tickers:
        rows.append(f"{t},{t} Corp,NASDAQ,False,False,nasdaqlisted.txt")
    (cfg.paths.tickers_csv).write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_last_n(
    cfg: LoaderConfig,
    *,
    tickers: dict[str, dict],
    lookback: int = LOOKBACK,
    asof: date = ASOF,
) -> None:
    rows: list[dict] = []
    for ticker, spec in tickers.items():
        n = int(spec.get("n", lookback))
        end = spec.get("end", asof)
        start = end - timedelta(days=n - 1)
        close = float(spec.get("close", 10.0))
        dollar_vol = spec.get("dollar_vol", 8_000_000.0)
        adr = spec.get("adr", 0.05)
        for i in range(n):
            d = start + timedelta(days=i)
            rn = n - i
            rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1_000_000,
                    "adj_close": close,
                    "dollar_vol_avg_20": dollar_vol,
                    "adr_pct_20": adr,
                    "rn": rn,
                    "source": spec.get("source"),
                }
            )
    pd.DataFrame(rows).to_parquet(cfg.paths.last_100_bars_parquet, index=False)


def write_raw_window(
    cfg: LoaderConfig,
    ticker: str,
    asof: date,
    n: int,
    *,
    close: float = 50.0,
) -> None:
    start = asof - timedelta(days=n - 1)
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        px = close + i * 0.1
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px,
                "volume": 1_000_000 + i,
                "adj_close": px,
                "source": "test",
            }
        )
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cfg.paths.raw_ticker_parquet(ticker), index=False)


def build_catalog(
    tmp_path: Path,
    *,
    lookback: int = LOOKBACK,
    volume: bool = True,
    moving_averages: tuple[int, ...] = (10, 20),
    include_examples: bool = True,
    second_lookback: int | None = None,
    market_cap: bool = False,
    extra_last_n: dict[str, dict] | None = None,
    universe: list[str] | None = None,
) -> CatalogHandles:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    svc = SetupService(cfg.paths, market_cap_available=False)
    svc.save_global_filters(GlobalFilters(min_price=2.0, min_dollar_vol_20d=1_000_000.0))

    flag = svc.create("Flag", lookback_bars=lookback)
    flag = svc.save(
        update_spec_fields(
            flag,
            criteria=SetupCriteria(required=("tight flag", "dry-up"), preferred=("gap",), disqualifiers=("extended",)),
            filters=SetupFilters(min_adr_pct_20=0.04),
            chart=ChartStyle(volume=volume, moving_averages=moving_averages),
            lookback_bars=lookback,
            description="Coiled flag",
        )
    )
    ep = svc.create("EP", lookback_bars=second_lookback or lookback)
    svc.save(
        update_spec_fields(
            ep,
            criteria=SetupCriteria(required=("gap",), preferred=("volume",)),
            filters=SetupFilters(min_adr_pct_20=0.08),
            chart=ChartStyle(volume=volume, moving_averages=moving_averages),
            lookback_bars=second_lookback or lookback,
            description="Episodic pivot",
        )
    )
    if market_cap:
        yaml_path = cfg.paths.setups_dir / "flag" / "setup.yaml"
        text = yaml_path.read_text(encoding="utf-8")
        yaml_path.write_text(text.replace("min_market_cap: null", "min_market_cap: 100000000"), encoding="utf-8")

    if include_examples:
        write_raw_window(cfg, "NVDA", date(2026, 5, 1), lookback)
        svc.add_market_window_example(
            "flag",
            ticker="NVDA",
            asof_date=date(2026, 5, 1),
            polarity="positive",
            quality="canonical",
            note="held-out must not use this window",
            example_id="z_edge",
        )
        svc.add_market_window_example(
            "flag",
            ticker="AAPL",
            asof_date=date(2026, 5, 1),
            polarity="positive",
            quality="decent",
            example_id="b_decent",
        )
        write_raw_window(cfg, "AAPL", date(2026, 5, 1), lookback)
        src_img = tmp_path / "upload_src.png"
        shutil.copy(SYNTHETIC_PNG, src_img)
        svc.add_image_example(
            "flag",
            src_img,
            polarity="positive",
            quality="canonical",
            note="uploaded bytes",
            example_id="upload",
        )
        svc.add_market_window_example(
            "flag",
            ticker="MSFT",
            asof_date=date(2026, 5, 1),
            polarity="negative",
            quality=None,
            example_id="n_unspec",
        )
        write_raw_window(cfg, "MSFT", date(2026, 5, 1), lookback)
        svc.add_market_window_example(
            "ep",
            ticker="NVDA",
            asof_date=date(2026, 5, 1),
            polarity="positive",
            quality="canonical",
            example_id="nvda_shared",
        )

    last_n = {
        "AAPL": {"close": 10.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": lookback, "end": ASOF},
        "NVDA": {"close": 20.0, "dollar_vol": 20_000_000, "adr": 0.10, "n": lookback, "end": ASOF},
        "NA": {"close": 15.0, "dollar_vol": 9_000_000, "adr": 0.09, "n": lookback, "end": ASOF},
        "OUT": {"close": 30.0, "dollar_vol": 9_000_000, "adr": 0.12, "n": lookback, "end": ASOF},
        "NULLADR": {"close": 12.0, "dollar_vol": 8_000_000, "adr": None, "n": lookback, "end": ASOF},
    }
    if extra_last_n:
        last_n.update(extra_last_n)
    write_last_n(cfg, tickers=last_n, lookback=lookback, asof=ASOF)
    write_universe(cfg, universe or ["AAPL", "NVDA", "NA", "MISS", "SHORT", "STALE", "NULLADR"])
    return CatalogHandles(cfg=cfg, service=svc, asof=ASOF, lookback=lookback)
