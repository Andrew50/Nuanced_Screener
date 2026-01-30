from __future__ import annotations

from .nasdaq_trader import fetch_universe as fetch_universe_nasdaq_trader
from .stooq import StooqVendor
from .yahoo import YahooFinanceVendor


def fetch_universe(config):
    name = (config.ticker_source or "").strip().lower()
    if name in {"nasdaq_trader", "nasdaq-trader", "nasdaq"}:
        return fetch_universe_nasdaq_trader(config)
    raise ValueError(f"Unknown ticker_source: {config.ticker_source!r}")


def get_ohlcv_vendor(config):
    name = (config.ohlcv_vendor or "").strip().lower()
    if name in {"yahoo", "yfinance", "yf"}:
        return YahooFinanceVendor()
    if name in {"stooq"}:
        return StooqVendor()
    raise ValueError(f"Unknown ohlcv_vendor: {config.ohlcv_vendor!r}")

