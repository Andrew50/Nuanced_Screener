from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from ..config import LoaderConfig


@dataclass(frozen=True)
class YahooFinanceVendor:
    name: str = "yfinance"

    def normalize_symbol(self, canonical_ticker: str) -> str:
        # Yahoo often uses '-' for class shares where NASDAQ uses '.'.
        # Example: BRK.B (NASDAQ directory) -> BRK-B (Yahoo)
        t = canonical_ticker.strip().upper()
        t = t.replace(".", "-")
        return t

    def fetch_daily_ohlcv(
        self,
        vendor_symbol: str,
        start: Optional[date],
        end: Optional[date],
        config: LoaderConfig,
    ) -> pd.DataFrame:
        """
        Uses yfinance for initial prototype. Returns a DataFrame with:
          date, open, high, low, close, volume, adj_close, source, asof_ts
        """
        try:
            import yfinance as yf
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "yfinance is required for the Yahoo vendor. Install with `pip install yfinance`."
            ) from e

        # yfinance treats `end` as exclusive; treat user/config end as inclusive.
        yf_start = start.isoformat() if start else None
        yf_end = (end + timedelta(days=1)).isoformat() if end else None

        df = yf.download(
            tickers=vendor_symbol,
            start=yf_start,
            end=yf_end,
            interval=config.interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close"])

        df = df.reset_index()
        # Normalize columns
        col_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        keep = ["date", "open", "high", "low", "close", "volume"]
        if "adj_close" in df.columns:
            keep.append("adj_close")
        df = df[keep].copy()

        # Coerce date to date (not datetime) for stable Parquet schema.
        df["date"] = pd.to_datetime(df["date"], utc=False).dt.date
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        for c in ["open", "high", "low", "close", "adj_close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["date", "close"]).copy()
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        asof = datetime.now(timezone.utc)
        df["source"] = self.name
        df["asof_ts"] = asof
        return df

