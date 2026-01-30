from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from ..config import LoaderConfig
from ..http_client import get_thread_local_session


@dataclass(frozen=True)
class StooqVendor:
    """
    Free/no-auth daily OHLCV via Stooq CSV endpoint.

    Note: Stooq's US symbols are typically like `aapl.us` (lowercase).
    """

    name: str = "stooq"

    def normalize_symbol(self, canonical_ticker: str) -> str:
        # Stooq uses '.' for class shares (e.g. brk.b.us).
        t = canonical_ticker.strip().upper().replace("-", ".")
        return f"{t.lower()}.us"

    def fetch_daily_ohlcv(
        self,
        vendor_symbol: str,
        start: Optional[date],
        end: Optional[date],
        config: LoaderConfig,
    ) -> pd.DataFrame:
        if config.interval != "1d":
            raise ValueError("Stooq vendor currently supports only interval=1d")

        url = f"https://stooq.com/q/d/l/?s={vendor_symbol}&i=d"
        session = get_thread_local_session()
        resp = session.get(
            url,
            timeout=float(config.timeout_seconds),
            headers={"User-Agent": "Nuanced_Screener/0.1"},
        )
        # Stooq uses 404 for unknown symbols; treat as "no data" rather than failure.
        if resp.status_code == 404:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close"])
        resp.raise_for_status()

        text = resp.text or ""
        first_line = text.lstrip().splitlines()[0].strip() if text.strip() else ""
        expected_header = "Date,Open,High,Low,Close,Volume"

        # Stooq sometimes returns non-CSV error pages; don't silently treat those as NO_DATA.
        if first_line and first_line != expected_header:
            # Common "no data" messages (keep small; add as encountered)
            lower = first_line.lower()
            if "no data" in lower or "brak danych" in lower:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close"])
            snippet = text[:200].replace("\n", "\\n")
            ct = resp.headers.get("content-type", "")
            raise RuntimeError(
                f"Unexpected Stooq response for {vendor_symbol} (status={resp.status_code}, content-type={ct}): {snippet}"
            )

        # CSV columns: Date,Open,High,Low,Close,Volume
        df = pd.read_csv(io.StringIO(text))
        if df is None or df.empty or "Date" not in df.columns:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adj_close"])

        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        keep = ["date", "open", "high", "low", "close", "volume"]
        df = df[keep].copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date", "close"]).copy()

        # Apply optional start/end filtering (inclusive).
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        df["adj_close"] = pd.NA  # not provided
        df["source"] = self.name
        df["asof_ts"] = datetime.now(timezone.utc)
        return df

