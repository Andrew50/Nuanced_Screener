from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

from ..config import LoaderConfig


class OHLCVVendor(Protocol):
    name: str

    def normalize_symbol(self, canonical_ticker: str) -> str:
        ...

    def fetch_daily_ohlcv(
        self,
        vendor_symbol: str,
        start: date | None,
        end: date | None,
        config: LoaderConfig,
    ) -> pd.DataFrame:
        """
        Return daily bars as a DataFrame with columns:
          - date (datetime64[ns] or date)
          - open, high, low, close (float)
          - volume (int)
        May include optional fields like adj_close.
        Must not include duplicate (date) rows for a single ticker.
        """

