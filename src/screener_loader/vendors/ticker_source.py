from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..config import LoaderConfig


class TickerSource(Protocol):
    def fetch_universe(self, config: LoaderConfig) -> pd.DataFrame:
        """
        Return a DataFrame with (at minimum):
          - ticker: str
          - name: str
          - exchange: str
          - is_etf: bool
          - is_test_issue: bool
        """

