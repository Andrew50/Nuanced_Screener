from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.vendors import nasdaq_trader


def test_parse_files_and_filtering_via_fetch(monkeypatch) -> None:
    nas_text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares|
AAPL|Apple Inc.|Q|N|N|100|N|N|
TEST|Test Issue|Q|Y|N|100|N|N|
SPY|SPDR S&P 500 ETF|G|N|N|100|Y|N|
File Creation Time: 01292026 00:00||
"""

    oth_text = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol|
IBM|International Business Machines|N|IBM|N|100|N|IBM|
QQQ|Invesco QQQ Trust|P|QQQ|Y|100|N|QQQ|
TST|Test security|A|TST|N|100|Y|TST|
File Creation Time: 01292026 00:00||
"""

    # Sanity check the pure parsers
    nas = nasdaq_trader.parse_nasdaqlisted_text(nas_text)
    oth = nasdaq_trader.parse_otherlisted_text(oth_text)
    assert set(nas["ticker"]) == {"AAPL", "TEST", "SPY"}
    assert set(oth["ticker"]) == {"IBM", "QQQ", "TST"}
    assert oth.set_index("ticker").loc["IBM", "exchange"] == "NYSE"
    assert oth.set_index("ticker").loc["QQQ", "exchange"] == "NYSEARCA"
    assert oth.set_index("ticker").loc["TST", "exchange"] == "AMEX"

    @dataclass
    class _Resp:
        text: str

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, *args, **kwargs):  # noqa: ANN001
        if url.endswith("nasdaqlisted.txt"):
            return _Resp(nas_text)
        if url.endswith("otherlisted.txt"):
            return _Resp(oth_text)
        raise AssertionError(f"unexpected url: {url}")

    class _Session:
        def get(self, url: str, *args, **kwargs):  # noqa: ANN001
            return fake_get(url, *args, **kwargs)

    monkeypatch.setattr(nasdaq_trader, "get_thread_local_session", lambda: _Session())

    cfg = LoaderConfig(
        repo_root=Path("."),
        exclude_test_issues=True,
        exclude_etfs=True,
        include_exchanges=("NASDAQ", "NYSE", "AMEX"),
    )
    df, meta = nasdaq_trader.fetch_universe(cfg)
    assert meta.filters["exclude_test_issues"] is True
    assert meta.filters["exclude_etfs"] is True

    # Should include AAPL (NASDAQ, non-test, non-ETF) and IBM (NYSE, non-test, non-ETF)
    assert set(df["ticker"]) == {"AAPL", "IBM"}

