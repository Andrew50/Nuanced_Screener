from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.vendors.stooq import StooqVendor


def test_stooq_raises_on_html_like_response(monkeypatch, tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    vendor = StooqVendor()

    @dataclass
    class _Resp:
        status_code: int = 200
        text: str = "<html>blocked</html>"
        headers: dict = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.headers is None:
                self.headers = {"content-type": "text/html"}

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, *args, **kwargs):  # noqa: ANN001
            return _Resp()

    monkeypatch.setattr("screener_loader.vendors.stooq.get_thread_local_session", lambda: _Session())

    try:
        vendor.fetch_daily_ohlcv("aapl.us", start=date(2020, 1, 1), end=None, config=cfg)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "Unexpected Stooq response" in str(e)

