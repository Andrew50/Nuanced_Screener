from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from screener_loader.vendors.polygon_grouped import PolygonGroupedDailyVendor


def test_polygon_grouped_normalization_monkeypatched_session(monkeypatch) -> None:
    payload = {
        "status": "OK",
        "adjusted": True,
        "results": [
            {"T": "AAPL", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100, "vw": 1.4, "n": 10, "t": 1700000000000},
            {"T": "msft", "o": 10, "h": 12, "l": 9, "c": 11, "v": 200},
        ],
    }

    @dataclass
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class _Session:
        def get(self, url, params=None, timeout=None):  # noqa: ANN001
            return _Resp()

    monkeypatch.setattr(
        "screener_loader.vendors.polygon_grouped.get_thread_local_session",
        lambda: _Session(),
    )

    v = PolygonGroupedDailyVendor()
    d = date(2026, 1, 29)
    df = v.fetch_grouped_daily(d, api_key="X", adjusted=True, include_otc=False, timeout_seconds=1.0)
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    assert set(df["date"]) == {d}
    assert "adj_close" in df.columns
    assert "source" in df.columns

