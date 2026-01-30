from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from screener_loader.config import LoaderConfig
from screener_loader.update import _plan_polygon_dates


@dataclass(frozen=True)
class _FakeCal:
    trading_days: list[date]

    def valid_trading_days(self, start: date, end: date) -> list[date]:
        return [d for d in self.trading_days if start <= d <= end]


def test_plan_polygon_dates_orders_latest_then_missing_then_existing() -> None:
    # Build 10 trading days ending at 2026-01-10
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(10)]
    cal = _FakeCal(trading_days=days)

    cfg = LoaderConfig(lookback_years=1, refresh_tail_days=3, calls_per_minute=5)

    existing = {date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 9), date(2026, 1, 10)}
    planned = _plan_polygon_dates(cfg, cal=cal, today=date(2026, 1, 10), existing_partitions=existing)

    # Required ordering:
    # 1) Latest day first, always.
    assert planned[0] == date(2026, 1, 10)

    # 2) Missing days (never loaded), newest -> oldest (exclude latest which is already planned).
    expected_missing = [date(2026, 1, d) for d in [8, 7, 6, 5, 3, 1]]
    assert planned[1 : 1 + len(expected_missing)] == expected_missing

    # 3) Already-loaded days, newest -> oldest (exclude latest which was planned first).
    expected_existing_after = [date(2026, 1, d) for d in [9, 4, 2]]
    assert planned[1 + len(expected_missing) :] == expected_existing_after

