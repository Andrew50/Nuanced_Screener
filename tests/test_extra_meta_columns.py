from __future__ import annotations

from datetime import date

import pandas as pd

from screener_loader.normalization import build_standard_batch_from_windowed_long


def test_build_standard_batch_includes_extra_meta_columns() -> None:
    rows = []
    for sid, w in [("a", 0.25), ("b", 2.0)]:
        for t in range(3):
            rows.append(
                {
                    "sample_id": sid,
                    "t": t,
                    "ticker": "T",
                    "asof_date": date(2025, 1, 10),
                    "setup": "flag",
                    "label": True,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                    "weight": w,
                }
            )
    df = pd.DataFrame(rows)
    b = build_standard_batch_from_windowed_long(
        df,
        feature_columns=["open", "high", "low", "close", "volume"],
        window_size=3,
        extra_meta_columns=["weight"],
    )
    assert "weight" in b.meta
    assert [float(x) for x in b.meta["weight"].tolist()] == [0.25, 2.0]

