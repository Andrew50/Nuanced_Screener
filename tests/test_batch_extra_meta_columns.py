from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener_loader.normalization import build_standard_batch_from_windowed_long


def test_build_standard_batch_extra_meta_columns() -> None:
    rows = []
    sid = "abc123"
    for t in range(3):
        rows.append(
            {
                "sample_id": sid,
                "t": t,
                "ticker": "AAPL",
                "asof_date": date(2025, 3, 6),
                "setup": "flag",
                "label": False,
                "open": 10.0 + t,
                "high": 11.0 + t,
                "low": 9.0 + t,
                "close": 10.5 + t,
                "volume": 1000 + t,
                "cand_t_start": 0,
                "cand_t_end": 2,
            }
        )
    df = pd.DataFrame(rows)
    batch = build_standard_batch_from_windowed_long(
        df,
        feature_columns=["open", "high", "low", "close", "volume"],
        window_size=3,
        extra_meta_columns=["cand_t_start", "cand_t_end"],
    )
    assert "cand_t_start" in batch.meta
    assert "cand_t_end" in batch.meta
    assert batch.meta["cand_t_start"].shape == (1,)
    assert batch.meta["cand_t_end"].shape == (1,)
    assert int(batch.meta["cand_t_start"][0]) == 0
    assert int(batch.meta["cand_t_end"][0]) == 2
    assert np.asarray(batch.x_seq).shape == (1, 3, 5)

