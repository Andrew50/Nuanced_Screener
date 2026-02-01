from __future__ import annotations

import math

import numpy as np

from screener_loader.normalization import StandardBatch
from screener_loader.stack6_features import Stack6FeatureSpec, stack6_tabular_features


def _idx(names: list[str], key: str) -> int:
    assert key in names, f"missing feature {key!r}; have={names}"
    return names.index(key)


def test_stack6_is_masking_safe_and_computes_basic_features() -> None:
    # Small window with one short horizon so the test is deterministic.
    spec = Stack6FeatureSpec(horizons=(2,), atr_short=2, atr_long=3, ema_span=3, ema_slope_lookback=2, channel_endpoint_k=2)

    # Build a single-sample window: (T=6, F=5) == [open, high, low, close, volume]
    # Completed bars are t=0..4. Decision bar is t=5 (open only).
    T = 6
    x = np.full((1, T, 5), np.nan, dtype=float)
    m = np.zeros_like(x, dtype=bool)

    # Completed bars.
    closes = [10.0, 11.0, 12.0, 10.0, 14.0]
    highs = [11.0, 12.0, 13.0, 14.0, 15.0]
    lows = [9.0, 10.0, 11.0, 12.0, 13.0]
    opens = [10.0, 11.0, 12.0, 10.0, 14.0]
    vols = [100.0, 100.0, 100.0, 100.0, 200.0]

    for t in range(5):
        x[0, t, 0] = opens[t]
        x[0, t, 1] = highs[t]
        x[0, t, 2] = lows[t]
        x[0, t, 3] = closes[t]
        x[0, t, 4] = vols[t]
        m[0, t, :] = True

    # Decision day (t=5): only open is valid. Set masked OHLCV to extreme values to detect leakage.
    x[0, 5, 0] = 14.7
    m[0, 5, 0] = True
    x[0, 5, 1] = 1e9
    x[0, 5, 2] = 1e9
    x[0, 5, 3] = 1e9
    x[0, 5, 4] = 1e9
    m[0, 5, 1:] = False

    batch = StandardBatch(
        x_seq=x,
        mask_seq=m,
        y=np.array([1], dtype=int),
        meta={
            "sample_id": np.array(["s1"], dtype=object),
            "ticker": np.array(["TEST"], dtype=object),
            "asof_date": np.array(["2025-01-10"], dtype="datetime64[ns]"),
            "setup": np.array(["flag"], dtype=object),
        },
    )

    X, names = stack6_tabular_features(batch, spec=spec)
    assert X.shape[0] == 1
    assert X.shape[1] == len(names)

    # Gap_pct = open_now / close_prev - 1
    gap = X[0, _idx(names, "gap_pct")]
    assert math.isfinite(gap)
    assert abs(gap - (14.7 / 14.0 - 1.0)) < 1e-9

    # Return_log_2 = log(open_now / close[T-1-2]) = log(open_now / close[3])
    ret2 = X[0, _idx(names, "return_log_2")]
    assert math.isfinite(ret2)
    assert abs(ret2 - math.log(14.7 / 10.0)) < 1e-9

    # Ensure masked decision-day close didn't leak into return.
    assert ret2 < 10.0

    # ATR windows in this synthetic series are constant TR=2.0 => ATR_short=2, ATR_long=2.
    atr_s = X[0, _idx(names, "atr_short")]
    atr_l = X[0, _idx(names, "atr_long")]
    assert abs(atr_s - 2.0) < 1e-9
    assert abs(atr_l - 2.0) < 1e-9

    # prev_day_range_pct = (high_prev-low_prev)/close_prev = (15-13)/14
    pdr = X[0, _idx(names, "prev_day_range_pct")]
    assert abs(pdr - ((15.0 - 13.0) / 14.0)) < 1e-9

