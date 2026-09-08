from __future__ import annotations

from typing import Iterable

# DuckDB SELECT fragments. `adr_pct_20` requires `_daily_range_pct` on the input relation
# (see `staged_ohlcv_sql`).
FEATURE_SQL: dict[str, str] = {
    "ret_1d": "(close / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_1d",
    "ret_5d": "(close / LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_5d",
    "ret_21d": "(close / LAG(close, 21) OVER (PARTITION BY ticker ORDER BY date) - 1.0) AS ret_21d",
    "vol_avg_20": "AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_avg_20",
    "dollar_vol_avg_20": "AVG(volume * close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS dollar_vol_avg_20",
    "range_pct": "((high - low) / NULLIF(close, 0.0)) AS range_pct",
    # Frozen definition: mean of (high-low)/prior_close over 20 bars with 20 defined terms.
    # Stored as a fraction (0.04 = 4%). Not ATR and not (high-low)/same-bar close.
    "adr_pct_20": (
        "CASE WHEN COUNT(_daily_range_pct) OVER "
        "(PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) >= 20 "
        "THEN AVG(_daily_range_pct) OVER "
        "(PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) "
        "ELSE NULL END AS adr_pct_20"
    ),
}

FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    "ret_1d": 1,
    "ret_5d": 5,
    "ret_21d": 21,
    "vol_avg_20": 19,
    "dollar_vol_avg_20": 19,
    "range_pct": 0,
    # 20 ranges, each needing the previous close of the earliest bar.
    "adr_pct_20": 20,
}

# Always materialize these on last-N rebuilds so eligibility can filter without extra flags.
SCREENER_FILTER_FEATURES: tuple[str, ...] = ("dollar_vol_avg_20", "adr_pct_20")

DAILY_RANGE_PCT_SQL = (
    "(high - low) / NULLIF(LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date), 0.0) AS _daily_range_pct"
)


def merge_filter_features(feature_columns: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for f in list(feature_columns) + list(SCREENER_FILTER_FEATURES):
        key = str(f).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def max_feature_lookback_days(feature_columns: Iterable[str]) -> int:
    m = 0
    for f in feature_columns:
        key = str(f).strip()
        if not key:
            continue
        m = max(m, FEATURE_LOOKBACK_DAYS.get(key, 0))
    return m


def feature_sql_fragments(feature_columns: Iterable[str]) -> list[str]:
    exprs: list[str] = []
    for f in feature_columns:
        key = str(f).strip()
        if not key:
            continue
        if key not in FEATURE_SQL:
            raise ValueError(f"Unknown feature column: {key}. Known: {sorted(FEATURE_SQL)}")
        exprs.append(FEATURE_SQL[key])
    return exprs
