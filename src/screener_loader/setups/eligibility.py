from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import LoaderConfig
from ..duckdb_utils import connect
from ..feature_sql import SCREENER_FILTER_FEATURES
from .service import SetupService
from .spec import (
    GlobalFilters,
    MarketCapUnavailableError,
    SetupSpec,
    effective_min,
)


@dataclass(frozen=True)
class EligibilityResult:
    asof_date: object | None
    tickers: pd.DataFrame
    # ticker -> tuple of setup ids
    eligible_setups: dict[str, tuple[str, ...]]


def compute_eligibility(
    config: LoaderConfig,
    service: SetupService,
    *,
    setups: list[SetupSpec] | None = None,
) -> EligibilityResult:
    """
    Apply global hard filters, then per-setup eligibility, then union.

    Returns one row per ticker that matches at least one enabled setup.
    """
    enabled = list(setups) if setups is not None else service.list_enabled()
    global_filters = service.load_global_filters()
    for spec in enabled:
        if spec.filters.requires_market_cap() and not service.market_cap_available:
            raise MarketCapUnavailableError(
                "Setup requires market_cap but market-cap data source is unavailable."
            )

    src = config.paths.last_100_bars_parquet
    if not src.exists():
        raise FileNotFoundError(
            f"Derived last-N bars not found: {src}. Run `ns rebuild-last100` first."
        )

    con = connect(config)
    sample = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(src)]).df()
    have = set(sample.columns.astype(str).tolist())
    missing = [c for c in SCREENER_FILTER_FEATURES if c not in have]
    if missing:
        raise RuntimeError(
            f"last_100_bars.parquet missing {missing}. Rebuild with default filter features "
            f"(`ns rebuild-last100`)."
        )

    latest = con.execute(
        """
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS asof_date,
          CAST(close AS DOUBLE) AS close,
          CAST(dollar_vol_avg_20 AS DOUBLE) AS dollar_vol_avg_20,
          CAST(adr_pct_20 AS DOUBLE) AS adr_pct_20
        FROM read_parquet(?)
        WHERE rn = 1
        """,
        [str(src)],
    ).df()
    if latest.empty:
        return EligibilityResult(asof_date=None, tickers=latest, eligible_setups={})

    latest = _apply_global(latest, global_filters)
    if latest.empty or not enabled:
        empty = latest.iloc[0:0].copy()
        return EligibilityResult(
            asof_date=latest["asof_date"].max() if not latest.empty else None,
            tickers=empty,
            eligible_setups={},
        )

    members: dict[str, list[str]] = {str(t): [] for t in latest["ticker"].astype(str)}
    for spec in enabled:
        mask = _setup_mask(latest, spec, global_filters)
        for ticker in latest.loc[mask, "ticker"].astype(str):
            members[ticker].append(spec.id)

    keep = [t for t, ids in members.items() if ids]
    out = latest[latest["ticker"].astype(str).isin(keep)].copy().reset_index(drop=True)
    eligible = {t: tuple(members[t]) for t in keep}
    out["eligible_setups"] = out["ticker"].astype(str).map(lambda t: list(eligible[t]))
    asof = out["asof_date"].max() if not out.empty else None
    return EligibilityResult(asof_date=asof, tickers=out, eligible_setups=eligible)


def _apply_global(df: pd.DataFrame, global_filters: GlobalFilters) -> pd.DataFrame:
    out = df
    if global_filters.min_price is not None:
        out = out[out["close"].notna() & (out["close"] >= float(global_filters.min_price))]
    if global_filters.min_dollar_vol_20d is not None:
        out = out[
            out["dollar_vol_avg_20"].notna()
            & (out["dollar_vol_avg_20"] >= float(global_filters.min_dollar_vol_20d))
        ]
    return out.reset_index(drop=True)


def _setup_mask(df: pd.DataFrame, spec: SetupSpec, global_filters: GlobalFilters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    min_price = effective_min(spec.filters.min_price, global_filters.min_price)
    min_dv = effective_min(spec.filters.min_dollar_vol_20d, global_filters.min_dollar_vol_20d)
    if min_price is not None:
        mask = mask & df["close"].notna() & (df["close"] >= float(min_price))
    if min_dv is not None:
        mask = mask & df["dollar_vol_avg_20"].notna() & (df["dollar_vol_avg_20"] >= float(min_dv))
    if spec.filters.min_adr_pct_20 is not None:
        mask = mask & df["adr_pct_20"].notna() & (df["adr_pct_20"] >= float(spec.filters.min_adr_pct_20))
    return mask
