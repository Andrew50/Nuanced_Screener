from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import LoaderConfig
from .duckdb_utils import connect


@dataclass(frozen=True)
class NamedQuery:
    name: str
    sql: str
    description: str


NAMED_QUERIES: dict[str, NamedQuery] = {
    "top_momentum_21d": NamedQuery(
        name="top_momentum_21d",
        description="Rank tickers by 21d return (requires feature_column=ret_21d).",
        sql="""
        WITH latest AS (
          SELECT *
          FROM read_parquet(?)
          WHERE rn = 1
        )
        SELECT
          ticker,
          date,
          close,
          volume,
          ret_21d
        FROM latest
        WHERE ret_21d IS NOT NULL
        ORDER BY ret_21d DESC
        LIMIT ?
        """,
    ),
    "liquid_momo_21d": NamedQuery(
        name="liquid_momo_21d",
        description="Momentum + liquidity filter (requires ret_21d and dollar_vol_avg_20).",
        sql="""
        WITH latest AS (
          SELECT *
          FROM read_parquet(?)
          WHERE rn = 1
        )
        SELECT
          ticker,
          date,
          close,
          dollar_vol_avg_20,
          ret_21d
        FROM latest
        WHERE
          ret_21d IS NOT NULL
          AND dollar_vol_avg_20 IS NOT NULL
          AND close >= 2
          AND dollar_vol_avg_20 >= 5e6
        ORDER BY ret_21d DESC
        LIMIT ?
        """,
    ),
    "range_pct_today": NamedQuery(
        name="range_pct_today",
        description="Rank tickers by today's intraday range percent (requires range_pct).",
        sql="""
        WITH latest AS (
          SELECT *
          FROM read_parquet(?)
          WHERE rn = 1
        )
        SELECT
          ticker,
          date,
          close,
          range_pct
        FROM latest
        WHERE range_pct IS NOT NULL
        ORDER BY range_pct DESC
        LIMIT ?
        """,
    ),
}


def run_named_query(config: LoaderConfig, query: str, limit: int = 50) -> pd.DataFrame:
    if query not in NAMED_QUERIES:
        raise KeyError(f"Unknown query {query!r}. Known: {sorted(NAMED_QUERIES)}")
    con = connect(config)
    q = NAMED_QUERIES[query]
    return con.execute(q.sql, [str(config.paths.last_100_bars_parquet), int(limit)]).df()

