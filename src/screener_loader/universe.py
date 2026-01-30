from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import LoaderConfig
from .paths import ensure_dirs
from .vendors import registry


def build_universe(config: LoaderConfig) -> Path:
    """
    Fetch/parse/filter NASDAQ Trader symbol directory and cache to data/meta/tickers.csv.
    """
    ensure_dirs(config.paths)
    df, meta = registry.fetch_universe(config)
    out_csv = config.paths.tickers_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta_path = config.paths.tickers_meta_json
    meta_path.write_text(
        json.dumps(
            {
                "universe_version": meta.universe_version,
                "source_urls": list(meta.source_urls),
                "filters": meta.filters,
                "row_count": int(len(df)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out_csv


def load_universe(config: LoaderConfig) -> pd.DataFrame:
    path = config.paths.tickers_csv
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. Run `ns universe` to build it."
        )
    # IMPORTANT: disable default NA parsing so tickers like "NA" don't become NaN.
    df = pd.read_csv(path, na_filter=False)
    if "ticker" not in df.columns:
        raise ValueError(f"Invalid tickers.csv (missing 'ticker' column): {path}")

    tickers = df["ticker"].astype("string").str.strip().str.upper()
    # Remove blank rows (and any weird whitespace-only tickers).
    mask = tickers.notna() & tickers.ne("")
    df = df.loc[mask].copy()
    df["ticker"] = tickers.loc[mask]

    # De-dupe (universe can occasionally contain overlaps; keep first).
    df = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return df

