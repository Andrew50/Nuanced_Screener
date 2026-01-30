from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from ..config import LoaderConfig
from ..http_client import get_thread_local_session


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


_OTHER_EXCHANGE_CODE_MAP: dict[str, str] = {
    # NASDAQ Trader "otherlisted" exchange codes.
    # Kept intentionally small; you can widen with config.include_exchanges if desired.
    "A": "AMEX",  # NYSE American (formerly AMEX)
    "N": "NYSE",
    "P": "NYSEARCA",
    "Z": "BATS",
    "V": "IEXG",  # seen in some symdirs
}


@dataclass(frozen=True)
class UniverseMeta:
    universe_version: str
    source_urls: tuple[str, str]
    filters: dict[str, Any]


def _download_text(url: str, timeout_seconds: float) -> str:
    session = get_thread_local_session()
    resp = session.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": "Nuanced_Screener/0.1 (+local loader)"},
    )
    resp.raise_for_status()
    return resp.text


def _iter_pipe_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in io.StringIO(text):
        line = raw_line.strip("\n").strip("\r").strip()
        if not line:
            continue
        if line.startswith("File Creation Time:"):
            continue
        parts = [p.strip() for p in line.split("|")]
        # Many lines end with a trailing pipe -> last element is "".
        if parts and parts[-1] == "":
            parts = parts[:-1]
        rows.append(parts)
    return rows


def _parse_nasdaqlisted(text: str) -> pd.DataFrame:
    rows = _iter_pipe_rows(text)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data_rows = rows[1:]
    # Expected fields:
    # Symbol | Security Name | Market Category | Test Issue | Financial Status
    # | Round Lot Size | ETF | NextShares
    if len(header) < 8:
        raise ValueError(f"Unexpected nasdaqlisted header: {header}")
    df = pd.DataFrame(data_rows, columns=header[: len(data_rows[0])])
    # Some files can have extra columns; ensure known names exist.
    required = ["Symbol", "Security Name", "Test Issue", "ETF"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"nasdaqlisted missing column {col!r}; columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "ticker": df["Symbol"].astype(str).str.strip().str.upper(),
            "name": df["Security Name"].astype(str).str.strip(),
            "exchange": "NASDAQ",
            "is_etf": df["ETF"].astype(str).str.strip().str.upper().eq("Y"),
            "is_test_issue": df["Test Issue"].astype(str).str.strip().str.upper().eq("Y"),
            "source_file": "nasdaqlisted.txt",
        }
    )
    out = out[out["ticker"].ne("")].copy()
    return out


def _parse_otherlisted(text: str) -> pd.DataFrame:
    rows = _iter_pipe_rows(text)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data_rows = rows[1:]
    # Expected fields:
    # ACT Symbol | Security Name | Exchange | CQS Symbol | ETF | Round Lot Size | Test Issue | NASDAQ Symbol
    if len(header) < 8:
        raise ValueError(f"Unexpected otherlisted header: {header}")
    df = pd.DataFrame(data_rows, columns=header[: len(data_rows[0])])
    required = ["ACT Symbol", "Security Name", "Exchange", "ETF", "Test Issue"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"otherlisted missing column {col!r}; columns={list(df.columns)}")
    exch = df["Exchange"].astype(str).str.strip().str.upper().map(_OTHER_EXCHANGE_CODE_MAP).fillna("OTHER")
    out = pd.DataFrame(
        {
            "ticker": df["ACT Symbol"].astype(str).str.strip().str.upper(),
            "name": df["Security Name"].astype(str).str.strip(),
            "exchange": exch,
            "is_etf": df["ETF"].astype(str).str.strip().str.upper().eq("Y"),
            "is_test_issue": df["Test Issue"].astype(str).str.strip().str.upper().eq("Y"),
            "source_file": "otherlisted.txt",
        }
    )
    out = out[out["ticker"].ne("")].copy()
    return out


def fetch_universe(config: LoaderConfig) -> tuple[pd.DataFrame, UniverseMeta]:
    """
    Fetch NASDAQ Trader symbol directory and return filtered universe.

    Filtering is applied AFTER merge of nasdaqlisted + otherlisted.
    """
    nas_text = _download_text(NASDAQ_LISTED_URL, timeout_seconds=config.timeout_seconds)
    oth_text = _download_text(OTHER_LISTED_URL, timeout_seconds=config.timeout_seconds)

    nas = _parse_nasdaqlisted(nas_text)
    oth = _parse_otherlisted(oth_text)
    df = pd.concat([nas, oth], ignore_index=True)

    # De-dupe tickers (prefer NASDAQ row if overlap).
    df["source_rank"] = df["source_file"].map({"nasdaqlisted.txt": 0, "otherlisted.txt": 1}).fillna(9).astype(int)
    df = (
        df.sort_values(["ticker", "source_rank"])
        .drop_duplicates(subset=["ticker"], keep="first")
        .drop(columns=["source_rank"])
    )

    if config.exclude_test_issues:
        df = df[~df["is_test_issue"]]
    if config.exclude_etfs:
        df = df[~df["is_etf"]]

    include_ex = {x.upper() for x in config.include_exchanges}
    df = df[df["exchange"].astype(str).str.upper().isin(include_ex)]

    df = df.sort_values(["exchange", "ticker"]).reset_index(drop=True)

    meta = UniverseMeta(
        universe_version=datetime.now(timezone.utc).isoformat(),
        source_urls=(NASDAQ_LISTED_URL, OTHER_LISTED_URL),
        filters={
            "exclude_test_issues": config.exclude_test_issues,
            "exclude_etfs": config.exclude_etfs,
            "include_exchanges": list(config.include_exchanges),
        },
    )
    return df, meta


# Public helpers for testing / alternative integration
def parse_nasdaqlisted_text(text: str) -> pd.DataFrame:
    return _parse_nasdaqlisted(text)


def parse_otherlisted_text(text: str) -> pd.DataFrame:
    return _parse_otherlisted(text)

