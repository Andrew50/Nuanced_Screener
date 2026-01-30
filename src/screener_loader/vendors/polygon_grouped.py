from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import pandas as pd

from ..config import LoaderConfig
from ..http_client import get_thread_local_session
from ..dotenv import load_dotenv_file


@dataclass(frozen=True)
class PolygonGroupedDailyVendor:
    """
    Polygon/Massive 'Daily Market Summary (OHLC)' endpoint:
      GET /v2/aggs/grouped/locale/us/market/stocks/{date}
    """

    name: str = "polygon_grouped"

    def fetch_grouped_daily(
        self,
        trading_date: date,
        api_key: str,
        adjusted: bool,
        include_otc: bool,
        timeout_seconds: float,
    ) -> pd.DataFrame:
        session = get_thread_local_session()
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{trading_date.isoformat()}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "include_otc": "true" if include_otc else "false",
            "apiKey": api_key,
        }
        resp = session.get(url, params=params, timeout=float(timeout_seconds))
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

        status = str(payload.get("status", "")).upper()
        results = payload.get("results") or []
        if status not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Polygon grouped daily returned status={payload.get('status')!r}: {payload}")

        if not isinstance(results, list):
            raise RuntimeError(f"Polygon grouped daily unexpected results type: {type(results)}")

        rows = []
        asof = datetime.now(timezone.utc)
        for r in results:
            if not isinstance(r, dict):
                continue
            ticker = r.get("T")
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": str(ticker).upper(),
                    "date": trading_date,
                    "open": r.get("o"),
                    "high": r.get("h"),
                    "low": r.get("l"),
                    "close": r.get("c"),
                    "volume": r.get("v"),
                    "adj_close": None,
                    "vwap": r.get("vw"),
                    "transactions": r.get("n"),
                    "ts_end_ms": r.get("t"),
                    "otc": bool(r.get("otc", False)),
                    "adjusted": bool(payload.get("adjusted", adjusted)),
                    "source": self.name,
                    "asof_ts": asof,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            # Could be a holiday or a date outside your plan's history.
            return df

        # Coerce numeric fields
        for c in ["open", "high", "low", "close", "vwap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        if "transactions" in df.columns:
            df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
        if "ts_end_ms" in df.columns:
            df["ts_end_ms"] = pd.to_numeric(df["ts_end_ms"], errors="coerce")

        df = df.dropna(subset=["ticker", "date", "close"]).copy()
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        return df


def require_polygon_api_key(repo_root) -> str:
    import os
    from pathlib import Path

    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        root = Path(repo_root)
        dotenv_path = root / ".env"
        loaded = load_dotenv_file(dotenv_path)
        if loaded.loaded:
            key = loaded.values.get("POLYGON_API_KEY", "").strip()
            if key:
                # Make it available to downstream libraries too.
                os.environ.setdefault("POLYGON_API_KEY", key)

    if not key:
        raise RuntimeError(
            "Missing POLYGON_API_KEY. Set it in the environment or create `.env` "
            "(you can copy `env.example` -> `.env`)."
        )
    return key

