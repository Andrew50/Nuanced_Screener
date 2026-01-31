from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pandas as pd

from .normalization import StandardBatch, normalize_batch


MaskPolicy = Literal["none", "open_only_last_bar"]


def stable_window_id(
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    window_size: int,
    stride: int,
    start_idx: int,
    end_idx: int,
) -> str:
    """
    Stable identifier for a specific raw-history window (independent of encoder).
    """
    s = (
        f"{ticker.upper()}|{start_date.isoformat()}|{end_date.isoformat()}|"
        f"L={int(window_size)}|stride={int(stride)}|i0={int(start_idx)}|i1={int(end_idx)}"
    )
    h = hashlib.sha1(s.encode("utf-8")).digest()[:8]
    return h.hex()


@dataclass(frozen=True)
class NormalizationConfig:
    """
    Normalization pipeline for embedding index build + pattern fit + search.

    We keep it explicit and serializable so it can be hashed and enforced.
    """

    steps: tuple[str, ...] = ("returns_relative", "per_window_zscore")

    def as_dict(self) -> dict:
        return {"steps": list(self.steps)}


def apply_mask_policy_inplace(
    x_seq: np.ndarray,
    mask_seq: np.ndarray,
    *,
    feature_columns: list[str],
    mask_policy: MaskPolicy,
) -> None:
    """
    Apply masking semantics to the last bar in each window.
    """
    if mask_policy == "none":
        return
    if mask_policy != "open_only_last_bar":
        raise ValueError(f"Unknown mask_policy: {mask_policy}")

    if x_seq.ndim != 3 or mask_seq.ndim != 3:
        raise ValueError("x_seq/mask_seq must be (batch,T,F)")
    if x_seq.shape != mask_seq.shape:
        raise ValueError("x_seq and mask_seq shapes must match")

    cols = [str(c).strip() for c in feature_columns]
    if "open" not in cols:
        raise ValueError("mask_policy open_only_last_bar requires 'open' in feature_columns")
    open_idx = cols.index("open")
    last_t = x_seq.shape[1] - 1
    # Mask all features except open at the last bar.
    for j in range(x_seq.shape[2]):
        if j == open_idx:
            continue
        x_seq[:, last_t, j] = np.nan
        mask_seq[:, last_t, j] = False


def normalize_windows(
    x_seq: np.ndarray,
    mask_seq: np.ndarray,
    *,
    normalization: NormalizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a normalization pipeline to windows.
    """
    b = StandardBatch(x_seq=x_seq, mask_seq=mask_seq, y=np.zeros((x_seq.shape[0],), dtype=int), meta={})
    for step in normalization.steps:
        b = normalize_batch(b, mode=step)  # type: ignore[arg-type]
    return b.x_seq, b.mask_seq


def load_raw_ticker_history(
    parquet_path: str,
    *,
    date_start: date | None = None,
    date_end: date | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df
    if "date" not in df.columns:
        raise ValueError(f"Raw parquet missing 'date' column: {parquet_path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date"]).reset_index(drop=True)
    if date_start is not None:
        df = df[df["date"] >= date_start].reset_index(drop=True)
    if date_end is not None:
        df = df[df["date"] <= date_end].reset_index(drop=True)
    return df


def iter_sliding_windows_for_ticker(
    df: pd.DataFrame,
    *,
    ticker: str,
    feature_columns: list[str],
    window_size: int,
    stride: int,
    mask_policy: MaskPolicy,
    normalization: NormalizationConfig,
    batch_size: int = 2048,
) -> Iterator[tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """
    Yield batches of normalized windows for one ticker and one (window_size, stride).

    Returns tuples of:
    - meta_df: one row per window with ids/dates/indices
    - x_seq: (batch,T,F)
    - mask_seq: (batch,T,F)
    """
    if df.empty:
        return
    cols = [str(c).strip() for c in feature_columns if str(c).strip()]
    if not cols:
        raise ValueError("feature_columns must be non-empty")
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column in raw history: {c}")
    if int(window_size) <= 1:
        raise ValueError("window_size must be >= 2")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")

    # Enforce alignment with normalize_batch(..., mode="returns_relative") expectations.
    if "returns_relative" in set(normalization.steps):
        must = ["open", "high", "low", "close", "volume"]
        if len(cols) < 5 or [c.lower() for c in cols[:5]] != must:
            raise ValueError(
                "Normalization step returns_relative requires the first 5 feature_columns to be "
                f"{must} (in that exact order). Got: {cols[:5]}"
            )

    dates = df["date"].tolist()
    n = len(df)
    L = int(window_size)
    s = int(stride)
    if n < L:
        return

    # Dense array for slice speed: (n,F)
    x_full = df[cols].to_numpy(dtype=float, copy=True)

    metas: list[dict] = []
    xs: list[np.ndarray] = []
    ms: list[np.ndarray] = []

    def _flush() -> Iterator[tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        nonlocal metas, xs, ms
        if not metas:
            return iter(())
        meta_df = pd.DataFrame(metas)
        x_seq = np.stack(xs, axis=0)
        mask_seq = np.stack(ms, axis=0)
        # Apply mask policy (e.g. open-only last bar) BEFORE normalization.
        apply_mask_policy_inplace(x_seq, mask_seq, feature_columns=cols, mask_policy=mask_policy)
        # Normalize deterministically.
        x_seq, mask_seq = normalize_windows(x_seq, mask_seq, normalization=normalization)
        out = (meta_df, x_seq, mask_seq)
        metas, xs, ms = [], [], []
        return iter((out,))

    for start_idx in range(0, n - L + 1, s):
        end_idx = start_idx + L - 1
        start_d = dates[start_idx]
        end_d = dates[end_idx]
        w = x_full[start_idx : end_idx + 1, :]
        xw = w.astype(float, copy=True)
        mw = ~np.isnan(xw)
        wid = stable_window_id(
            ticker=ticker,
            start_date=start_d,
            end_date=end_d,
            window_size=L,
            stride=s,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        metas.append(
            {
                "window_id": wid,
                "ticker": str(ticker).upper(),
                "start_date": start_d,
                "end_date": end_d,
                "end_year_month": f"{end_d.year:04d}-{end_d.month:02d}",
                "window_size": int(L),
                "stride": int(s),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
            }
        )
        xs.append(xw)
        ms.append(mw)
        if len(metas) >= int(batch_size):
            yield from _flush()

    yield from _flush()


def iter_raw_ticker_parquet_files(raw_dir: str) -> Iterable[str]:
    """
    Deterministic listing of per-ticker raw Parquet files.
    """
    root = Path(raw_dir)
    if not root.exists():
        return []
    files = sorted(root.glob("*.parquet"))
    return [str(p) for p in files]

