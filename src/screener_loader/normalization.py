from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import warnings


NormalizationMode = Literal["none", "per_window_zscore", "per_window_robust_zscore", "returns_relative", "global_fit"]


@dataclass(frozen=True)
class StandardBatch:
    # (batch, T, F)
    x_seq: np.ndarray
    # (batch, T, F) True where value is present/usable
    mask_seq: np.ndarray
    # (batch,) binary labels (0/1)
    y: np.ndarray
    # metadata arrays (batch,)
    meta: dict[str, np.ndarray]


@dataclass
class GlobalFitStats:
    mean: np.ndarray  # (F,)
    std: np.ndarray  # (F,)


def _nanmean(x: np.ndarray, mask: np.ndarray, axis: Any) -> np.ndarray:
    """
    Mean over `axis` ignoring masked values.

    Unlike `np.nanmean(np.where(mask, x, np.nan))`, this avoids RuntimeWarnings
    ("Mean of empty slice") for fully-masked slices by returning 0.0 for them.
    """
    m = np.asarray(mask, dtype=bool)
    x0 = np.where(m, x, 0.0)
    cnt = m.sum(axis=axis).astype(float)
    denom = np.where(cnt > 0, cnt, 1.0)
    return x0.sum(axis=axis) / denom


def _nanstd(x: np.ndarray, mask: np.ndarray, axis: Any) -> np.ndarray:
    """
    Std-dev over `axis` ignoring masked values (population std, ddof=0).

    Avoids RuntimeWarnings from NumPy's nanvar/nanstd for fully-masked slices by
    returning 0.0 for them.
    """
    m = np.asarray(mask, dtype=bool)
    x0 = np.where(m, x, 0.0)
    cnt = m.sum(axis=axis).astype(float)
    denom = np.where(cnt > 0, cnt, 1.0)
    mean = x0.sum(axis=axis) / denom
    mean2 = (x0 * x0).sum(axis=axis) / denom
    var = np.maximum(mean2 - (mean * mean), 0.0)
    return np.sqrt(var)


def _nanmedian(x: np.ndarray, mask: np.ndarray, axis: Any) -> np.ndarray:
    x2 = np.where(mask, x, np.nan)
    # NumPy warns on all-NaN slices; treat them as 0.0 (neutral for normalization).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        med = np.nanmedian(x2, axis=axis)
    return np.nan_to_num(med, nan=0.0)


def _nanmad(x: np.ndarray, mask: np.ndarray, axis: Any) -> np.ndarray:
    """
    Median absolute deviation (MAD), ignoring masked values.
    """
    med = _nanmedian(x, mask, axis=axis)
    # Broadcast med back to x shape for abs(x - med).
    if axis == 1:
        med_b = med[:, None, :]
    else:
        med_b = med
    dev = np.abs(x - med_b)
    return _nanmedian(dev, mask, axis=axis)


def build_standard_batch_from_windowed_long(
    windowed_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    window_size: int,
    label_column: str = "label",
    extra_meta_columns: list[str] | None = None,
) -> StandardBatch:
    """
    Convert long windowed bars into a dense batch tensor.

    Expects columns:
    - sample_id, t, ticker, asof_date, setup, <label_column>
    - feature columns (numeric; NULLs allowed)
    """
    if windowed_df.empty:
        raise ValueError("windowed_df is empty")
    required = {"sample_id", "t", "ticker", "asof_date", "setup", label_column}
    missing = required - set(windowed_df.columns)
    if missing:
        raise ValueError(f"windowed_df missing required columns: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    for c in feature_columns:
        if c not in windowed_df.columns:
            raise ValueError(f"feature column not in windowed_df: {c}")

    extra_meta_columns = extra_meta_columns or []
    for c in extra_meta_columns:
        key = str(c).strip()
        if not key:
            continue
        if key not in windowed_df.columns:
            raise ValueError(f"extra_meta_column not in windowed_df: {key}")

    df = windowed_df.sort_values(["sample_id", "t"]).reset_index(drop=True)
    T = int(window_size)
    if T <= 0:
        raise ValueError("window_size must be > 0")

    # Validate each sample has exactly T rows.
    sizes = df.groupby("sample_id")["t"].size().unique().tolist()
    if sizes != [T]:
        raise ValueError(f"Expected all samples to have {T} rows, got group sizes: {sorted(sizes)}")

    sample_ids = df["sample_id"].astype(str).unique()
    n = int(len(sample_ids))
    if len(df) != n * T:
        raise ValueError("Unexpected row count for dense reshape")

    x = df[feature_columns].to_numpy(dtype=float, copy=True).reshape(n, T, len(feature_columns))
    mask = ~np.isnan(x)

    # y/meta: take from the first row of each sample.
    first = df.groupby("sample_id", sort=False).head(1).set_index("sample_id")
    y_raw = first[label_column]
    if str(y_raw.dtype) in {"bool", "boolean"}:
        y = y_raw.astype(int).to_numpy()
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).to_numpy()

    meta = {
        "sample_id": sample_ids.astype(str),
        "ticker": first.loc[sample_ids, "ticker"].astype(str).to_numpy(),
        "asof_date": first.loc[sample_ids, "asof_date"].astype("datetime64[ns]").to_numpy(),
        "setup": first.loc[sample_ids, "setup"].astype(str).to_numpy(),
    }

    for c in extra_meta_columns:
        key = str(c).strip()
        if not key:
            continue
        meta[key] = first.loc[sample_ids, key].to_numpy()

    return StandardBatch(x_seq=x, mask_seq=mask, y=y, meta=meta)


def fit_global_zscore_stats(
    batches: Iterable[StandardBatch],
) -> GlobalFitStats:
    """
    Fit global mean/std per feature across a stream of batches, ignoring masked values.
    """
    sum_x = None
    sum_x2 = None
    count = None

    for b in batches:
        x = b.x_seq
        m = b.mask_seq
        # Reduce across batch and time -> (F,)
        x_masked = np.where(m, x, 0.0)
        c = m.sum(axis=(0, 1)).astype(float)  # (F,)
        s = x_masked.sum(axis=(0, 1))
        s2 = (x_masked**2).sum(axis=(0, 1))

        if sum_x is None:
            sum_x = s
            sum_x2 = s2
            count = c
        else:
            sum_x += s
            sum_x2 += s2
            count += c

    if sum_x is None or count is None or sum_x2 is None:
        raise ValueError("No batches provided to fit_global_zscore_stats")

    denom = np.where(count > 0, count, np.nan)
    mean = sum_x / denom
    var = (sum_x2 / denom) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return GlobalFitStats(mean=mean, std=std)


def normalize_batch(
    batch: StandardBatch,
    *,
    mode: NormalizationMode,
    global_stats: GlobalFitStats | None = None,
) -> StandardBatch:
    """
    Apply normalization/transformations while preserving NaN masking semantics.
    """
    x = batch.x_seq.copy()
    mask = batch.mask_seq.copy()

    if mode == "none":
        return batch

    if mode == "returns_relative":
        # Interpret feature set as [open, high, low, close, volume, ...]
        # and build a compact return-like representation from the first 5.
        if x.shape[-1] < 5:
            raise ValueError("returns_relative requires at least 5 features: open, high, low, close, volume")
        open_, high_, low_, close_, vol_ = [x[..., i] for i in range(5)]
        m_open, m_high, m_low, m_close, m_vol = [mask[..., i] for i in range(5)]

        prev_close = np.roll(close_, shift=1, axis=1)
        prev_close[:, 0] = np.nan
        m_prev_close = np.roll(m_close, shift=1, axis=1)
        m_prev_close[:, 0] = False

        def _ret(a, ma):
            ok = ma & m_prev_close & (prev_close != 0) & ~np.isnan(prev_close)
            out = np.full_like(a, np.nan, dtype=float)
            out[ok] = (a[ok] / prev_close[ok]) - 1.0
            return out, ok

        ret_open, m_ro = _ret(open_, m_open)
        ret_high, m_rh = _ret(high_, m_high)
        ret_low, m_rl = _ret(low_, m_low)
        ret_close, m_rc = _ret(close_, m_close)

        log_vol = np.full_like(vol_, np.nan, dtype=float)
        okv = m_vol & (vol_ >= 0)
        log_vol[okv] = np.log1p(vol_[okv])

        x = np.stack([ret_open, ret_high, ret_low, ret_close, log_vol], axis=-1)
        mask = np.stack([m_ro, m_rh, m_rl, m_rc, okv], axis=-1)

    if mode == "per_window_zscore":
        mean = _nanmean(x, mask, axis=1)  # (batch, F)
        std = _nanstd(x, mask, axis=1)  # (batch, F)
        std = np.where(std > 0, std, 1.0)
        x = (x - mean[:, None, :]) / std[:, None, :]
        return StandardBatch(x_seq=x, mask_seq=mask, y=batch.y, meta=batch.meta)

    if mode == "per_window_robust_zscore":
        # Robust z-score per window: (x - median) / (1.4826 * MAD)
        med = _nanmedian(x, mask, axis=1)  # (batch, F)
        mad = _nanmad(x, mask, axis=1)  # (batch, F)
        scale = 1.4826 * mad
        scale = np.where(scale > 0, scale, 1.0)
        x = (x - med[:, None, :]) / scale[:, None, :]
        return StandardBatch(x_seq=x, mask_seq=mask, y=batch.y, meta=batch.meta)

    if mode == "global_fit":
        if global_stats is None:
            raise ValueError("global_fit normalization requires global_stats")
        std = np.where(global_stats.std > 0, global_stats.std, 1.0)
        x = (x - global_stats.mean[None, None, :]) / std[None, None, :]
        return StandardBatch(x_seq=x, mask_seq=mask, y=batch.y, meta=batch.meta)

    raise ValueError(f"Unknown normalization mode: {mode}")

