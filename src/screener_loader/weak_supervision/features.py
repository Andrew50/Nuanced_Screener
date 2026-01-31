from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextSpec:
    past_window: int
    future_window: int

    @property
    def asof_index(self) -> int:
        return int(self.past_window) - 1


def _nan_slope(y: np.ndarray) -> float:
    """
    Simple least-squares slope of y over x=0..n-1, ignoring NaNs.
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    x2 = x[ok]
    y2 = y[ok]
    x2 = x2 - x2.mean()
    denom = float(np.sum(x2**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x2 * (y2 - y2.mean())) / denom)


def _true_range(high: np.ndarray, low: np.ndarray, close_prev: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close_prev = np.asarray(close_prev, dtype=float)
    a = high - low
    b = np.abs(high - close_prev)
    c = np.abs(low - close_prev)
    out = np.nanmax(np.stack([a, b, c], axis=0), axis=0)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    """
    Average True Range over the last `window` elements.
    Uses simple average of TR; ignores NaNs.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return float("nan")
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = _true_range(high, low, close_prev)
    tr = tr[-int(window) :]
    if np.all(np.isnan(tr)):
        return float("nan")
    return float(np.nanmean(tr))


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score: (x - median) / (1.4826 * MAD)
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (x - med) / scale


def _get_scalar_at_trel(g: pd.DataFrame, col: str, t_rel: int) -> float:
    sub = g[g["t_rel"] == int(t_rel)]
    if sub.empty or col not in sub.columns:
        return float("nan")
    v = sub.iloc[0][col]
    try:
        return float(v)
    except Exception:
        return float("nan")


def build_sample_features_from_context_long(
    context_long: pd.DataFrame,
    *,
    spec: ContextSpec,
    cons_window: int = 8,
    impulse_lookback: int = 10,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Build per-sample, teacher-side features from a context window long table.

    Expects columns: sample_id, t_rel, open, high, low, close, volume
    """
    required = {"sample_id", "t_rel", "open", "high", "low", "close", "volume"}
    missing = required - set(context_long.columns)
    if missing:
        raise ValueError(f"context_long missing required columns: {sorted(missing)}")

    past = int(spec.past_window)
    future = int(spec.future_window)
    if past <= 1:
        raise ValueError("ContextSpec.past_window must be >= 2 to compute prev_close/gaps")

    rows: list[dict] = []
    for sid, g in context_long.groupby("sample_id", sort=False):
        g = g.sort_values("t_rel").reset_index(drop=True)

        # Scalars at/around asof.
        prev_close = _get_scalar_at_trel(g, "close", -1)
        asof_open = _get_scalar_at_trel(g, "open", 0)
        asof_high = _get_scalar_at_trel(g, "high", 0)
        asof_low = _get_scalar_at_trel(g, "low", 0)
        asof_close = _get_scalar_at_trel(g, "close", 0)
        asof_vol = _get_scalar_at_trel(g, "volume", 0)

        gap_pct = (asof_open / prev_close - 1.0) if (np.isfinite(asof_open) and np.isfinite(prev_close) and prev_close != 0) else float("nan")

        # Past-only slices (exclude t_rel=0 by default for "as-of-open" semantics).
        past_mask = (g["t_rel"] < 0) & (g["t_rel"] >= -(past - 1))
        gp = g.loc[past_mask]

        cons_n = int(min(cons_window, len(gp)))
        cons = gp.tail(cons_n) if cons_n > 0 else gp.iloc[0:0]

        impulse_k = int(min(impulse_lookback, len(gp)))
        impulse = gp.tail(impulse_k) if impulse_k > 0 else gp.iloc[0:0]

        # Consolidation stats (past window, recent).
        cons_high = float(np.nanmax(cons["high"].to_numpy(dtype=float))) if not cons.empty else float("nan")
        cons_low = float(np.nanmin(cons["low"].to_numpy(dtype=float))) if not cons.empty else float("nan")
        cons_range = cons_high - cons_low if np.isfinite(cons_high) and np.isfinite(cons_low) else float("nan")
        cons_mid = float(np.nanmedian(cons["close"].to_numpy(dtype=float))) if not cons.empty else float("nan")
        cons_range_pct = (cons_range / cons_mid) if (np.isfinite(cons_range) and np.isfinite(cons_mid) and cons_mid != 0) else float("nan")

        # Volume trend during consolidation.
        cons_vol = cons["volume"].to_numpy(dtype=float) if not cons.empty else np.array([], dtype=float)
        vol_slope = _nan_slope(np.log1p(cons_vol)) if cons_vol.size else float("nan")

        # Past volume baseline (liquidity proxy).
        past_vol = gp["volume"].to_numpy(dtype=float)
        past_vol_avg20 = float(np.nanmean(past_vol[-20:])) if past_vol.size else float("nan")
        volume_shock = (
            float(asof_vol / past_vol_avg20)
            if (np.isfinite(asof_vol) and np.isfinite(past_vol_avg20) and past_vol_avg20 > 0)
            else float("nan")
        )

        # ATR on past window (up to t_rel=-1).
        atr_val = atr(
            gp["high"].to_numpy(dtype=float),
            gp["low"].to_numpy(dtype=float),
            gp["close"].to_numpy(dtype=float),
            window=atr_window,
        )
        cons_range_atr = (cons_range / atr_val) if (_finite_number(cons_range) and _finite_number(atr_val) and atr_val > 0) else float("nan")

        # Impulse return (past lookback ending at t_rel=-1).
        imp_close = impulse["close"].to_numpy(dtype=float)
        impulse_return = float("nan")
        if imp_close.size >= 2 and np.isfinite(imp_close[0]) and imp_close[0] != 0 and np.isfinite(imp_close[-1]):
            impulse_return = float(imp_close[-1] / imp_close[0] - 1.0)

        # Future validation stats.
        fut_mask = (g["t_rel"] > 0) & (g["t_rel"] <= future)
        gf = g.loc[fut_mask]
        fut_max_close = float(np.nanmax(gf["close"].to_numpy(dtype=float))) if not gf.empty else float("nan")
        fut_min_close = float(np.nanmin(gf["close"].to_numpy(dtype=float))) if not gf.empty else float("nan")

        rows.append(
            {
                "sample_id": str(sid),
                "prev_close": prev_close,
                "asof_open": asof_open,
                "asof_high": asof_high,
                "asof_low": asof_low,
                "asof_close": asof_close,
                "asof_volume": asof_vol,
                "past_vol_avg20": past_vol_avg20,
                "volume_shock": volume_shock,
                "gap_pct": gap_pct,
                "cons_high": cons_high,
                "cons_low": cons_low,
                "cons_range_pct": cons_range_pct,
                "cons_range_atr": cons_range_atr,
                "cons_vol_slope": vol_slope,
                "atr": atr_val,
                "impulse_return": impulse_return,
                "future_max_close": fut_max_close,
                "future_min_close": fut_min_close,
            }
        )

    return pd.DataFrame(rows)


def _finite_number(x: float) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False

