from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .normalization import StandardBatch


@dataclass(frozen=True)
class Stack6FeatureSpec:
    horizons: tuple[int, ...] = (5, 10, 20, 40)
    volume_shock_ks: tuple[int, ...] = (10, 20)
    atr_short: int = 14
    atr_long: int = 50
    ema_span: int = 20
    ema_slope_lookback: int = 10
    channel_endpoint_k: int = 3


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmean(x, axis=axis)


def _nanstd(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanstd(x, axis=axis)


def _safe_at_time(x: np.ndarray, m: np.ndarray, t: int) -> np.ndarray:
    """
    Return x[:, t] where valid, else NaN. If t out of range, all NaN.
    """
    if t < 0 or t >= x.shape[1]:
        return np.full((x.shape[0],), np.nan, dtype=float)
    out = x[:, t].astype(float, copy=False)
    ok = m[:, t]
    return np.where(ok, out, np.nan)


def _safe_slice(x: np.ndarray, m: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Return x[:, start:end+1] with invalid entries set to NaN.
    If slice is empty/out of range, returns (n, 0).
    """
    n, T = x.shape
    if end < start:
        return np.empty((n, 0), dtype=float)
    start2 = max(0, int(start))
    end2 = min(T - 1, int(end))
    if end2 < start2:
        return np.empty((n, 0), dtype=float)
    xs = x[:, start2 : end2 + 1].astype(float, copy=False)
    ms = m[:, start2 : end2 + 1]
    return np.where(ms, xs, np.nan)


def _nanmedian_2d(x: np.ndarray) -> np.ndarray:
    # x: (n, k)
    if x.shape[1] == 0:
        return np.full((x.shape[0],), np.nan, dtype=float)
    return np.nanmedian(x, axis=1)


def _nanquantile_2d(x: np.ndarray, q: float) -> np.ndarray:
    if x.shape[1] == 0:
        return np.full((x.shape[0],), np.nan, dtype=float)
    return np.nanquantile(x, q=float(q), axis=1)


def _max_drawdown_closed_form(closes: np.ndarray) -> float:
    """
    O(n) max drawdown on a 1D close array with NaNs.
    Returns NaN if <2 valid points.
    """
    vals = closes[np.isfinite(closes)]
    if vals.size < 2:
        return float("nan")
    peak = float(vals[0])
    max_dd = 0.0
    for v in vals[1:]:
        if v > peak:
            peak = float(v)
            continue
        if peak > 0:
            dd = 1.0 - float(v / peak)
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def _ema_1d(values: np.ndarray, span: int) -> np.ndarray:
    """
    EMA for 1D values with NaNs. NaNs reset the EMA (next valid point seeds EMA).
    """
    out = np.full_like(values, np.nan, dtype=float)
    alpha = 2.0 / (float(span) + 1.0)
    last = float("nan")
    for i, v in enumerate(values.tolist()):
        if not np.isfinite(v):
            last = float("nan")
            continue
        if not np.isfinite(last):
            last = float(v)
        else:
            last = alpha * float(v) + (1.0 - alpha) * last
        out[i] = last
    return out


def stack6_tabular_features(
    batch: StandardBatch,
    *,
    spec: Stack6FeatureSpec = Stack6FeatureSpec(),
    eps: float = 1e-12,
    hmm_p_trend: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build Stack 6 tabular features from a `StandardBatch`.

    Leakage / masking semantics:
    - Decision time is at `t = T-1` open.
    - Only `open[T-1]` may be used from the decision-day bar.
    - All other features must be computed from completed bars: indices `0..T-2`.

    Assumes `x_seq[..., 0:5]` == [open, high, low, close, volume] (per `returns_relative` convention).
    """
    x = np.asarray(batch.x_seq, dtype=float)
    m = np.asarray(batch.mask_seq, dtype=bool)
    if x.ndim != 3 or m.ndim != 3:
        raise ValueError("Expected batch.x_seq and batch.mask_seq to be 3D arrays (batch, T, F).")
    n, T, F = x.shape
    if F < 5:
        raise ValueError("Stack6 requires at least 5 features: open, high, low, close, volume.")
    if T < 3:
        raise ValueError("Stack6 requires window_size >= 3 to have completed bars and a decision-day open.")

    open_ = x[:, :, 0]
    high_ = x[:, :, 1]
    low_ = x[:, :, 2]
    close_ = x[:, :, 3]
    vol_ = x[:, :, 4]

    m_open = m[:, :, 0]
    m_high = m[:, :, 1]
    m_low = m[:, :, 2]
    m_close = m[:, :, 3]
    m_vol = m[:, :, 4]

    t_now = T - 1
    t_prev = T - 2  # last completed bar

    # Decision price: open at asof_date (allowed, unmasked).
    price_now = _safe_at_time(open_, m_open, t_now)
    close_prev = _safe_at_time(close_, m_close, t_prev)
    high_prev = _safe_at_time(high_, m_high, t_prev)
    low_prev = _safe_at_time(low_, m_low, t_prev)
    vol_prev = _safe_at_time(vol_, m_vol, t_prev)

    # Completed bars are 0..T-2 inclusive.
    # True range requires previous close, so it's defined for t in 1..T-2.
    high_c = high_[:, : t_prev + 1]
    low_c = low_[:, : t_prev + 1]
    close_c = close_[:, : t_prev + 1]
    m_high_c = m_high[:, : t_prev + 1]
    m_low_c = m_low[:, : t_prev + 1]
    m_close_c = m_close[:, : t_prev + 1]

    prev_close = np.roll(close_c, shift=1, axis=1)
    prev_close[:, 0] = np.nan
    m_prev_close = np.roll(m_close_c, shift=1, axis=1)
    m_prev_close[:, 0] = False

    ok_tr = m_high_c & m_low_c & m_prev_close
    hl = np.where(m_high_c & m_low_c, high_c - low_c, np.nan)
    hc = np.where(m_high_c & m_prev_close, np.abs(high_c - prev_close), np.nan)
    lc = np.where(m_low_c & m_prev_close, np.abs(low_c - prev_close), np.nan)
    tr = np.where(ok_tr, np.nanmax(np.stack([hl, hc, lc], axis=-1), axis=-1), np.nan)

    def _atr(tr2: np.ndarray, window: int) -> np.ndarray:
        # trailing mean of TR over completed bars ending at t_prev; ignores NaNs.
        if window <= 0:
            return np.full((n,), np.nan, dtype=float)
        start = max(1, (t_prev + 1) - int(window))
        ts = tr2[:, start : t_prev + 1]
        return _nanmean(ts, axis=1) if ts.shape[1] else np.full((n,), np.nan, dtype=float)

    atr_s = _atr(tr, spec.atr_short)
    atr_l = _atr(tr, spec.atr_long)
    atr_l_safe = np.where(np.isfinite(atr_l) & (atr_l > 0), atr_l, np.nan)
    atr_ratio = atr_s / np.where(np.isfinite(atr_l) & (atr_l > 0), atr_l, np.nan)

    # Gap features (decision day open vs prev close).
    ok_gap = np.isfinite(price_now) & np.isfinite(close_prev) & (price_now > 0) & (close_prev > 0)
    gap_pct = np.full((n,), np.nan, dtype=float)
    gap_pct[ok_gap] = (price_now[ok_gap] / close_prev[ok_gap]) - 1.0
    gap_vs_atr = (price_now - close_prev) / atr_l_safe
    gap_direction = np.sign(gap_pct)
    prev_day_range_pct = np.full((n,), np.nan, dtype=float)
    ok_pdr = np.isfinite(high_prev) & np.isfinite(low_prev) & np.isfinite(close_prev) & (close_prev != 0)
    prev_day_range_pct[ok_pdr] = (high_prev[ok_pdr] - low_prev[ok_pdr]) / close_prev[ok_pdr]

    # Trend context: EMA20 slope and close vs EMA20.
    ema20_now = np.full((n,), np.nan, dtype=float)
    ema20_prev_k = np.full((n,), np.nan, dtype=float)
    close_vs_ema20_atr = np.full((n,), np.nan, dtype=float)
    ema20_slope_atr = np.full((n,), np.nan, dtype=float)
    for i in range(n):
        closes_i = np.where(m_close_c[i], close_c[i], np.nan)
        ema_i = _ema_1d(closes_i, span=spec.ema_span)
        ema_now = ema_i[t_prev] if t_prev < ema_i.shape[0] else np.nan
        ema20_now[i] = float(ema_now) if np.isfinite(ema_now) else np.nan
        t_back = t_prev - int(spec.ema_slope_lookback)
        if t_back >= 0 and t_back < ema_i.shape[0] and np.isfinite(ema_now) and np.isfinite(ema_i[t_back]):
            ema20_prev_k[i] = float(ema_i[t_back])
            denom = float(spec.ema_slope_lookback) if spec.ema_slope_lookback > 0 else 1.0
            slope = (float(ema_now) - float(ema_i[t_back])) / denom
            if np.isfinite(atr_l_safe[i]):
                ema20_slope_atr[i] = slope / float(atr_l_safe[i])
        if np.isfinite(close_prev[i]) and np.isfinite(ema_now) and np.isfinite(atr_l_safe[i]):
            close_vs_ema20_atr[i] = (float(close_prev[i]) - float(ema_now)) / float(atr_l_safe[i])

    # Z-score and ATR-distance to MA_k on completed closes (evaluate at decision price_now).
    ma20 = np.full((n,), np.nan, dtype=float)
    sd20 = np.full((n,), np.nan, dtype=float)
    ma50 = np.full((n,), np.nan, dtype=float)
    sd50 = np.full((n,), np.nan, dtype=float)

    def _rolling_stats_last_k(k: int) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, (t_prev + 1) - int(k))
        s = _safe_slice(close_c, m_close_c, start, t_prev)
        mu = _nanmean(s, axis=1) if s.shape[1] else np.full((n,), np.nan, dtype=float)
        sd = _nanstd(s, axis=1) if s.shape[1] else np.full((n,), np.nan, dtype=float)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
        return mu, sd

    ma20, sd20 = _rolling_stats_last_k(20)
    ma50, sd50 = _rolling_stats_last_k(50)

    dist_ma20_atr = (price_now - ma20) / atr_l_safe
    dist_ma50_atr = (price_now - ma50) / atr_l_safe
    dist_ma20_z = (price_now - ma20) / sd20
    dist_ma50_z = (price_now - ma50) / sd50

    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        if arr.shape != (n,):
            raise ValueError(f"Feature {name} wrong shape {arr.shape}, expected {(n,)}")
        feature_names.append(name)
        feature_cols.append(arr.astype(float, copy=False))

    # Base regime/vol context.
    add("atr_short", atr_s)
    add("atr_long", atr_l)
    add("atr_short_over_long", atr_ratio)

    # Gap / prior-day context.
    add("gap_pct", gap_pct)
    add("gap_vs_atr", gap_vs_atr)
    add("gap_direction", gap_direction)
    add("prev_day_range_pct", prev_day_range_pct)

    # Trend context.
    add("ema20_slope_atr", ema20_slope_atr)
    add("close_vs_ema20_atr", close_vs_ema20_atr)

    # MA distances.
    add("dist_ma20_atr", dist_ma20_atr)
    add("dist_ma50_atr", dist_ma50_atr)
    add("dist_ma20_z", dist_ma20_z)
    add("dist_ma50_z", dist_ma50_z)

    # Multi-horizon features from completed bars only.
    for h in spec.horizons:
        hh = int(h)
        start = (t_now - hh)  # == (T-1-h)
        end = t_prev
        # Returns: log(open_now / close[T-1-h])
        denom_close = _safe_at_time(close_, m_close, t_now - hh)
        ret_log = np.full((n,), np.nan, dtype=float)
        ok_ret = np.isfinite(price_now) & np.isfinite(denom_close) & (price_now > 0) & (denom_close > 0)
        ret_log[ok_ret] = np.log(price_now[ok_ret] / denom_close[ok_ret])
        add(f"return_log_{hh}", ret_log)

        # Range on highs/lows over last h completed bars.
        hi = _safe_slice(high_c, m_high_c, start, end)
        lo = _safe_slice(low_c, m_low_c, start, end)
        rng = np.nanmax(hi, axis=1) - np.nanmin(lo, axis=1) if hi.shape[1] else np.full((n,), np.nan, dtype=float)
        add(f"range_atr_{hh}", rng / atr_l_safe)

        # Impulse/retrace relative to decision price.
        impulse = rng
        retrace = (np.nanmax(hi, axis=1) - price_now) / np.maximum(impulse, eps) if hi.shape[1] else np.full((n,), np.nan)
        add(f"retrace_pct_{hh}", retrace)

        # Max drawdown on closes over last h completed bars.
        cl = _safe_slice(close_c, m_close_c, start, end)
        mdd = np.full((n,), np.nan, dtype=float)
        if cl.shape[1]:
            for i in range(n):
                mdd[i] = _max_drawdown_closed_form(cl[i])
        add(f"max_drawdown_{hh}", mdd)

        # Candle aggregates: mean upper/lower wick and body%range over last h completed bars.
        o = _safe_slice(open_, m_open, start, end)  # open is present on completed bars too
        c = _safe_slice(close_c, m_close_c, start, end)
        # high/low already sliced: hi/lo
        if hi.shape[1]:
            ok = np.isfinite(o) & np.isfinite(c) & np.isfinite(hi) & np.isfinite(lo)
            upper = np.where(ok, hi - np.maximum(o, c), np.nan)
            lower = np.where(ok, np.minimum(o, c) - lo, np.nan)
            body = np.where(ok, np.abs(c - o), np.nan)
            denom = np.where(ok, np.maximum(hi - lo, eps), np.nan)
            body_pct = body / denom
            add(f"upper_wick_mean_{hh}", np.nanmean(upper, axis=1))
            add(f"lower_wick_mean_{hh}", np.nanmean(lower, axis=1))
            add(f"body_pct_range_mean_{hh}", np.nanmean(body_pct, axis=1))
        else:
            add(f"upper_wick_mean_{hh}", np.full((n,), np.nan, dtype=float))
            add(f"lower_wick_mean_{hh}", np.full((n,), np.nan, dtype=float))
            add(f"body_pct_range_mean_{hh}", np.full((n,), np.nan, dtype=float))

        # Channel slope (endpoint-of-means) and width (robust quantiles), all in ATR units.
        k = int(spec.channel_endpoint_k)
        if hi.shape[1] and hi.shape[1] >= max(2, k):
            first_hi = np.nanmean(hi[:, :k], axis=1)
            last_hi = np.nanmean(hi[:, -k:], axis=1)
            first_lo = np.nanmean(lo[:, :k], axis=1)
            last_lo = np.nanmean(lo[:, -k:], axis=1)
            denom = float(max(hh - 1, 1))
            slope_hi = (last_hi - first_hi) / denom
            slope_lo = (last_lo - first_lo) / denom
            add(f"channel_slope_high_atr_{hh}", slope_hi / atr_l_safe)
            add(f"channel_slope_low_atr_{hh}", slope_lo / atr_l_safe)
            width = (_nanquantile_2d(hi, 0.90) - _nanquantile_2d(lo, 0.10))
            add(f"channel_width_atr_{hh}", width / atr_l_safe)
        else:
            add(f"channel_slope_high_atr_{hh}", np.full((n,), np.nan, dtype=float))
            add(f"channel_slope_low_atr_{hh}", np.full((n,), np.nan, dtype=float))
            add(f"channel_width_atr_{hh}", np.full((n,), np.nan, dtype=float))

    # Width contraction ratio (20/40) if both exist.
    def _get_width(h: int) -> np.ndarray:
        key = f"channel_width_atr_{int(h)}"
        try:
            idx = feature_names.index(key)
        except ValueError:
            return np.full((n,), np.nan, dtype=float)
        return feature_cols[idx]

    w20 = _get_width(20)
    w40 = _get_width(40)
    add("channel_width_contraction_20_over_40", w20 / w40)

    # Multi-k volume shock (use previous day's volume vs trailing medians).
    for k in spec.volume_shock_ks:
        kk = int(k)
        start = (t_now - kk)
        end = t_prev
        vols = _safe_slice(vol_[:, : t_prev + 1], m_vol[:, : t_prev + 1], start, end)
        med = _nanmedian_2d(vols)
        vs = np.full((n,), np.nan, dtype=float)
        ok_vs = np.isfinite(vol_prev) & np.isfinite(med) & (med > 0)
        vs[ok_vs] = vol_prev[ok_vs] / med[ok_vs]
        add(f"volume_shock_{kk}", vs)

    if hmm_p_trend is not None:
        hmm_p_trend = np.asarray(hmm_p_trend, dtype=float)
        if hmm_p_trend.shape != (n,):
            raise ValueError(f"hmm_p_trend must be shape {(n,)}, got {hmm_p_trend.shape}")
        add("hmm_p_trend", hmm_p_trend)

    X = np.stack(feature_cols, axis=1) if feature_cols else np.empty((n, 0), dtype=float)
    return X, feature_names


def stack6_required_window_size(spec: Stack6FeatureSpec) -> int:
    """
    Minimum window size required to compute all requested horizons safely.

    We need:
    - decision bar at T-1
    - completed bars at least up to T-2
    - largest horizon h requires access to close[T-1-h] and slice [T-1-h..T-2]
    """
    max_h = max((int(h) for h in spec.horizons), default=1)
    # Need T-1-h >= 0 => T >= h+1, and also have completed bar T-2 => T>=2.
    return max(3, max_h + 1)

