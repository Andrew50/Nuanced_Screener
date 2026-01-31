from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .config import LoaderConfig


@dataclass(frozen=True)
class CandidateSpec:
    """
    Controls high-recall candidate emission over a fixed-length window.

    `end_lookback_bars` allows "latest-only, but not last-candle-only" proposals by emitting
    candidates whose pattern-end is slightly before the most recent bar.
    """

    window_size: int = 100
    end_lookback_bars: int = 7
    flag_lengths: tuple[int, ...] = (10, 20, 30, 40, 60)
    gap_lengths: tuple[int, ...] = (8, 16, 32, 64)
    meanrev_lengths: tuple[int, ...] = (16, 32, 64)
    pivot_lengths: tuple[int, ...] = (8, 16, 24, 32)


def _safe_token(x: str) -> str:
    # Keep setup strings filesystem-ish and stable.
    out = []
    for ch in str(x):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        elif ch in {":", " "}:
            out.append("_")
        else:
            out.append("_")
    return "".join(out).strip("_")


def _setup(pattern_family: str, template: str, L: int, variant: str) -> str:
    # Canonical scheme: "{pattern_family}:{template}:L{L}:{variant}"
    return f"{_safe_token(pattern_family)}:{_safe_token(template)}:L{int(L)}:{_safe_token(variant)}"


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr.astype(float)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(float)
    s = pd.Series(x)
    return s.rolling(int(window), min_periods=int(window)).mean().to_numpy(dtype=float)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(x, dtype=float)
    s = pd.Series(x)
    return s.rolling(int(window), min_periods=int(window)).std(ddof=0).to_numpy(dtype=float)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(x)
    return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean().to_numpy(dtype=float)


def _in_bounds(t_start: int, t_end: int, T: int) -> bool:
    return 0 <= int(t_start) <= int(t_end) < int(T)


def _emit_rows_for_ticker_window(
    df: pd.DataFrame,
    *,
    spec: CandidateSpec,
) -> list[dict]:
    """
    `df` must be chronological for one ticker with exactly spec.window_size rows.
    """
    T = int(spec.window_size)
    if len(df) != T:
        return []

    ticker = str(df["ticker"].iloc[0]).upper()
    dates = df["date"].tolist()
    asof_date: date = dates[-1]

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    tr = _true_range(h, l, c)
    atr_5 = _rolling_mean(tr, 5)
    atr_14 = _rolling_mean(tr, 14)
    atr_20 = _rolling_mean(tr, 20)
    atr_ratio_5_20 = np.where((atr_20 > 0) & ~np.isnan(atr_5) & ~np.isnan(atr_20), atr_5 / atr_20, np.nan)

    ma_10 = _ema(c, 10)
    ma_20 = _ema(c, 20)

    vol_avg_20 = _rolling_mean(v, 20)
    vol_ratio = np.where((vol_avg_20 > 0) & ~np.isnan(vol_avg_20), v / vol_avg_20, np.nan)

    out: list[dict] = []
    t_end_candidates = []
    for end_offset in range(0, max(1, int(spec.end_lookback_bars))):
        t_end = (T - 1) - end_offset
        if t_end <= 0:
            continue
        t_end_candidates.append((t_end, end_offset))

    # Template 2: gap events
    for t_end, end_offset in t_end_candidates:
        prev = t_end - 1
        if prev < 0:
            continue
        prev_close = c[prev]
        if not np.isfinite(prev_close) or prev_close == 0:
            continue
        gap_pct = float(abs(o[t_end] - prev_close) / prev_close)
        dir_up = int(o[t_end] > prev_close)
        vratio = float(vol_ratio[t_end]) if np.isfinite(vol_ratio[t_end]) else np.nan

        # Wide thresholds for high recall.
        for g_thresh in (0.02, 0.04, 0.06):
            for v_thresh in (2.0, 3.0, 5.0):
                if not (gap_pct >= g_thresh and (np.isfinite(vratio) and vratio >= v_thresh)):
                    continue
                for L in spec.gap_lengths:
                    t_start = t_end - (int(L) - 1)
                    if not _in_bounds(t_start, t_end, T):
                        continue
                    variant = f"E{end_offset}_gap{int(round(g_thresh*100))}_vol{int(v_thresh)}_dir{'up' if dir_up else 'dn'}"
                    out.append(
                        {
                            "ticker": ticker,
                            "asof_date": asof_date,
                            "setup": _setup("gap", "gap_event", int(L), variant),
                            "label": False,
                            "cand_t_start": int(t_start),
                            "cand_t_end": int(t_end),
                            "cand_length": int(L),
                            "cand_end_offset": int(end_offset),
                            "cand_gap_pct": float(gap_pct),
                            "cand_vol_ratio": float(vratio),
                            "cand_dir_up": int(dir_up),
                        }
                    )

    # Template 3: mean reversion candidates (z-score vs EMA20)
    # Use a std window that matches z lookback.
    for t_end, end_offset in t_end_candidates:
        for z_win in (20, 50):
            if t_end < z_win:
                continue
            mu = float(ma_20[t_end]) if np.isfinite(ma_20[t_end]) else np.nan
            sd = float(_rolling_std(c, z_win)[t_end])
            if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
                continue
            z = float((c[t_end] - mu) / sd)
            for z_thresh in (1.5, 2.0, 3.0):
                if not (abs(z) >= z_thresh):
                    continue
                for L in spec.meanrev_lengths:
                    t_start = t_end - (int(L) - 1)
                    if not _in_bounds(t_start, t_end, T):
                        continue
                    variant = f"E{end_offset}_z{z_thresh:g}_w{z_win}"
                    out.append(
                        {
                            "ticker": ticker,
                            "asof_date": asof_date,
                            "setup": _setup("meanrev", "zscore_extreme", int(L), variant),
                            "label": False,
                            "cand_t_start": int(t_start),
                            "cand_t_end": int(t_end),
                            "cand_length": int(L),
                            "cand_end_offset": int(end_offset),
                            "cand_z": float(z),
                            "cand_z_window": int(z_win),
                        }
                    )

    # Template 4: pivot / shakeout then burst (heuristic)
    for t_end, end_offset in t_end_candidates:
        # Define a short pullback window ending at t_end-1 and a reclaim at t_end.
        if t_end < 5:
            continue
        reclaim = t_end
        pull_end = t_end - 1
        # Pullback length 1..4 bars.
        for pb_len in (1, 2, 3, 4):
            pb_start = pull_end - (pb_len - 1)
            if pb_start < 1:
                continue
            # Trend context: close above MA20 before pullback starts.
            ctx = pb_start - 1
            if not (np.isfinite(ma_20[ctx]) and c[ctx] > ma_20[ctx]):
                continue
            # Pullback: down move and a shakeout wick signature (optional).
            pb_ret = float(c[pull_end] / c[ctx] - 1.0) if c[ctx] != 0 else np.nan
            pull_low = float(np.min(l[pb_start : pull_end + 1]))
            # Reclaim: close back above MA10 and above previous close.
            if not (np.isfinite(ma_10[reclaim]) and np.isfinite(c[reclaim]) and c[reclaim] > ma_10[reclaim]):
                continue
            if not (c[reclaim] > c[pull_end]):
                continue
            # Depth in ATR units at pullback end.
            atr = float(atr_14[pull_end]) if np.isfinite(atr_14[pull_end]) else np.nan
            depth_atr = float((ma_10[pull_end] - c[pull_end]) / atr) if np.isfinite(atr) and atr > 0 else np.nan

            for L in spec.pivot_lengths:
                t_start = t_end - (int(L) - 1)
                if not _in_bounds(t_start, t_end, T):
                    continue
                variant = f"E{end_offset}_pb{pb_len}"
                out.append(
                    {
                        "ticker": ticker,
                        "asof_date": asof_date,
                        "setup": _setup("pivot", "shakeout_reclaim", int(L), variant),
                        "label": False,
                        "cand_t_start": int(t_start),
                        "cand_t_end": int(t_end),
                        "cand_length": int(L),
                        "cand_end_offset": int(end_offset),
                        "cand_pullback_len": int(pb_len),
                        "cand_pullback_ret": float(pb_ret),
                        "cand_pull_low": float(pull_low),
                        "cand_depth_atr": float(depth_atr),
                    }
                )

    # Template 1: impulse + consolidation (flag-like)
    # Sloppy high-recall heuristic: look for a strong k-bar move somewhere in the window,
    # followed by ATR contraction into the candidate end.
    for t_end, end_offset in t_end_candidates:
        for L in spec.flag_lengths:
            t_start = t_end - (int(L) - 1)
            if not _in_bounds(t_start, t_end, T):
                continue
            seg = slice(t_start, t_end + 1)
            seg_close = c[seg]
            if not np.isfinite(seg_close[0]) or seg_close[0] == 0:
                continue
            total_ret = float(seg_close[-1] / seg_close[0] - 1.0)

            # Impulse: max 5-bar return in the first ~2/3 of the segment.
            k = 5
            if (t_end - t_start + 1) < (k + 3):
                continue
            max_idx = None
            max_ret = -np.inf
            search_end = t_start + int(0.66 * (t_end - t_start))
            for i in range(t_start, max(t_start, search_end - k + 1) + 1):
                j = i + k - 1
                if j > t_end:
                    break
                if c[i] == 0 or not np.isfinite(c[i]) or not np.isfinite(c[j]):
                    continue
                r = float(c[j] / c[i] - 1.0)
                if r > max_ret:
                    max_ret = r
                    max_idx = (i, j)
            if max_idx is None:
                continue
            imp_start, imp_end = max_idx

            # ATR expansion during impulse and contraction after.
            imp_slice = atr_ratio_5_20[imp_start : imp_end + 1]
            cons_slice = atr_ratio_5_20[imp_end + 1 : t_end + 1]
            imp_atr_ratio = float(np.nanmean(imp_slice)) if np.isfinite(imp_slice).any() else np.nan
            cons_atr_ratio = float(np.nanmean(cons_slice)) if np.isfinite(cons_slice).any() else np.nan

            # Retrace depth: how much of impulse was given back.
            imp_move = float(c[imp_end] - c[imp_start])
            cons_min = float(np.nanmin(c[imp_end : t_end + 1]))
            retrace = float((c[imp_end] - cons_min) / imp_move) if imp_move != 0 else np.nan

            # Wide thresholds for recall.
            if not (max_ret >= 0.05):
                continue
            if not (np.isfinite(cons_atr_ratio) and cons_atr_ratio <= 1.0):
                continue
            if not (np.isfinite(retrace) and retrace <= 0.8):
                continue

            variant = f"E{end_offset}_k{k}_imp{max_ret:.2f}_atrexp{imp_atr_ratio:.2f}_atrcon{cons_atr_ratio:.2f}"
            out.append(
                {
                    "ticker": ticker,
                    "asof_date": asof_date,
                    "setup": _setup("flag", "impulse_consolidation", int(L), variant),
                    "label": False,
                    "cand_t_start": int(t_start),
                    "cand_t_end": int(t_end),
                    "cand_length": int(L),
                    "cand_end_offset": int(end_offset),
                    "cand_total_ret": float(total_ret),
                    "cand_impulse_ret": float(max_ret),
                    "cand_impulse_k": int(k),
                    "cand_impulse_atr_ratio": float(imp_atr_ratio),
                    "cand_consolidation_atr_ratio": float(cons_atr_ratio),
                    "cand_retrace": float(retrace),
                }
            )

    return out


def propose_latest_candidates(
    config: LoaderConfig,
    *,
    spec: CandidateSpec | None = None,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Generate high-recall heuristic candidates for the *latest* window per ticker.

    Reads `config.paths.last_100_bars_parquet` which contains last-N bars per ticker.
    Returns a DataFrame with at least: ticker, asof_date, setup, label, cand_t_start, cand_t_end.
    """
    spec = spec or CandidateSpec(window_size=int(config.window_size))
    T = int(spec.window_size)
    if T <= 0:
        raise ValueError("CandidateSpec.window_size must be > 0")

    src = config.paths.last_100_bars_parquet
    if not src.exists():
        raise FileNotFoundError(f"Derived last-N bars not found: {src}. Run `ns rebuild-last100` first.")

    df = pd.read_parquet(src, columns=["ticker", "date", "open", "high", "low", "close", "volume", "adj_close", "rn"])
    if df.empty:
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if tickers is not None:
        tset = set(str(t).upper() for t in tickers)
        df = df[df["ticker"].isin(tset)]

    # Ensure chronological per ticker.
    # last_100_bars.parquet stores rn=1 as most recent; sort by date for time-series ops.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    for ticker, sub in df.groupby("ticker", sort=True):
        sub = sub.reset_index(drop=True)
        # Only accept complete windows; keep sequences consistent.
        if len(sub) != T:
            continue
        rows.extend(_emit_rows_for_ticker_window(sub, spec=spec))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Enforce uniqueness for downstream window builder: (ticker, asof_date, setup).
    dup = out.duplicated(subset=["ticker", "asof_date", "setup"], keep=False)
    if dup.any():
        # Should be very rare; keep first occurrence deterministically.
        out = out.drop_duplicates(subset=["ticker", "asof_date", "setup"], keep="first").reset_index(drop=True)

    # Stable ordering: most recent patterns first (same asof_date, but smaller end_offset first).
    out = out.sort_values(["ticker", "asof_date", "cand_end_offset", "setup"]).reset_index(drop=True)
    return out

