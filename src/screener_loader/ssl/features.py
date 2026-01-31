from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..normalization import StandardBatch


@dataclass(frozen=True)
class ShapeFeatureSpec:
    """
    Feature spec for OHLCV -> market-shape features.

    - base features (6): r, rr, u, l, body, logv
    - optional context features: pos (trend-relative position)
    """

    include_pos: bool = False
    pos_ema_n: int = 20
    pos_atr_n: int = 20

    @property
    def feature_names(self) -> list[str]:
        base = ["r", "rr", "u", "l", "body", "logv"]
        if self.include_pos:
            base.append("pos")
        return base


def _safe_log_ratio(num: np.ndarray, den: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log(num/den) with masking.
    """
    out = np.full_like(num, np.nan, dtype=float)
    ok = valid & np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out[ok] = np.log(num[ok] / den[ok])
    return out, ok


def _ema_1d(x: np.ndarray, valid: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple EMA over time axis=1 for each batch row.

    This is intentionally conservative with missing values:
    - If a timestep is invalid, the EMA output is NaN and we do not update state.
    - EMA starts once the first valid value is observed.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("EMA n must be > 0")
    alpha = 2.0 / (n + 1.0)

    B, T = x.shape
    out = np.full((B, T), np.nan, dtype=float)
    out_valid = np.zeros((B, T), dtype=bool)
    state = np.full((B,), np.nan, dtype=float)
    state_valid = np.zeros((B,), dtype=bool)

    for t in range(T):
        xt = x[:, t]
        vt = valid[:, t] & np.isfinite(xt)
        # Init where state missing but xt valid
        init = vt & ~state_valid
        state[init] = xt[init]
        state_valid[init] = True

        # Update where both state and xt valid
        upd = vt & state_valid
        state[upd] = alpha * xt[upd] + (1.0 - alpha) * state[upd]

        out[:, t] = state
        out_valid[:, t] = state_valid

    # Output where state_valid is False is NaN already; mask via out_valid.
    return out, out_valid


def _atr_from_ohlc(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    mask_o: np.ndarray,
    mask_h: np.ndarray,
    mask_l: np.ndarray,
    mask_c: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ATR via EMA of True Range.
    """
    # prev close
    prev_close = np.roll(close, shift=1, axis=1)
    prev_close[:, 0] = np.nan
    m_prev_close = np.roll(mask_c, shift=1, axis=1)
    m_prev_close[:, 0] = False

    # True range: max(high-low, |high-prev_close|, |low-prev_close|)
    tr = np.full_like(close, np.nan, dtype=float)
    tr_valid = mask_h & mask_l & np.isfinite(high) & np.isfinite(low) & (high >= low)
    # We can improve TR when prev_close exists; otherwise fall back to (high-low).
    with_prev = tr_valid & m_prev_close & np.isfinite(prev_close)

    tr[tr_valid] = (high[tr_valid] - low[tr_valid]).astype(float)
    if with_prev.any():
        a = (high[with_prev] - low[with_prev]).astype(float)
        b = np.abs(high[with_prev] - prev_close[with_prev]).astype(float)
        c = np.abs(low[with_prev] - prev_close[with_prev]).astype(float)
        tr[with_prev] = np.maximum(a, np.maximum(b, c))
        tr_valid = tr_valid  # unchanged

    atr, atr_valid = _ema_1d(tr, tr_valid, n=int(n))
    return atr, atr_valid


def build_shape_features_from_ohlcv_batch(
    batch: StandardBatch,
    *,
    ohlcv_feature_indices: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    spec: ShapeFeatureSpec | None = None,
) -> StandardBatch:
    """
    Convert a StandardBatch containing OHLCV (at least 5 features) into a StandardBatch
    containing market-shape features.

    Assumptions:
    - `batch.x_seq[..., idx]` are [open, high, low, close, volume] by default.
    - Missing values are represented as NaN, and `batch.mask_seq` indicates valid entries.

    Output:
    - x_seq: (N, T, Fshape)
    - mask_seq: (N, T, Fshape)
    - y/meta preserved
    """
    spec = spec or ShapeFeatureSpec()
    o_i, h_i, l_i, c_i, v_i = [int(x) for x in ohlcv_feature_indices]

    x = np.asarray(batch.x_seq, dtype=float)
    m = np.asarray(batch.mask_seq, dtype=bool)
    if x.ndim != 3:
        raise ValueError("batch.x_seq must be (N,T,F)")
    if x.shape[-1] <= max(o_i, h_i, l_i, c_i, v_i):
        raise ValueError("batch.x_seq does not include required OHLCV feature indices")

    open_ = x[..., o_i]
    high = x[..., h_i]
    low = x[..., l_i]
    close = x[..., c_i]
    vol = x[..., v_i]

    m_open = m[..., o_i]
    m_high = m[..., h_i]
    m_low = m[..., l_i]
    m_close = m[..., c_i]
    m_vol = m[..., v_i]

    # r_t = log(C_t / C_{t-1})
    prev_close = np.roll(close, shift=1, axis=1)
    prev_close[:, 0] = np.nan
    m_prev_close = np.roll(m_close, shift=1, axis=1)
    m_prev_close[:, 0] = False
    r, m_r = _safe_log_ratio(close, prev_close, valid=(m_close & m_prev_close))

    # rr_t = log(H/L)
    rr, m_rr = _safe_log_ratio(high, low, valid=(m_high & m_low))

    # u_t = log(H / max(O,C))
    denom_u = np.maximum(open_, close)
    m_denom_u = m_open & m_close
    u, m_u = _safe_log_ratio(high, denom_u, valid=(m_high & m_denom_u))

    # l_t = log(min(O,C) / L)
    num_l = np.minimum(open_, close)
    m_num_l = m_open & m_close
    l, m_l = _safe_log_ratio(num_l, low, valid=(m_num_l & m_low))

    # body_t = log(max(O,C) / min(O,C))
    num_b = np.maximum(open_, close)
    den_b = np.minimum(open_, close)
    m_b = m_open & m_close
    body, m_body = _safe_log_ratio(num_b, den_b, valid=m_b)

    # v_t = log(1 + Volume_t)
    logv = np.full_like(vol, np.nan, dtype=float)
    m_logv = m_vol & np.isfinite(vol) & (vol >= 0)
    logv[m_logv] = np.log1p(vol[m_logv].astype(float))

    feats = [r, rr, u, l, body, logv]
    masks = [m_r, m_rr, m_u, m_l, m_body, m_logv]

    if spec.include_pos:
        # pos_t = (C_t - EMA_n(C)) / ATR_n
        ema, m_ema = _ema_1d(close, m_close & np.isfinite(close), n=int(spec.pos_ema_n))
        atr, m_atr = _atr_from_ohlc(
            open_=open_,
            high=high,
            low=low,
            close=close,
            mask_o=m_open,
            mask_h=m_high,
            mask_l=m_low,
            mask_c=m_close,
            n=int(spec.pos_atr_n),
        )
        pos = np.full_like(close, np.nan, dtype=float)
        m_pos = m_close & m_ema & m_atr & np.isfinite(ema) & np.isfinite(atr) & (atr > 0)
        pos[m_pos] = (close[m_pos] - ema[m_pos]) / atr[m_pos]
        feats.append(pos)
        masks.append(m_pos)

    x_out = np.stack(feats, axis=-1).astype(float)
    m_out = np.stack(masks, axis=-1).astype(bool)
    return StandardBatch(x_seq=x_out, mask_seq=m_out, y=batch.y, meta=batch.meta)

