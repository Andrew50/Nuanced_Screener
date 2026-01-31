from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_config_hash(obj: Any) -> str:
    """
    Stable short hash for any JSON-serializable config.
    """
    s = _stable_json_dumps(obj)
    h = hashlib.sha1(s.encode("utf-8")).digest()[:8]
    return h.hex()


def l2_normalize(x: np.ndarray, *, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between row vectors:
    - a: (N,D)
    - b: (M,D)
    Returns: (N,M)
    """
    a2 = l2_normalize(a, axis=-1)
    b2 = l2_normalize(b, axis=-1)
    return a2 @ b2.T


class EmbeddingModel(Protocol):
    """
    Encode a masked sequence into a fixed-dimensional embedding.

    Inputs are dense tensors:
    - x_seq: (batch, T, F)
    - mask_seq: (batch, T, F) boolean; True where value is valid
    """

    def config(self) -> dict[str, Any]:
        ...

    def encode(self, x_seq: np.ndarray, mask_seq: np.ndarray) -> np.ndarray:
        ...


def _masked_nan(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, x, np.nan)


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmean(x, axis=axis)


def _nanstd(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanstd(x, axis=axis)


def _nanmin(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmin(x, axis=axis)


def _nanmax(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmax(x, axis=axis)


def _last_valid(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Last valid value per (batch, feature) across time.
    Returns (batch, F) with NaN where none valid.
    """
    # mask_any: (B,T,F)
    m = mask.astype(bool)
    B, T, F = x.shape
    out = np.full((B, F), np.nan, dtype=float)
    # For each (B,F), find last True along T.
    # Reverse time and find first True.
    rev = m[:, ::-1, :]
    has = rev.any(axis=1)  # (B,F)
    # argmax gives first occurrence of max; since False=0 True=1, argmax finds first True.
    idx_rev = rev.argmax(axis=1)  # (B,F) but meaningless when has=False
    idx = (T - 1) - idx_rev
    for b in range(B):
        # vectorized gather per b
        ii = idx[b]
        ok = has[b]
        if not ok.any():
            continue
        out[b, ok] = x[b, ii[ok], np.arange(F)[ok]]
    return out


def _slope(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute a simple linear slope over time for each feature, using masked least squares.
    Returns (batch,F).
    """
    B, T, F = x.shape
    t = np.arange(T, dtype=float)[None, :, None]  # (1,T,1)
    m = mask.astype(bool)
    xv = _masked_nan(x, m)

    # Weighted least squares slope:
    # slope = cov(t,x)/var(t) with weights and NaNs
    # Center t and x under weights.
    w = m.astype(float)
    w_sum = w.sum(axis=1)  # (B,F)
    w_sum = np.where(w_sum > 0, w_sum, np.nan)
    t_mean = (w * t).sum(axis=1) / w_sum  # (B,F)
    x_mean = np.nansum(xv * w, axis=1) / w_sum  # (B,F)
    t_c = t - t_mean[:, None, :]  # (B,T,F)
    x_c = xv - x_mean[:, None, :]  # (B,T,F)
    cov = np.nansum(w * t_c * x_c, axis=1)  # (B,F)
    var = np.nansum(w * (t_c**2), axis=1)  # (B,F)
    slope = cov / np.where(var > 0, var, np.nan)
    return slope


def _interp_resample_1d(y: np.ndarray, *, out_len: int) -> np.ndarray:
    """
    Linear interpolation resample of a 1D array with NaNs. NaNs are forward/backfilled
    before interpolation; if all NaN -> all NaN.
    """
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    if n == 0:
        return np.full((out_len,), np.nan, dtype=float)
    if np.all(np.isnan(y)):
        return np.full((out_len,), np.nan, dtype=float)

    # Fill NaNs by nearest valid (ffill then bfill).
    y2 = y.copy()
    # forward fill
    last = np.nan
    for i in range(n):
        if np.isnan(y2[i]):
            y2[i] = last
        else:
            last = y2[i]
    # back fill
    last = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(y2[i]):
            y2[i] = last
        else:
            last = y2[i]

    x_old = np.linspace(0.0, 1.0, num=n, dtype=float)
    x_new = np.linspace(0.0, 1.0, num=out_len, dtype=float)
    return np.interp(x_new, x_old, y2).astype(float)


@dataclass(frozen=True)
class PooledStatsEncoder:
    """
    Deterministic baseline embedding:
    - pooled stats per feature across time (mean, std, min, max, last, slope)
    - plus a downsampled “shape” descriptor per feature (resampled curve)
    - optional random projection to fixed dim
    """

    resample_len: int = 32
    include_stats: bool = True
    include_shape: bool = True
    project_dim: int | None = 128
    project_seed: int = 1337

    def config(self) -> dict[str, Any]:
        return {
            "type": "PooledStatsEncoder",
            "resample_len": int(self.resample_len),
            "include_stats": bool(self.include_stats),
            "include_shape": bool(self.include_shape),
            "project_dim": (int(self.project_dim) if self.project_dim is not None else None),
            "project_seed": int(self.project_seed),
        }

    def _project(self, feats: np.ndarray) -> np.ndarray:
        if self.project_dim is None:
            return feats.astype(float, copy=False)
        d_in = int(feats.shape[-1])
        d_out = int(self.project_dim)
        if d_out <= 0:
            raise ValueError("project_dim must be > 0 or None")
        if d_out >= d_in:
            return feats.astype(float, copy=False)

        rng = np.random.default_rng(int(self.project_seed))
        # Gaussian random projection, scaled.
        W = rng.standard_normal(size=(d_in, d_out)).astype(float) / np.sqrt(d_out)
        return feats @ W

    def encode(self, x_seq: np.ndarray, mask_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=float)
        m = np.asarray(mask_seq, dtype=bool)
        if x.ndim != 3 or m.ndim != 3:
            raise ValueError("x_seq and mask_seq must be (batch,T,F)")
        if x.shape != m.shape:
            raise ValueError("x_seq and mask_seq shapes must match")

        B, T, F = x.shape
        x_nan = _masked_nan(x, m)

        parts: list[np.ndarray] = []

        if self.include_stats:
            mean = _nanmean(x_nan, axis=1)
            std = _nanstd(x_nan, axis=1)
            mn = _nanmin(x_nan, axis=1)
            mx = _nanmax(x_nan, axis=1)
            last = _last_valid(x, m)
            slope = _slope(x, m)
            stats = np.concatenate([mean, std, mn, mx, last, slope], axis=1)  # (B, 6F)
            parts.append(stats)

        if self.include_shape:
            out_len = int(self.resample_len)
            if out_len <= 1:
                raise ValueError("resample_len must be >= 2 when include_shape is True")
            curves = np.zeros((B, F * out_len), dtype=float)
            for b in range(B):
                row = []
                for f in range(F):
                    y = x_nan[b, :, f]
                    row.append(_interp_resample_1d(y, out_len=out_len))
                curves[b, :] = np.concatenate(row, axis=0)
            parts.append(curves)

        if not parts:
            raise ValueError("Encoder configured with no features (include_stats and include_shape are both False)")

        feats = np.concatenate(parts, axis=1).astype(float, copy=False)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        emb = self._project(feats)
        # Normalize so cosine similarity is stable.
        emb = l2_normalize(emb, axis=-1)
        return emb.astype(np.float32, copy=False)

