from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CropConfig:
    crop_lengths: tuple[int, ...] = (64, 96)


def random_crop(x: np.ndarray, mask: np.ndarray, *, rng: np.random.Generator, cfg: CropConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly crop along time dimension (axis=1).
    """
    if x.ndim != 3:
        raise ValueError("x must be (B,T,F)")
    if mask.shape != x.shape:
        raise ValueError("mask must match x shape")
    T = int(x.shape[1])
    choices = [int(L) for L in cfg.crop_lengths if int(L) > 0 and int(L) <= T]
    if not choices:
        return x, mask
    L = int(choices[int(rng.integers(0, len(choices)))])
    if L == T:
        return x, mask
    start = int(rng.integers(0, T - L + 1))
    return x[:, start : start + L, :], mask[:, start : start + L, :]


@dataclass(frozen=True)
class JitterConfig:
    sigma: float = 0.0


def jitter(x: np.ndarray, mask: np.ndarray, *, rng: np.random.Generator, cfg: JitterConfig) -> np.ndarray:
    """
    Add small iid Gaussian noise to observed entries (mask=True).
    """
    sigma = float(cfg.sigma)
    if sigma <= 0:
        return x
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(float)
    return np.where(mask, x + noise, x)


@dataclass(frozen=True)
class FeatureDropoutConfig:
    drop_prob: float = 0.0
    # indices in feature dimension to consider dropping
    feature_indices: tuple[int, ...] = ()


def feature_dropout(x: np.ndarray, mask: np.ndarray, *, rng: np.random.Generator, cfg: FeatureDropoutConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly drop some whole feature channels (set to NaN/invalid) for the entire window.
    """
    p = float(cfg.drop_prob)
    if p <= 0 or not cfg.feature_indices:
        return x, mask
    out_x = x.copy()
    out_m = mask.copy()
    for fi in cfg.feature_indices:
        if rng.random() < p:
            out_x[:, :, fi] = np.nan
            out_m[:, :, fi] = False
    return out_x, out_m


@dataclass(frozen=True)
class CensorConfig:
    """
    Optional augmentation: simulate inference-time censoring on the last timestep.
    """

    prob: float = 0.0
    # feature indices to censor at last timestep (default: all features)
    feature_indices: tuple[int, ...] = ()


def maybe_censor_last_timestep(
    x: np.ndarray, mask: np.ndarray, *, rng: np.random.Generator, cfg: CensorConfig
) -> tuple[np.ndarray, np.ndarray]:
    p = float(cfg.prob)
    if p <= 0:
        return x, mask
    if rng.random() >= p:
        return x, mask
    out_x = x.copy()
    out_m = mask.copy()
    if out_x.shape[1] <= 0:
        return out_x, out_m
    t = out_x.shape[1] - 1
    idxs: Sequence[int] = cfg.feature_indices if cfg.feature_indices else range(out_x.shape[2])
    for fi in idxs:
        out_x[:, t, fi] = np.nan
        out_m[:, t, fi] = False
    return out_x, out_m

