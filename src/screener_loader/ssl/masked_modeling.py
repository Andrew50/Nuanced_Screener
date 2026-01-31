from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .tcn import TCNConfig, TCNEncoder, _require_torch


MaskMode = Literal["time", "feature", "both"]
LossType = Literal["huber", "l1"]


@dataclass(frozen=True)
class MaskingConfig:
    mask_mode: tuple[MaskMode, ...] = ("both",)
    mask_rate_time: float = 0.10
    mask_rate_feat: float = 0.15
    use_mask_token: bool = True


@dataclass(frozen=True)
class MaskedModelingConfig:
    encoder: TCNConfig
    masking: MaskingConfig = MaskingConfig()
    loss: LossType = "huber"
    huber_delta: float = 1.0


def _build_mask_to_predict(valid_mask: np.ndarray, *, rng: np.random.Generator, cfg: MaskingConfig) -> np.ndarray:
    """
    valid_mask: (B,T,F) bool
    returns mask_to_predict: (B,T,F) bool
    """
    B, T, F = valid_mask.shape
    out = np.zeros((B, T, F), dtype=bool)

    modes = set(cfg.mask_mode or ())
    if "both" in modes:
        modes = {"time", "feature"}

    if "time" in modes:
        rate = float(cfg.mask_rate_time)
        if rate > 0:
            mt = rng.random((B, T)) < rate
            out |= mt[:, :, None]

    if "feature" in modes:
        rate = float(cfg.mask_rate_feat)
        if rate > 0:
            mf = rng.random((B, T, F)) < rate
            out |= mf

    out &= valid_mask
    return out


def _build_mask_to_predict_torch(valid_mask_t, *, rng: np.random.Generator, cfg: MaskingConfig):  # noqa: ANN001
    """
    Torch-native mask sampling to avoid CPU roundtrips.

    valid_mask_t: (B,T,F) bool torch tensor on target device
    returns: (B,T,F) bool torch tensor on same device
    """
    torch, _nn = _require_torch()
    if valid_mask_t.ndim != 3:
        raise ValueError("valid_mask_t must be (B,T,F)")
    B, T, F = [int(x) for x in valid_mask_t.shape]

    modes = set(cfg.mask_mode or ())
    if "both" in modes:
        modes = {"time", "feature"}

    # Prefer deterministic sampling by seeding a device generator from numpy RNG.
    # If the backend doesn't support device Generators (some builds/backends), fall back to global RNG.
    gen = None
    try:
        seed = int(rng.integers(0, 2**63 - 1))
        gen = torch.Generator(device=valid_mask_t.device)
        gen.manual_seed(seed)
    except Exception:
        gen = None

    out = torch.zeros((B, T, F), dtype=torch.bool, device=valid_mask_t.device)

    if "time" in modes:
        rate = float(cfg.mask_rate_time)
        if rate > 0:
            mt = torch.rand((B, T), device=valid_mask_t.device, generator=gen) < rate
            out |= mt[:, :, None]

    if "feature" in modes:
        rate = float(cfg.mask_rate_feat)
        if rate > 0:
            mf = torch.rand((B, T, F), device=valid_mask_t.device, generator=gen) < rate
            out |= mf

    out &= valid_mask_t.to(dtype=torch.bool)
    return out


class MaskedModelingModel:  # noqa: ANN001
    """
    TCN encoder + per-timestep reconstruction head for masked modeling.

    Forward returns (loss, recon, mask_to_predict).
    """

    def __init__(self, cfg: MaskedModelingConfig):
        torch, nn = _require_torch()
        self.cfg = cfg
        self.encoder = TCNEncoder(cfg.encoder)
        self.recon_head = nn.Linear(int(cfg.encoder.d_model), int(cfg.encoder.in_features))
        self.mask_token = nn.Parameter(torch.zeros(int(cfg.encoder.in_features))) if cfg.masking.use_mask_token else None

    def to(self, device):  # noqa: ANN001
        self.encoder.to(device)
        self.recon_head.to(device)
        if self.mask_token is not None:
            self.mask_token.data = self.mask_token.data.to(device)
        return self

    def train(self, mode: bool = True):  # noqa: ANN001
        self.encoder.train(mode)
        self.recon_head.train(mode)
        return self

    def eval(self):  # noqa: ANN001
        return self.train(False)

    def parameters(self):  # noqa: ANN001
        yield from self.encoder.parameters()
        yield from self.recon_head.parameters()
        if self.mask_token is not None:
            yield self.mask_token

    def state_dict(self):  # noqa: ANN001
        # Prefer standard torch state_dict format for persistence.
        torch, _nn = _require_torch()
        sd = {
            "encoder": self.encoder.state_dict(),
            "recon_head": self.recon_head.state_dict(),
            "cfg": {
                "encoder": self.cfg.encoder.__dict__,
                "masking": self.cfg.masking.__dict__,
                "loss": self.cfg.loss,
                "huber_delta": float(self.cfg.huber_delta),
            },
        }
        if self.mask_token is not None:
            sd["mask_token"] = self.mask_token.detach().cpu()
        return sd

    def load_state_dict(self, sd):  # noqa: ANN001
        self.encoder.load_state_dict(sd["encoder"])
        self.recon_head.load_state_dict(sd["recon_head"])
        if self.mask_token is not None and "mask_token" in sd:
            self.mask_token.data = sd["mask_token"].to(self.mask_token.data.device)
        return self

    def forward(self, x, valid_mask, *, rng: np.random.Generator):  # noqa: ANN001
        """
        x: (B,T,F) float32 torch tensor
        valid_mask: (B,T,F) bool torch tensor (True=observed/usable)
        """
        torch, _nn = _require_torch()
        if x.ndim != 3:
            raise ValueError("x must be (B,T,F)")
        if valid_mask.shape != x.shape:
            raise ValueError("valid_mask must match x shape")

        # Decide which entries to reconstruct (torch-native to avoid CPU sync).
        mask_to_predict_t = _build_mask_to_predict_torch(valid_mask, rng=rng, cfg=self.cfg.masking)

        # Prepare masked input.
        x_in = x
        if self.cfg.masking.use_mask_token and self.mask_token is not None:
            token = self.mask_token.to(dtype=x.dtype, device=x.device)
            x_in = torch.where(mask_to_predict_t, token.view(1, 1, -1), x_in)
        else:
            x_in = torch.where(mask_to_predict_t, torch.zeros_like(x_in), x_in)

        # Encoder -> per-timestep hidden.
        h = self.encoder.forward_seq(x_in)  # (B,T,D)
        recon = self.recon_head(h)  # (B,T,F)

        # Loss only on (masked ∩ valid).
        loss_mask = mask_to_predict_t & valid_mask
        if loss_mask.any():
            if self.cfg.loss == "l1":
                loss = torch.abs(recon - x)[loss_mask].mean()
            else:
                # Huber / smooth L1
                #
                # Avoid relying on torch.nn.smooth_l1_loss (doesn't exist) and keep this
                # compatible even if callers alias torch.nn as "F" by mistake elsewhere.
                d = recon[loss_mask] - x[loss_mask]
                abs_d = torch.abs(d)
                beta = float(self.cfg.huber_delta)
                if beta <= 0:
                    # Degenerates to L1.
                    loss = abs_d.mean()
                else:
                    # Matches torch.nn.functional.smooth_l1_loss(..., beta=beta, reduction="mean").
                    quad = 0.5 * (d * d) / beta
                    lin = abs_d - 0.5 * beta
                    loss = torch.where(abs_d < beta, quad, lin).mean()
        else:
            loss = torch.zeros((), device=x.device, dtype=x.dtype)

        return loss, recon, mask_to_predict_t

