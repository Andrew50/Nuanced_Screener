from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCNConfig:
    in_features: int
    d_model: int = 128
    num_blocks: int = 8
    kernel_size: int = 3
    dropout: float = 0.1


try:  # keep module importable without torch
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch():  # noqa: ANN001
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for SSL models. Install with: pip install -e '.[ml]'")
    return torch, nn


def masked_mean_pool(x, mask_time):  # noqa: ANN001
    """
    Masked mean over time.

    x: (B, T, D)
    mask_time: (B, T) bool (True=keep)
    """
    torch2, _nn = _require_torch()
    if mask_time is None:
        return x.mean(dim=1)
    m = mask_time.to(dtype=x.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)


BaseModule = nn.Module if nn is not None else object


class TCNBlock(BaseModule):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, dilation: int, dropout: float):
        torch2, nn2 = _require_torch()
        super().__init__()  # type: ignore[misc]
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for 'same' padding")
        pad = (kernel_size - 1) * int(dilation) // 2
        self.net = nn2.Sequential(
            nn2.Conv1d(int(in_ch), int(out_ch), kernel_size=int(kernel_size), dilation=int(dilation), padding=int(pad)),
            nn2.GELU(),
            nn2.Dropout(p=float(dropout)),
            nn2.Conv1d(
                int(out_ch), int(out_ch), kernel_size=int(kernel_size), dilation=int(dilation), padding=int(pad)
            ),
            nn2.GELU(),
            nn2.Dropout(p=float(dropout)),
        )
        self.proj = (
            nn2.Identity()
            if int(in_ch) == int(out_ch)
            else nn2.Conv1d(int(in_ch), int(out_ch), kernel_size=1)
        )

    def forward(self, x):  # noqa: ANN001
        return self.net(x) + self.proj(x)


class TCNEncoder(BaseModule):
    """
    Temporal CNN encoder producing:
    - per-timestep representations: (B,T,D)
    - pooled window embedding: (B,D)
    """

    def __init__(self, cfg: TCNConfig):
        torch2, nn2 = _require_torch()
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg

        self.in_proj = nn2.Conv1d(int(cfg.in_features), int(cfg.d_model), kernel_size=1)
        blocks: list[TCNBlock] = []
        for i in range(int(cfg.num_blocks)):
            dilation = 2**i
            blocks.append(
                TCNBlock(
                    int(cfg.d_model),
                    int(cfg.d_model),
                    kernel_size=int(cfg.kernel_size),
                    dilation=int(dilation),
                    dropout=float(cfg.dropout),
                )
            )
        self.blocks = nn2.ModuleList(blocks)
        self.out_norm = nn2.LayerNorm(int(cfg.d_model))

    def forward_seq(self, x):  # noqa: ANN001
        """
        x: (B,T,F) float
        returns: (B,T,D)
        """
        if x.ndim != 3:
            raise ValueError("x must be (B,T,F)")
        # Conv1d expects (B,C,T)
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        # Residual stack
        for block in self.blocks:
            h = block(h)
        # Back to (B,T,D)
        h = h.transpose(1, 2)
        # LayerNorm over D
        h = self.out_norm(h)
        return h

    def encode(self, x, mask_time=None):  # noqa: ANN001
        h = self.forward_seq(x)
        return masked_mean_pool(h, mask_time=mask_time)

