from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from ..normalization import StandardBatch, normalize_batch
from ..progress import progress_ctx
from .features import ShapeFeatureSpec, build_shape_features_from_ohlcv_batch
from .schema import read_schema
from .tcn import TCNConfig, TCNEncoder

if TYPE_CHECKING:  # pragma: no cover
    from ..model_registry import TrainedArtifact, WarmArtifact


def _require_torch():  # noqa: ANN001
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required for SSL models. Install with: pip install -e '.[ml]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e
    return torch, nn


def _load_encoder_from_dir(encoder_dir: Path) -> tuple[TCNEncoder, dict]:
    schema_path = encoder_dir / "schema.json"
    enc_path = encoder_dir / "encoder.pt"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema.json in encoder_dir={encoder_dir}")
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing encoder.pt in encoder_dir={encoder_dir}")

    schema = read_schema(schema_path)
    enc_info = schema.encoder or {}
    if (enc_info.get("type") or "").strip().lower() != "tcn":
        raise ValueError(f"Unsupported encoder type in schema: {enc_info.get('type')!r}")

    cfg = TCNConfig(
        in_features=int(enc_info.get("in_features")),
        d_model=int(enc_info.get("d_model", 128)),
        num_blocks=int(enc_info.get("num_blocks", 8)),
        kernel_size=int(enc_info.get("kernel_size", 3)),
        dropout=float(enc_info.get("dropout", 0.1)),
    )
    encoder = TCNEncoder(cfg)
    torch, _nn = _require_torch()
    sd = torch.load(enc_path, map_location="cpu")
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder, {"schema": schema, "schema_fingerprint": schema.fingerprint()}


def _masked_time(mask_seq: np.ndarray) -> np.ndarray:
    """
    Reduce (B,T,F) bool to (B,T) bool.
    """
    if mask_seq.ndim != 3:
        raise ValueError("mask_seq must be (B,T,F)")
    return mask_seq.any(axis=-1)


class _MLPHead:  # noqa: ANN001
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        torch, nn = _require_torch()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def to(self, device):  # noqa: ANN001
        self.net.to(device)
        return self

    def train(self, mode: bool = True):  # noqa: ANN001
        self.net.train(mode)
        return self

    def eval(self):  # noqa: ANN001
        return self.train(False)

    def parameters(self):  # noqa: ANN001
        return self.net.parameters()

    def state_dict(self):  # noqa: ANN001
        return self.net.state_dict()

    def load_state_dict(self, sd):  # noqa: ANN001
        self.net.load_state_dict(sd)
        return self

    def __call__(self, x):  # noqa: ANN001
        return self.net(x)


def _read_head_config(run_dir: Path) -> dict:
    p = run_dir / "head_config.json"
    if not p.exists():
        return {"head_epochs": 10, "head_lr": 1e-3, "head_hidden_dim": 128, "train_seed": 1337}
    return json.loads(p.read_text(encoding="utf-8"))


class SSLTCNClassifierRunner:
    """
    Finetune runner that consumes a pretrained SSL encoder and trains a small head.
    """

    name: str = "ssl_tcn_classifier"
    task_type: str = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> WarmArtifact:
        raise RuntimeError(f"{self.name} does not support warm()")

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> TrainedArtifact:
        from ..model_registry import TrainedArtifact

        run_dir.mkdir(parents=True, exist_ok=True)
        enc_dir_txt = run_dir / "encoder_dir.txt"
        if not enc_dir_txt.exists():
            raise ValueError("ssl_tcn_classifier requires encoder_dir.txt in run_dir (set via --encoder-dir)")
        encoder_dir = Path(enc_dir_txt.read_text(encoding="utf-8").strip())
        encoder, info = _load_encoder_from_dir(encoder_dir)
        schema = info["schema"]
        cfg = _read_head_config(run_dir)

        # Build feature spec from schema.
        include_pos = "pos" in set(schema.feature_names)
        feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))

        # Collect embeddings + labels (single-batch is fine for now).
        xs = []
        ys = []
        metas = []

        torch, nn = _require_torch()
        device = torch.device(str(cfg.get("device", "cpu")))
        encoder.to(device)
        encoder.eval()

        for b in batches:
            shape_b = build_shape_features_from_ohlcv_batch(b, spec=feat_spec)
            # Normalize on shape features (schema-enforced).
            shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]
            x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
            mt = _masked_time(shape_b.mask_seq)
            xt = torch.from_numpy(x_in).to(device)
            mtt = torch.from_numpy(mt).to(device)
            with torch.no_grad():
                emb = encoder.encode(xt, mask_time=mtt)  # (B,D)
            xs.append(emb.detach().cpu().numpy())
            ys.append(np.asarray(b.y).astype(int))
            metas.append(b.meta)

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0).astype(int)
        if X.shape[0] != y.shape[0]:
            raise RuntimeError("Embedding/label size mismatch")

        # Train head.
        head_epochs = int(cfg.get("head_epochs", 10))
        head_lr = float(cfg.get("head_lr", 1e-3))
        head_hidden = int(cfg.get("head_hidden_dim", 128))
        head_dropout = float(cfg.get("head_dropout", 0.1))

        head = _MLPHead(in_dim=int(X.shape[1]), hidden_dim=int(head_hidden), dropout=float(head_dropout)).to(device).train(True)
        opt = torch.optim.Adam(head.parameters(), lr=head_lr)

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        yt = torch.from_numpy(y.astype(np.float32)).to(device)

        # Simple minibatching.
        bs_cfg = int(cfg.get("batch_size", 256))
        bs = max(1, min(int(bs_cfg), int(len(y)))) if len(y) else 1
        rng = np.random.default_rng(int(cfg.get("train_seed", 1337)))
        steps_per_epoch = (int(len(y)) + int(bs) - 1) // int(bs) if len(y) else 0
        total_steps = int(head_epochs) * int(steps_per_epoch)
        step_n = 0
        with progress_ctx(transient=False) as progress:
            ep_task = progress.add_task("SSL finetune epochs", total=int(head_epochs))
            step_task = progress.add_task("SSL finetune steps", total=int(total_steps))

            for ep in range(int(head_epochs)):
                progress.update(ep_task, completed=ep, description=f"SSL finetune epochs (epoch {ep+1}/{head_epochs})")
                order = rng.permutation(len(y))
                for i0 in range(0, len(y), bs):
                    idx = order[i0 : i0 + bs]
                    xmb = Xt[idx]
                    ymb = yt[idx]
                    opt.zero_grad(set_to_none=True)
                    logits = head(xmb).squeeze(-1)
                    loss = loss_fn(logits, ymb)
                    loss.backward()
                    opt.step()

                    step_n += 1
                    if step_n % 10 == 0:
                        progress.update(step_task, advance=1, description=f"SSL finetune steps (loss={float(loss.detach().cpu().item()):.4f})")
                    else:
                        progress.update(step_task, advance=1)

            progress.update(ep_task, completed=int(head_epochs))

        head_path = run_dir / "head.pt"
        torch.save(head.state_dict(), head_path)

        trained_path = run_dir / "trained.json"
        trained_path.write_text(
            json.dumps(
                {
                    "runner": self.name,
                    "encoder_dir": str(encoder_dir),
                    "schema_fingerprint": info["schema_fingerprint"],
                    "schema_path": str(encoder_dir / "schema.json"),
                    "head_path": str(head_path),
                    "head_config": {
                        "head_epochs": head_epochs,
                        "head_lr": head_lr,
                        "head_hidden_dim": head_hidden,
                        "head_dropout": float(head_dropout),
                        "batch_size": int(bs),
                        "device": str(cfg.get("device", "cpu")),
                        "train_seed": int(cfg.get("train_seed", 1337)),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return TrainedArtifact(runner_name=self.name, path=trained_path)

    def predict(self, batches: Iterable[StandardBatch], *, artifact: TrainedArtifact) -> pd.DataFrame:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        encoder_dir = Path(payload["encoder_dir"])
        encoder, info = _load_encoder_from_dir(encoder_dir)
        schema = info["schema"]
        include_pos = "pos" in set(schema.feature_names)
        feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))

        torch, _nn = _require_torch()
        device = torch.device("cpu")
        encoder.to(device)
        encoder.eval()

        head = _MLPHead(
            in_dim=int(schema.encoder.get("d_model", 128)),
            hidden_dim=int(payload.get("head_config", {}).get("head_hidden_dim", 128)),
            dropout=0.0,
        ).to(device)
        head_sd = torch.load(Path(payload["head_path"]), map_location="cpu")
        head.load_state_dict(head_sd)
        head.eval()

        rows = []
        for b in batches:
            shape_b = build_shape_features_from_ohlcv_batch(b, spec=feat_spec)
            shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]
            x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
            mt = _masked_time(shape_b.mask_seq)
            xt = torch.from_numpy(x_in).to(device)
            mtt = torch.from_numpy(mt).to(device)
            with torch.no_grad():
                emb = encoder.encode(xt, mask_time=mtt)
                logits = head(emb).squeeze(-1)
                score = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            y_true = np.asarray(b.y).astype(int)

            for sid, sc, yy, ss, tkr, d in zip(
                sample_ids.tolist(),
                score.tolist(),
                y_true.tolist(),
                (setups.tolist() if setups is not None else [None] * len(y_true)),
                (tickers.tolist() if tickers is not None else [None] * len(y_true)),
                (asof_dates.tolist() if asof_dates is not None else [None] * len(y_true)),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sid),
                        "setup": str(ss) if ss is not None else None,
                        "ticker": str(tkr) if tkr is not None else None,
                        "asof_date": d,
                        "y_true": int(yy),
                        "score": float(sc),
                    }
                )

        return pd.DataFrame(rows)

