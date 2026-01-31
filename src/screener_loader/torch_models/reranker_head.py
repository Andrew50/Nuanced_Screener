from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from ..normalization import StandardBatch
from ..progress import progress_ctx

if TYPE_CHECKING:  # pragma: no cover
    from ..model_registry import TrainedArtifact, WarmArtifact


def _require_torch():  # noqa: ANN001
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required for torch models. Install with: pip install -e '.[ml]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e
    return torch, nn


def _masked_time(mask_seq: np.ndarray) -> np.ndarray:
    """
    Reduce (B,T,F) bool to (B,T) bool.
    """
    if mask_seq.ndim != 3:
        raise ValueError("mask_seq must be (B,T,F)")
    return np.asarray(mask_seq, dtype=bool).any(axis=-1)


def _cand_meta_columns_from_batch(batch: StandardBatch) -> list[str]:
    cols = []
    for k in batch.meta.keys():
        if str(k).startswith("cand_"):
            cols.append(str(k))
    return sorted(cols)


def _stack_cand_meta(batch: StandardBatch, cols: list[str]) -> np.ndarray:
    """
    Build a dense (N, M) float matrix from per-sample candidate meta columns.
    Missing columns default to 0.0.
    NaNs are filled with 0.0.
    """
    n = int(batch.x_seq.shape[0])
    if not cols:
        return np.zeros((n, 0), dtype=np.float32)
    mats = []
    for c in cols:
        v = batch.meta.get(c)
        if v is None:
            mats.append(np.zeros((n,), dtype=np.float32))
            continue
        arr = np.asarray(v)
        # Try numeric conversion; non-numeric becomes NaN then 0.
        if arr.dtype.kind not in {"i", "u", "f"}:
            arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(arr)
    return np.stack(mats, axis=1).astype(np.float32, copy=False)


def _encode_with_stack1_tcn(batch: StandardBatch, encoder_dir: Path) -> tuple[np.ndarray, dict]:
    """
    Encode via Stack 1 TCN encoder artifact (schema.json + encoder.pt).

    Returns:
    - emb: (N, D) float32
    - info: dict with schema_fingerprint etc.
    """
    # Import from Stack 1 utilities (torch-only).
    from ..ssl.features import ShapeFeatureSpec, build_shape_features_from_ohlcv_batch
    from ..ssl.runner_classifier import _load_encoder_from_dir  # type: ignore[attr-defined]
    from ..ssl.schema import read_schema

    schema_path = encoder_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema.json in encoder_dir={encoder_dir}")
    schema = read_schema(schema_path)

    # Determine feature spec from schema.
    include_pos = "pos" in set(schema.feature_names)
    feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))

    # Normalize on shape features (schema-enforced).
    from ..normalization import normalize_batch

    shape_b = build_shape_features_from_ohlcv_batch(batch, spec=feat_spec)
    shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]

    x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
    mt = _masked_time(shape_b.mask_seq)

    encoder, info = _load_encoder_from_dir(encoder_dir)
    torch, _nn = _require_torch()
    device = torch.device("cpu")
    encoder.to(device)
    encoder.eval()

    xt = torch.from_numpy(x_in).to(device)
    mtt = torch.from_numpy(mt).to(device)
    with torch.no_grad():
        emb_t = encoder.encode(xt, mask_time=mtt)  # (N,D)
    emb = emb_t.detach().cpu().numpy().astype(np.float32)
    return emb, {"schema_fingerprint": info.get("schema_fingerprint")}


def _encode_fallback_masked_mean(batch: StandardBatch) -> np.ndarray:
    """
    Torch-free fallback encoder: masked mean over time -> (N, F).
    """
    x = np.asarray(batch.x_seq, dtype=float)
    m = np.asarray(batch.mask_seq, dtype=bool)
    # mean over time axis where any feature is valid at that time.
    # Keep per-feature mean, ignoring masked values.
    x2 = np.where(m, x, np.nan)
    emb = np.nanmean(x2, axis=1)  # (N,F)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return emb


class _HeadMLP:  # noqa: ANN001
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        torch, nn = _require_torch()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
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


@dataclass(frozen=True)
class RerankerHeadConfig:
    encoder_dir: str | None
    cand_meta_columns: list[str]
    head_hidden_dim: int
    head_dropout: float


class TorchRerankerHeadRunner:
    """
    Stack 2 reranker head:

    - Consumes embeddings from a Stack 1 encoder directory (schema.json + encoder.pt),
      and concatenates candidate meta features (cand_*) from StandardBatch.meta.
    - Trains only a small MLP head. No SSL/warm stage here.
    """

    name: str = "torch_reranker_head"
    task_type: str = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> WarmArtifact:
        raise RuntimeError(f"{self.name} does not support warm()")

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> TrainedArtifact:
        from ..model_registry import TrainedArtifact

        torch, nn = _require_torch()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Runner-specific config files (optional but useful).
        enc_dir_txt = run_dir / "encoder_dir.txt"
        encoder_dir: Path | None = None
        if enc_dir_txt.exists():
            txt = enc_dir_txt.read_text(encoding="utf-8").strip()
            if txt:
                encoder_dir = Path(txt)

        head_cfg_path = run_dir / "head_config.json"
        head_cfg = {}
        if head_cfg_path.exists():
            head_cfg = json.loads(head_cfg_path.read_text(encoding="utf-8"))

        hidden_dim = int(head_cfg.get("head_hidden_dim", 128))
        dropout = float(head_cfg.get("head_dropout", 0.1))
        epochs = int(head_cfg.get("head_epochs", 10))
        lr = float(head_cfg.get("head_lr", 1e-3))
        device_str = str(head_cfg.get("device", "cpu"))

        # Collect data.
        Xs: list[np.ndarray] = []
        Ms: list[np.ndarray] = []
        Ys: list[np.ndarray] = []

        cand_cols: list[str] | None = None
        schema_fp = None
        for b in batches:
            if cand_cols is None:
                cand_cols = _cand_meta_columns_from_batch(b)
            cand = _stack_cand_meta(b, cols=cand_cols)
            if encoder_dir is not None:
                emb, info = _encode_with_stack1_tcn(b, encoder_dir)
                schema_fp = info.get("schema_fingerprint")
            else:
                emb = _encode_fallback_masked_mean(b)
            Xs.append(emb)
            Ms.append(cand)
            Ys.append(np.asarray(b.y).astype(int))

        X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
        M = np.concatenate(Ms, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(Ys, axis=0).astype(np.float32, copy=False)
        if X.shape[0] != y.shape[0] or M.shape[0] != y.shape[0]:
            raise RuntimeError("Batch size mismatch while training reranker head")

        Z = np.concatenate([X, M], axis=1).astype(np.float32, copy=False)
        in_dim = int(Z.shape[1])

        device = torch.device(device_str)
        head = _HeadMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device).train(True)
        opt = torch.optim.Adam(head.parameters(), lr=float(lr))

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

        Zt = torch.from_numpy(Z).to(device)
        yt = torch.from_numpy(y).to(device)

        bs = min(256, int(len(y))) if len(y) else 1
        rng = np.random.default_rng(int(head_cfg.get("train_seed", 1337)))
        steps_per_epoch = (int(len(y)) + int(bs) - 1) // int(bs) if len(y) else 0
        total_steps = int(epochs) * int(steps_per_epoch)
        step_n = 0
        with progress_ctx(transient=False) as progress:
            ep_task = progress.add_task("Reranker head epochs", total=int(epochs))
            step_task = progress.add_task("Reranker head steps", total=int(total_steps))

            for ep in range(int(epochs)):
                progress.update(ep_task, completed=ep, description=f"Reranker head epochs (epoch {ep+1}/{epochs})")
                order = rng.permutation(len(y))
                for i0 in range(0, len(y), bs):
                    idx = order[i0 : i0 + bs]
                    xmb = Zt[idx]
                    ymb = yt[idx]
                    opt.zero_grad(set_to_none=True)
                    logits = head(xmb).squeeze(-1)
                    loss = loss_fn(logits, ymb)
                    loss.backward()
                    opt.step()

                    step_n += 1
                    if step_n % 10 == 0:
                        progress.update(step_task, advance=1, description=f"Reranker head steps (loss={float(loss.detach().cpu().item()):.4f})")
                    else:
                        progress.update(step_task, advance=1)

            progress.update(ep_task, completed=int(epochs))

        head_path = run_dir / "head.pt"
        torch.save(head.state_dict(), head_path)

        cfg = RerankerHeadConfig(
            encoder_dir=str(encoder_dir) if encoder_dir is not None else None,
            cand_meta_columns=list(cand_cols or []),
            head_hidden_dim=int(hidden_dim),
            head_dropout=float(dropout),
        )
        trained_path = run_dir / "reranker_head.json"
        trained_path.write_text(
            json.dumps(
                {
                    "runner": self.name,
                    "encoder_dir": cfg.encoder_dir,
                    "schema_fingerprint": schema_fp,
                    "cand_meta_columns": cfg.cand_meta_columns,
                    "head_path": str(head_path),
                    "head_config": {
                        "head_hidden_dim": cfg.head_hidden_dim,
                        "head_dropout": cfg.head_dropout,
                        "head_epochs": int(epochs),
                        "head_lr": float(lr),
                        "device": str(device_str),
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
        encoder_dir = payload.get("encoder_dir")
        encoder_dir_p = Path(encoder_dir) if encoder_dir else None
        cand_cols = list(payload.get("cand_meta_columns") or [])

        torch, _nn = _require_torch()
        device = torch.device("cpu")

        head_sd = torch.load(Path(payload["head_path"]), map_location="cpu")
        # Infer in_dim from first layer weight.
        first_w = head_sd.get("0.weight")
        if first_w is None:
            raise ValueError("Invalid head state dict: missing first layer weights")
        in_dim = int(first_w.shape[1])
        head = _HeadMLP(
            in_dim=in_dim,
            hidden_dim=int(payload.get("head_config", {}).get("head_hidden_dim", 128)),
            dropout=0.0,
        ).to(device)
        head.load_state_dict(head_sd)
        head.eval()

        rows = []
        for b in batches:
            if not cand_cols:
                cand_cols = _cand_meta_columns_from_batch(b)
            cand = _stack_cand_meta(b, cols=cand_cols)
            if encoder_dir_p is not None:
                emb, _info = _encode_with_stack1_tcn(b, encoder_dir_p)
            else:
                emb = _encode_fallback_masked_mean(b)
            Z = np.concatenate([emb, cand], axis=1).astype(np.float32, copy=False)
            zt = torch.from_numpy(Z).to(device)
            with torch.no_grad():
                logits = head(zt).squeeze(-1)
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

