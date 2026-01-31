from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from ..normalization import StandardBatch, build_standard_batch_from_windowed_long
from ..progress import progress_ctx
from ..windowed_dataset import WindowedBuildSpec, build_windowed_bars
from ..calendar_utils import TradingCalendar
from ..config import LoaderConfig
from ..labels import load_labels_csv
from .features import ShapeFeatureSpec, build_shape_features_from_ohlcv_batch
from .runner_classifier import _MLPHead, _load_encoder_from_dir, _masked_time, _require_torch

if TYPE_CHECKING:  # pragma: no cover
    from ..model_registry import TrainedArtifact, WarmArtifact


def _read_head_config(run_dir: Path) -> dict:
    p = run_dir / "head_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _read_optional_temperature(run_dir: Path) -> float | None:
    p = run_dir / "calibration.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        t = float(payload.get("temperature"))
        if t > 0:
            return t
    except Exception:
        return None
    return None


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray, *, max_iter: int = 200) -> float:
    """
    Fit a single temperature scalar T>0 via NLL minimization:
      p = sigmoid(logits / T)
    """
    torch, nn = _require_torch()
    device = torch.device("cpu")
    logit_t = torch.from_numpy(np.asarray(logits, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y_true, dtype=np.float32)).to(device)
    # Optimize log(T) for positivity and stability.
    logT = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=int(max_iter))
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():  # noqa: ANN001
        opt.zero_grad(set_to_none=True)
        T = torch.exp(logT).clamp(min=1e-3, max=1e3)
        loss = loss_fn(logit_t / T, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(logT).detach().cpu().item())
    return float(max(1e-3, min(1e3, T)))


class TorchSSLHeadStudentRunner:
    """
    Stack-3 student: reuse Stack-1 SSL encoder, train a head using soft pseudo-labels + weights.

    - Encoder loaded from encoder_dir.txt (same convention as ssl_tcn_classifier)
    - Targets from StandardBatch.meta[target_column] when present, else from b.y
    - Weights from StandardBatch.meta[weight_column] when present, else 1.0
    - Optional: unfreeze last N encoder blocks and finetune end-to-end
    - Optional: temperature scaling calibration on a small gold set (calibration_labels_csv.txt)
    """

    name: str = "torch_ssl_head_student"
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
            raise ValueError("torch_ssl_head_student requires encoder_dir.txt in run_dir (set via --encoder-dir)")
        encoder_dir = Path(enc_dir_txt.read_text(encoding="utf-8").strip())
        encoder, info = _load_encoder_from_dir(encoder_dir)
        schema = info["schema"]

        cfg = _read_head_config(run_dir)
        head_epochs = int(cfg.get("head_epochs", 10))
        head_lr = float(cfg.get("head_lr", 1e-3))
        head_hidden = int(cfg.get("head_hidden_dim", 128))
        head_dropout = float(cfg.get("head_dropout", 0.1))
        target_column = str(cfg.get("target_column", "p_label"))
        weight_column = str(cfg.get("weight_column", "weight"))
        unfreeze_last_n = int(cfg.get("unfreeze_last_n_blocks", 0))
        batch_size = int(cfg.get("batch_size", 256))

        # Build feature spec from schema.
        include_pos = "pos" in set(schema.feature_names)
        feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))

        # Collect all samples into a single tensor batch (current pipeline already uses single-batch splits).
        all_shape: list[StandardBatch] = []
        for b in batches:
            shape_b = build_shape_features_from_ohlcv_batch(b, spec=feat_spec)
            # Normalize on shape features (schema-enforced).
            from ..normalization import normalize_batch

            shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]
            all_shape.append(shape_b)
        if not all_shape:
            raise ValueError("No batches provided")
        if len(all_shape) != 1:
            # Keep it simple; CLI currently provides one batch anyway.
            raise ValueError("torch_ssl_head_student currently expects a single StandardBatch")

        shape_b = all_shape[0]
        x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
        mt = _masked_time(shape_b.mask_seq)

        # Targets + weights.
        y_hard = np.asarray(shape_b.y).astype(float)
        y_soft = None
        if target_column in shape_b.meta:
            try:
                y_soft = np.asarray(shape_b.meta[target_column]).astype(float)
            except Exception:
                y_soft = None
        if y_soft is None:
            y = y_hard
        else:
            y = np.where(np.isfinite(y_soft), y_soft, y_hard)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        w = None
        if weight_column in shape_b.meta:
            try:
                w = np.asarray(shape_b.meta[weight_column]).astype(float)
            except Exception:
                w = None
        if w is None:
            w = np.ones_like(y, dtype=np.float32)
        else:
            w = np.where(np.isfinite(w), w, 1.0).astype(np.float32)
            w = np.clip(w, 0.0, 1e6).astype(np.float32)

        torch, nn = _require_torch()
        device = torch.device(str(cfg.get("device", "cpu")))
        encoder.to(device)
        head = _MLPHead(in_dim=int(schema.encoder.get("d_model", 128)), hidden_dim=int(head_hidden), dropout=float(head_dropout)).to(
            device
        )

        # Freeze encoder by default; optionally unfreeze last N blocks.
        for p in encoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            try:
                # Unfreeze last N TCN blocks + output norm.
                for blk in list(encoder.blocks)[-int(unfreeze_last_n) :]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in encoder.out_norm.parameters():
                    p.requires_grad = True
            except Exception:
                # If encoder does not have expected structure, fall back to fully frozen.
                pass

        params = list(head.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=float(head_lr))
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        Xt = torch.from_numpy(x_in).to(device)
        Mt = torch.from_numpy(mt).to(device)
        Yt = torch.from_numpy(y).to(device)
        Wt = torch.from_numpy(w).to(device)

        n = int(len(y))
        bs = max(1, min(int(batch_size), n))
        rng = np.random.default_rng(int(cfg.get("train_seed", 1337)))
        encoder.train(bool(unfreeze_last_n > 0))
        head.train(True)

        steps_per_epoch = (int(n) + int(bs) - 1) // int(bs) if n else 0
        total_steps = int(head_epochs) * int(steps_per_epoch)
        step_n = 0
        with progress_ctx(transient=False) as progress:
            ep_task = progress.add_task("Head-student epochs", total=int(head_epochs))
            step_task = progress.add_task("Head-student steps", total=int(total_steps))

            for ep in range(int(head_epochs)):
                progress.update(ep_task, completed=ep, description=f"Head-student epochs (epoch {ep+1}/{head_epochs})")
                order = rng.permutation(n)
                for i0 in range(0, n, bs):
                    idx = order[i0 : i0 + bs]
                    xmb = Xt[idx]
                    mmb = Mt[idx]
                    ymb = Yt[idx]
                    wmb = Wt[idx]
                    opt.zero_grad(set_to_none=True)
                    # Encode: allow gradients only if some encoder params are trainable.
                    if any(p.requires_grad for p in encoder.parameters()):
                        emb = encoder.encode(xmb, mask_time=mmb)
                    else:
                        with torch.no_grad():
                            emb = encoder.encode(xmb, mask_time=mmb)
                    logits = head(emb).squeeze(-1)
                    loss_vec = loss_fn(logits, ymb) * wmb
                    loss = loss_vec.mean()
                    loss.backward()
                    opt.step()

                    step_n += 1
                    if step_n % 10 == 0:
                        progress.update(step_task, advance=1, description=f"Head-student steps (loss={float(loss.detach().cpu().item()):.4f})")
                    else:
                        progress.update(step_task, advance=1)

            progress.update(ep_task, completed=int(head_epochs))

        # Save artifacts.
        head_path = run_dir / "head.pt"
        torch.save(head.state_dict(), head_path)
        enc_ft_path = None
        if any(p.requires_grad for p in encoder.parameters()):
            enc_ft_path = run_dir / "encoder_finetuned.pt"
            torch.save(encoder.state_dict(), enc_ft_path)

        # Optional calibration on a gold set (labels CSV path stored in run_dir).
        calib_temp = None
        calib_txt = run_dir / "calibration_labels_csv.txt"
        if calib_txt.exists():
            try:
                calib_csv = Path(calib_txt.read_text(encoding="utf-8").strip())
                if calib_csv.exists():
                    calib_temp = self._calibrate_temperature(
                        run_dir=run_dir,
                        repo_root=run_dir.parents[4],
                        labels_csv=calib_csv,
                        encoder_dir=encoder_dir,
                        encoder_finetuned_path=enc_ft_path,
                        head_path=head_path,
                        window_size=int(shape_b.x_seq.shape[1]),
                    )
            except Exception:
                calib_temp = None

        trained_path = run_dir / "trained.json"
        trained_path.write_text(
            json.dumps(
                {
                    "runner": self.name,
                    "encoder_dir": str(encoder_dir),
                    "encoder_finetuned_path": str(enc_ft_path) if enc_ft_path is not None else None,
                    "schema_fingerprint": info["schema_fingerprint"],
                    "schema_path": str(encoder_dir / "schema.json"),
                    "head_path": str(head_path),
                    "head_config": {
                        "head_epochs": head_epochs,
                        "head_lr": head_lr,
                        "head_hidden_dim": head_hidden,
                        "head_dropout": head_dropout,
                        "device": str(cfg.get("device", "cpu")),
                        "train_seed": int(cfg.get("train_seed", 1337)),
                        "target_column": target_column,
                        "weight_column": weight_column,
                        "unfreeze_last_n_blocks": unfreeze_last_n,
                        "batch_size": bs,
                    },
                    "calibration": {"temperature": float(calib_temp)} if calib_temp is not None else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return TrainedArtifact(runner_name=self.name, path=trained_path)

    def _calibrate_temperature(
        self,
        *,
        run_dir: Path,
        repo_root: Path,
        labels_csv: Path,
        encoder_dir: Path,
        encoder_finetuned_path: Path | None,
        head_path: Path,
        window_size: int,
    ) -> float:
        """
        Build a calibration batch from gold labels and fit a temperature scalar.
        """
        torch, _nn = _require_torch()
        device = torch.device("cpu")

        # Load gold labels and build leakage-safe windows.
        cal = TradingCalendar("NYSE")
        labels_res = load_labels_csv(labels_csv, cal=cal, resolve_non_trading="previous")
        df = labels_res.df.copy()
        # Keep all setups, calibration is task-agnostic (still binary).
        cfg = LoaderConfig(repo_root=repo_root, window_size=int(window_size))
        spec = WindowedBuildSpec(
            window_size=int(window_size),
            feature_columns=tuple(),
            sample_meta_columns=tuple(),
            mask_current_day_to_open_only=True,
            require_full_window=True,
        )
        windowed_path = build_windowed_bars(df, config=cfg, spec=spec, out_path=run_dir / "calibration_windowed.parquet", reuse_if_unchanged=False, cal=cal)
        windowed_long = pd.read_parquet(windowed_path)
        if windowed_long.empty:
            raise RuntimeError("Calibration windowed dataset is empty")
        batch = build_standard_batch_from_windowed_long(
            windowed_long,
            feature_columns=["open", "high", "low", "close", "volume"],
            window_size=int(window_size),
        )

        # Build shape features + normalize per encoder schema.
        encoder, info = _load_encoder_from_dir(encoder_dir)
        if encoder_finetuned_path is not None and Path(encoder_finetuned_path).exists():
            sd = torch.load(Path(encoder_finetuned_path), map_location="cpu")
            encoder.load_state_dict(sd)
        schema = info["schema"]
        include_pos = "pos" in set(schema.feature_names)
        feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))
        shape_b = build_shape_features_from_ohlcv_batch(batch, spec=feat_spec)
        from ..normalization import normalize_batch

        shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]

        x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
        mt = _masked_time(shape_b.mask_seq)

        # Load head.
        head = _MLPHead(in_dim=int(schema.encoder.get("d_model", 128)), hidden_dim=int(_read_head_config(run_dir).get("head_hidden_dim", 128)), dropout=0.0).to(device)
        head_sd = torch.load(head_path, map_location="cpu")
        head.load_state_dict(head_sd)
        head.eval()

        encoder.to(device)
        encoder.eval()
        xt = torch.from_numpy(x_in).to(device)
        mtt = torch.from_numpy(mt).to(device)
        with torch.no_grad():
            emb = encoder.encode(xt, mask_time=mtt)
            logits = head(emb).squeeze(-1).detach().cpu().numpy().astype(float)
        y_true = np.asarray(shape_b.y).astype(int)

        T = _fit_temperature(logits, y_true)
        (run_dir / "calibration.json").write_text(json.dumps({"temperature": float(T), "n": int(len(y_true))}, indent=2) + "\n", encoding="utf-8")
        return float(T)

    def predict(self, batches: Iterable[StandardBatch], *, artifact: TrainedArtifact) -> pd.DataFrame:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        encoder_dir = Path(payload["encoder_dir"])
        encoder, info = _load_encoder_from_dir(encoder_dir)
        schema = info["schema"]

        torch, _nn = _require_torch()
        device = torch.device("cpu")
        encoder.to(device)
        encoder.eval()

        # Optional finetuned encoder weights.
        enc_ft = payload.get("encoder_finetuned_path")
        if enc_ft:
            p = Path(enc_ft)
            if p.exists():
                sd = torch.load(p, map_location="cpu")
                encoder.load_state_dict(sd)
                encoder.eval()

        include_pos = "pos" in set(schema.feature_names)
        feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))

        head = _MLPHead(
            in_dim=int(schema.encoder.get("d_model", 128)),
            hidden_dim=int(payload.get("head_config", {}).get("head_hidden_dim", 128)),
            dropout=0.0,
        ).to(device)
        head_sd = torch.load(Path(payload["head_path"]), map_location="cpu")
        head.load_state_dict(head_sd)
        head.eval()

        T = _read_optional_temperature(artifact.path.parent)
        if T is None:
            T = 1.0

        rows = []
        for b in batches:
            shape_b = build_shape_features_from_ohlcv_batch(b, spec=feat_spec)
            from ..normalization import normalize_batch

            shape_b = normalize_batch(shape_b, mode=str(schema.normalization), global_stats=None)  # type: ignore[arg-type]
            x_in = np.where(shape_b.mask_seq, shape_b.x_seq, 0.0).astype(np.float32)
            mt = _masked_time(shape_b.mask_seq)
            xt = torch.from_numpy(x_in).to(device)
            mtt = torch.from_numpy(mt).to(device)
            with torch.no_grad():
                emb = encoder.encode(xt, mask_time=mtt)
                logits = head(emb).squeeze(-1) / float(T)
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

