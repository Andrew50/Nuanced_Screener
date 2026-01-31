from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Protocol

import numpy as np
import pandas as pd

from .normalization import StandardBatch


TaskType = Literal["binary_classification", "retrieval", "localization"]


@dataclass(frozen=True)
class WarmArtifact:
    runner_name: str
    path: Path


@dataclass(frozen=True)
class TrainedArtifact:
    runner_name: str
    path: Path


class Runner(Protocol):
    name: str
    task_type: TaskType

    def supports_warm(self) -> bool:
        ...

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> WarmArtifact:
        ...

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> TrainedArtifact:
        ...

    def predict(self, batches: Iterable[StandardBatch], *, artifact: TrainedArtifact) -> pd.DataFrame:
        """
        Return a DataFrame with at least:
        - sample_id (string)
        - score (float; higher => more positive)
        Optionally:
        - y_true
        - setup, ticker, asof_date
        """
        ...


class ConstantPriorRunner:
    """
    Dummy baseline runner: predict P(label=1) per setup based on training prevalence.
    """

    name: str = "dummy_constant_prior"
    task_type: TaskType = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return False

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> WarmArtifact:
        raise RuntimeError(f"{self.name} does not support warm()")

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> TrainedArtifact:
        run_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, list[int]] = {}  # setup -> [pos, total]
        global_pos = 0
        global_total = 0

        for b in batches:
            y = np.asarray(b.y).astype(int)
            setups = np.asarray(b.meta.get("setup"))
            if setups is None:
                raise ValueError("StandardBatch.meta must include 'setup'")
            for yy, ss in zip(y.tolist(), setups.tolist(), strict=True):
                ss = str(ss)
                pos, tot = counts.get(ss, [0, 0])
                counts[ss] = [pos + int(yy == 1), tot + 1]
                global_pos += int(yy == 1)
                global_total += 1

        priors = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in counts.items()}
        global_prior = global_pos / global_total if global_total else 0.0
        payload = {"priors_by_setup": priors, "global_prior": global_prior}

        out = run_dir / "constant_prior.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return TrainedArtifact(runner_name=self.name, path=out)

    def predict(self, batches: Iterable[StandardBatch], *, artifact: TrainedArtifact) -> pd.DataFrame:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        priors_by_setup = payload.get("priors_by_setup", {}) or {}
        global_prior = float(payload.get("global_prior", 0.0))

        rows = []
        for b in batches:
            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            if sample_ids is None or setups is None:
                raise ValueError("StandardBatch.meta must include 'sample_id' and 'setup'")
            y_true = np.asarray(b.y).astype(int)

            for sid, ss, yy, tkr, d in zip(
                sample_ids.tolist(),
                setups.tolist(),
                y_true.tolist(),
                (tickers.tolist() if tickers is not None else [None] * len(y_true)),
                (asof_dates.tolist() if asof_dates is not None else [None] * len(y_true)),
                strict=True,
            ):
                ss = str(ss)
                score = float(priors_by_setup.get(ss, global_prior))
                rows.append(
                    {
                        "sample_id": str(sid),
                        "setup": ss,
                        "ticker": str(tkr) if tkr is not None else None,
                        "asof_date": d,
                        "y_true": int(yy),
                        "score": score,
                    }
                )
        return pd.DataFrame(rows)


class NoOpWarmRunner:
    """
    Warm-capable dummy runner.

    - warm(): writes a small artifact proving the warm pipeline works
    - train()/predict(): trivial baseline (always score 0.0) so `--model-type all` is safe
    """

    name: str = "dummy_warm_noop"
    task_type: TaskType = "binary_classification"

    def supports_warm(self) -> bool:  # noqa: D401
        return True

    def warm(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> WarmArtifact:
        run_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        total_steps = 0
        for b in batches:
            total += int(b.x_seq.shape[0])
            total_steps += int(b.x_seq.shape[0] * b.x_seq.shape[1])
        payload = {"runner": self.name, "warmed_batches": True, "samples": total, "time_steps": total_steps}
        out = run_dir / "warm.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return WarmArtifact(runner_name=self.name, path=out)

    def train(self, batches: Iterable[StandardBatch], *, run_dir: Path) -> TrainedArtifact:
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "trained.json"
        out.write_text(json.dumps({"runner": self.name, "note": "no-op training"}, indent=2) + "\n", encoding="utf-8")
        return TrainedArtifact(runner_name=self.name, path=out)

    def predict(self, batches: Iterable[StandardBatch], *, artifact: TrainedArtifact) -> pd.DataFrame:
        rows = []
        for b in batches:
            sample_ids = np.asarray(b.meta.get("sample_id"))
            setups = np.asarray(b.meta.get("setup"))
            tickers = np.asarray(b.meta.get("ticker"))
            asof_dates = np.asarray(b.meta.get("asof_date"))
            if sample_ids is None:
                raise ValueError("StandardBatch.meta must include 'sample_id'")
            y_true = np.asarray(b.y).astype(int)

            for sid, yy, ss, tkr, d in zip(
                sample_ids.tolist(),
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
                        "score": 0.0,
                    }
                )
        return pd.DataFrame(rows)


def get_default_registry() -> dict[str, Runner]:
    # Register only concrete runners we actually implement.
    r1: Runner = ConstantPriorRunner()
    r2: Runner = NoOpWarmRunner()
    # Optional ML runner(s): import lazily so base CLI works without torch.
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise ImportError("torch not installed")
        from .ssl.runner_classifier import SSLTCNClassifierRunner
        from .ssl.runner_pseudo_head import TorchSSLHeadStudentRunner
        from .torch_models.reranker_head import TorchRerankerHeadRunner

        r3: Runner = SSLTCNClassifierRunner()
        r5: Runner = TorchSSLHeadStudentRunner()
        r4: Runner = TorchRerankerHeadRunner()
        return {r1.name: r1, r2.name: r2, r3.name: r3, r5.name: r5, r4.name: r4}
    except Exception:
        # If torch (or other ML deps) aren't installed, just omit the runner.
        return {r1.name: r1, r2.name: r2}


def resolve_model_types(model_type: str, registry: dict[str, Runner]) -> list[str]:
    m = (model_type or "").strip().lower()
    if m in {"all", "*"}:
        return sorted(registry.keys())
    if m not in registry:
        raise ValueError(f"Unknown model_type {model_type!r}. Known: {sorted(registry)} or 'all'.")
    return [m]

