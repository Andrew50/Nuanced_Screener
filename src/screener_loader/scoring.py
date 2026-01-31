from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreReport:
    metrics: dict
    metrics_by_setup: pd.DataFrame


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank scores with average ranks for ties (Mann-Whitney U).
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # Average ranks for ties.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # Step-function integration (average precision).
    # Ensure starts at recall=0.
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def _binary_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    if k <= 0:
        return {"k": int(k), "precision_at_k": None, "recall_at_k": None}
    k = min(int(k), len(y_true))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    tp = int((y_true[order] == 1).sum())
    prec = tp / k if k else None
    rec = tp / n_pos if n_pos else None
    return {"k": int(k), "precision_at_k": prec, "recall_at_k": rec, "tp_at_k": tp, "pos_total": n_pos}


def score_binary_predictions(
    preds: pd.DataFrame,
    *,
    threshold: float = 0.5,
    k_values: Iterable[int] = (25, 50, 100),
) -> ScoreReport:
    """
    Score binary classification predictions.

    Requires columns: y_true (0/1), score (float)
    Optional: setup (string)
    """
    if preds.empty:
        raise ValueError("predictions are empty")
    required = {"y_true", "score"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    df = preds.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("all prediction scores are NaN")

    def _metrics_for(sub: pd.DataFrame) -> dict:
        y = sub["y_true"].to_numpy()
        s = sub["score"].to_numpy()
        out = {
            "n": int(len(sub)),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
            "auroc": _auroc(y, s),
            "auprc": _auprc(y, s),
            "at_threshold": _binary_at_threshold(y, s, threshold),
            "at_k": [_precision_recall_at_k(y, s, int(k)) for k in k_values],
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s)),
        }
        return out

    overall = _metrics_for(df)

    if "setup" in df.columns:
        by = []
        for setup, sub in df.groupby("setup", sort=True):
            m = _metrics_for(sub)
            m["setup"] = str(setup)
            by.append(m)
        metrics_by_setup = pd.DataFrame(by).sort_values(["setup"]).reset_index(drop=True)
    else:
        metrics_by_setup = pd.DataFrame([])

    report = {
        "task_type": "binary_classification",
        "overall": overall,
    }
    return ScoreReport(metrics=report, metrics_by_setup=metrics_by_setup)


def write_score_artifacts(
    *,
    run_dir: Path,
    predictions: pd.DataFrame,
    report: ScoreReport,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "metrics.json"
    by_setup_path = run_dir / "metrics_by_setup.parquet"

    predictions.to_parquet(preds_path, index=False)
    metrics_path.write_text(json.dumps(report.metrics, indent=2, default=str) + "\n", encoding="utf-8")
    if not report.metrics_by_setup.empty:
        report.metrics_by_setup.to_parquet(by_setup_path, index=False)

    return {"predictions": preds_path, "metrics": metrics_path, "metrics_by_setup": by_setup_path}

