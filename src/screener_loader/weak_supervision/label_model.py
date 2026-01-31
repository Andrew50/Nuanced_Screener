from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..progress import progress_ctx


@dataclass(frozen=True)
class LabelModelParams:
    class_prior: float
    coverage: dict[str, float]
    accuracy: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "class_prior": float(self.class_prior),
                "coverage": {str(k): float(v) for k, v in self.coverage.items()},
                "accuracy": {str(k): float(v) for k, v in self.accuracy.items()},
            },
            indent=2,
            sort_keys=True,
        )


def _vote_cols(lf_matrix: pd.DataFrame) -> list[str]:
    if "sample_id" not in lf_matrix.columns:
        raise ValueError("lf_matrix must include sample_id")
    return [c for c in lf_matrix.columns if c != "sample_id"]


def majority_vote(lf_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline combiner: pos/(pos+neg) with abstains ignored.
    """
    if lf_matrix.empty:
        return pd.DataFrame([])
    cols = _vote_cols(lf_matrix)
    if not cols:
        raise ValueError("lf_matrix has no LF columns")

    v = lf_matrix[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    pos = (v == 1).sum(axis=1).astype(int)
    neg = (v == -1).sum(axis=1).astype(int)
    emitted = (pos + neg).astype(int)

    denom = emitted.replace({0: np.nan}).astype(float)
    p = (pos.astype(float) / denom).fillna(0.5).clip(0.0, 1.0)
    # Simple confidence: agreement fraction among emitted votes (0.5 at full conflict).
    agreement = (np.maximum(pos, neg).astype(float) / denom).fillna(0.0)
    weight = agreement.fillna(0.0)

    return pd.DataFrame(
        {
            "sample_id": lf_matrix["sample_id"].astype(str),
            "p_label": p.astype(float),
            "label_hard": (p >= 0.5),
            "weight": weight.astype(float),
            "pos_votes": pos,
            "neg_votes": neg,
            "emitted": emitted,
        }
    )


def fit_independent_label_model(
    lf_matrix: pd.DataFrame,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    init_class_prior: float = 0.2,
    min_accuracy: float = 0.55,
    max_accuracy: float = 0.95,
) -> tuple[pd.DataFrame, LabelModelParams]:
    """
    Snorkel-style generative label model with:
    - latent y in {0,1}
    - independent LFs conditioned on y
    - symmetric LF accuracies for +1/-1
    - LF coverage independent of y (estimated empirically)

    Returns:
    - posteriors: sample_id, p_label, label_hard, weight, emitted
    - params: learned pi and per-LF accuracies/coverage
    """
    if lf_matrix.empty:
        raise ValueError("lf_matrix is empty")

    cols = _vote_cols(lf_matrix)
    if not cols:
        raise ValueError("lf_matrix has no LF columns")

    V = lf_matrix[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).to_numpy()
    if not np.isin(V, [-1, 0, 1]).all():
        raise ValueError("lf_matrix must contain only -1/0/+1 votes")

    n, m = V.shape
    emitted = (V != 0).sum(axis=1).astype(int)

    # Empirical coverage per LF.
    cov = (V != 0).mean(axis=0)
    cov_map = {cols[j]: float(cov[j]) for j in range(m)}

    # Initialize.
    pi = float(np.clip(init_class_prior, 1e-3, 1 - 1e-3))
    acc = np.full(m, 0.7, dtype=float)
    acc = np.clip(acc, float(min_accuracy), float(max_accuracy))

    def _log_prob_y(v_row: np.ndarray, y: int) -> float:
        # y=1 corresponds to +1 being "correct"; y=0 corresponds to -1 being "correct".
        lp = np.log(pi if y == 1 else (1 - pi))
        for j in range(m):
            l = int(v_row[j])
            c = cov[j]
            a = acc[j]
            if l == 0:
                lp += np.log(max(1e-12, 1.0 - c))
            elif l == 1:
                lp += np.log(max(1e-12, c * (a if y == 1 else (1.0 - a))))
            elif l == -1:
                lp += np.log(max(1e-12, c * (a if y == 0 else (1.0 - a))))
        return float(lp)

    prev_ll = None
    p = np.full(n, pi, dtype=float)
    with progress_ctx(transient=False) as progress:
        it_task = progress.add_task("Label model EM iters", total=int(max_iter))

        for it in range(int(max_iter)):
        # E-step: compute p_i = P(y=1|v_i)
            ll = 0.0
            for i in range(n):
                lp1 = _log_prob_y(V[i], 1)
                lp0 = _log_prob_y(V[i], 0)
                # log-sum-exp
                mx = max(lp0, lp1)
                denom = mx + np.log(np.exp(lp0 - mx) + np.exp(lp1 - mx))
                ll += denom
                p[i] = float(np.exp(lp1 - denom))

            # Convergence check (log-likelihood).
            if prev_ll is not None and abs(ll - prev_ll) < float(tol):
                progress.update(it_task, completed=it + 1, description=f"Label model EM iters (converged, ll={ll:.2f})")
                break
            prev_ll = ll

            # M-step:
            pi = float(np.clip(p.mean(), 1e-3, 1 - 1e-3))
            # Accuracy: expected correctness among emitted.
            for j in range(m):
                emit = V[:, j] != 0
                if not np.any(emit):
                    acc[j] = 0.5
                    continue
                vj = V[emit, j]
                pj = p[emit]
                correct = (pj * (vj == 1).astype(float)) + ((1.0 - pj) * (vj == -1).astype(float))
                a = float(correct.sum() / max(1.0, float(len(vj))))
                acc[j] = float(np.clip(a, float(min_accuracy), float(max_accuracy)))

            if (it + 1) % 5 == 0:
                progress.update(it_task, advance=1, description=f"Label model EM iters (ll={ll:.2f})")
            else:
                progress.update(it_task, advance=1)

    acc_map = {cols[j]: float(acc[j]) for j in range(m)}

    # Weight: downweight samples with no LF emissions; otherwise use max(p,1-p) as confidence.
    conf = np.maximum(p, 1.0 - p)
    w = np.where(emitted > 0, conf, 0.0).astype(float)

    post = pd.DataFrame(
        {
            "sample_id": lf_matrix["sample_id"].astype(str),
            "p_label": p.astype(float),
            "label_hard": (p >= 0.5),
            "weight": w.astype(float),
            "emitted": emitted.astype(int),
        }
    )
    params = LabelModelParams(class_prior=float(pi), coverage=cov_map, accuracy=acc_map)
    return post, params

