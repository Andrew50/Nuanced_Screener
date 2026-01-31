from __future__ import annotations

import pandas as pd


def lf_summary(lf_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize LF behavior.

    Expects `lf_matrix` in wide form: sample_id + LF columns with votes in {-1,0,+1}.
    """
    if lf_matrix.empty:
        return pd.DataFrame([])
    if "sample_id" not in lf_matrix.columns:
        raise ValueError("lf_matrix must include sample_id")

    cols = [c for c in lf_matrix.columns if c != "sample_id"]
    rows = []
    n = int(len(lf_matrix))
    for c in cols:
        s = pd.to_numeric(lf_matrix[c], errors="coerce").fillna(0).astype(int)
        pos = int((s == 1).sum())
        neg = int((s == -1).sum())
        abst = int((s == 0).sum())
        emitted = pos + neg
        rows.append(
            {
                "lf": str(c),
                "n": n,
                "pos": pos,
                "neg": neg,
                "abstain": abst,
                "coverage": (emitted / n) if n else 0.0,
                "pos_rate_given_emit": (pos / emitted) if emitted else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["lf"]).reset_index(drop=True)


def sample_vote_stats(lf_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Per-sample vote totals for debugging and filtering.
    """
    if lf_matrix.empty:
        return pd.DataFrame([])
    if "sample_id" not in lf_matrix.columns:
        raise ValueError("lf_matrix must include sample_id")
    cols = [c for c in lf_matrix.columns if c != "sample_id"]
    if not cols:
        return lf_matrix[["sample_id"]].copy()

    v = lf_matrix[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    pos = (v == 1).sum(axis=1)
    neg = (v == -1).sum(axis=1)
    abst = (v == 0).sum(axis=1)
    conflict = ((pos > 0) & (neg > 0)).astype(int)
    return pd.DataFrame(
        {
            "sample_id": lf_matrix["sample_id"].astype(str),
            "pos_votes": pos.astype(int),
            "neg_votes": neg.astype(int),
            "abstains": abst.astype(int),
            "conflict": conflict.astype(int),
            "emitted": (pos + neg).astype(int),
        }
    )


def lf_signature(lf_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute a stable LF-vote signature string per sample.
    """
    if lf_matrix.empty:
        return pd.Series([], dtype=str)
    if "sample_id" not in lf_matrix.columns:
        raise ValueError("lf_matrix must include sample_id")
    cols = [c for c in lf_matrix.columns if c != "sample_id"]
    cols_sorted = sorted(cols)
    v = lf_matrix[cols_sorted].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    sig = v.astype(str).agg("|".join, axis=1)
    return sig

