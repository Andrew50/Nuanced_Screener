from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from screener_loader.labels import assign_split, load_labels_csv


def test_load_labels_csv_normalizes_and_coerces(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"ticker": "aapl", "date": "2025-03-06", "setup": "flag", "label": "true", "is_gap": "0"},
            {"ticker": "MSFT", "date": "2025-03-07", "setup": "flag", "label": 0, "is_gap": "1"},
        ]
    ).to_csv(p, index=False)

    res = load_labels_csv(p)
    df = res.df
    assert "asof_date" in df.columns
    assert "date" not in df.columns
    assert df["ticker"].tolist() == ["AAPL", "MSFT"]
    assert df["label"].dtype.name in {"bool", "boolean"}
    assert df["is_gap"].dtype.name in {"bool", "boolean"}


def test_load_labels_csv_dedupe_error(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2025-03-06", "setup": "flag", "label": True},
            {"ticker": "AAPL", "date": "2025-03-06", "setup": "flag", "label": False},
        ]
    ).to_csv(p, index=False)

    with pytest.raises(ValueError, match="Duplicate"):
        load_labels_csv(p, dedupe="error")


def test_assign_split_time_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    rows = []
    for d in ["2025-03-06", "2025-03-07", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14"]:
        rows.append({"ticker": "AAPL", "date": d, "setup": "flag", "label": True})
    pd.DataFrame(rows).to_csv(p, index=False)
    df = load_labels_csv(p).df

    out = assign_split(df, mode="time", train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert set(out["split"].unique().tolist()) <= {"train", "val", "test"}
    # Sorted by date, so first rows are train.
    assert out.iloc[0]["split"] == "train"

