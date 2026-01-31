from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .calendar_utils import TradingCalendar, latest_trading_day_on_or_before
from .paths import DataPaths, atomic_replace


ResolveNonTrading = Literal["error", "previous", "next"]
DedupePolicy = Literal["error", "keep_first", "keep_last"]
SplitMode = Literal["time", "random", "column"]


@dataclass(frozen=True)
class LabelsLoadResult:
    df: pd.DataFrame
    # Fields normalized into the store:
    key_columns: tuple[str, ...] = ("ticker", "asof_date", "setup")


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.astype("Int64").map({1: True, 0: False})

    # Strings and mixed types.
    def _one(v):  # noqa: ANN001
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            if int(v) == 1:
                return True
            if int(v) == 0:
                return False
        txt = str(v).strip().lower()
        if txt in {"true", "t", "yes", "y", "1"}:
            return True
        if txt in {"false", "f", "no", "n", "0"}:
            return False
        return pd.NA

    out = s.map(_one)
    return out.astype("boolean")


def _parse_date_series(s: pd.Series) -> pd.Series:
    # Accept date/datetime/str; normalize to Python date.
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.to_datetime(s).dt.date
    # For object/string, use pandas parsing.
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.date


def _resolve_to_trading_day(d: date, cal: TradingCalendar, mode: ResolveNonTrading) -> date:
    if mode == "error":
        return d
    if mode == "previous":
        return latest_trading_day_on_or_before(d, cal)
    if mode == "next":
        # Find the earliest trading day >= d via a small lookahead.
        days = cal.valid_trading_days(d, d + timedelta(days=10))
        if not days:
            return d
        return days[0]
    raise ValueError(f"Unknown resolve_non_trading={mode!r}")


def _infer_boolean_attribute_columns(df: pd.DataFrame, ignore: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in ignore:
            continue
        # Quick accept if already boolean dtype.
        if str(df[c].dtype) in {"bool", "boolean"}:
            cols.append(c)
            continue
        # Try coercion on a small sample; if mostly parseable, accept.
        sample = df[c].dropna().head(50)
        if sample.empty:
            continue
        coerced = _coerce_bool_series(sample)
        if coerced.isna().sum() == 0:
            cols.append(c)
    return cols


def load_labels_csv(
    csv_path: Path,
    *,
    cal: TradingCalendar | None = None,
    resolve_non_trading: ResolveNonTrading = "error",
    dedupe: DedupePolicy = "error",
) -> LabelsLoadResult:
    """
    Load and normalize a long-form labels CSV.

    Required input columns:
    - ticker
    - date (YYYY-MM-DD)
    - setup
    - label (boolean-ish)

    Output columns (normalized):
    - ticker (upper)
    - asof_date (datetime.date)
    - setup (string)
    - label (boolean)
    - plus any optional columns preserved
    """
    cal = cal or TradingCalendar("NYSE")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"labels CSV is empty: {csv_path}")

    required = {"ticker", "date", "setup", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing required columns: {sorted(missing)} (in {csv_path})")

    out = df.copy()

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["setup"] = out["setup"].astype(str).str.strip()
    out["label"] = _coerce_bool_series(out["label"])

    asof = _parse_date_series(out["date"])
    if asof.isna().any():
        bad = out.loc[asof.isna(), "date"].head(10).tolist()
        raise ValueError(f"Could not parse some dates in labels CSV; examples: {bad}")
    out = out.drop(columns=["date"])
    out["asof_date"] = asof

    # Basic integrity.
    if (out["ticker"].str.len() == 0).any():
        raise ValueError("labels CSV contains empty ticker values after stripping")
    if (out["setup"].str.len() == 0).any():
        raise ValueError("labels CSV contains empty setup values after stripping")
    if out["label"].isna().any():
        raise ValueError("labels CSV contains unparsable/empty label values")

    # Resolve trading days if requested, otherwise validate.
    if resolve_non_trading != "error":
        out["asof_date"] = out["asof_date"].map(lambda d: _resolve_to_trading_day(d, cal, resolve_non_trading))

    # Validate that asof_date is a trading day (after any resolution).
    unique_dates = sorted(set(out["asof_date"].tolist()))
    if unique_dates:
        min_d, max_d = unique_dates[0], unique_dates[-1]
        valid = set(cal.valid_trading_days(min_d - timedelta(days=10), max_d + timedelta(days=10)))
        bad = [d for d in unique_dates if d not in valid]
        if bad:
            raise ValueError(
                "Some label asof_date values are not trading days. "
                f"First few: {bad[:10]}. Consider resolve_non_trading='previous' or 'next'."
            )

    # Coerce optional well-known columns.
    if "weight" in out.columns:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if "fold" in out.columns:
        out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype("Int64")
    if "split" in out.columns:
        out["split"] = out["split"].astype(str).str.strip().str.lower()

    # Infer and coerce additional boolean attribute columns.
    ignore = {"ticker", "asof_date", "setup", "label", "weight", "fold", "split", "source", "notes", "created_ts"}
    bool_cols = _infer_boolean_attribute_columns(out, ignore=ignore)
    for c in bool_cols:
        out[c] = _coerce_bool_series(out[c])

    # Dedupe.
    key_cols = ["ticker", "asof_date", "setup"]
    dup_mask = out.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        if dedupe == "error":
            dups = out.loc[dup_mask, key_cols].head(10).to_dict(orient="records")
            raise ValueError(f"Duplicate (ticker, asof_date, setup) rows in labels CSV; examples: {dups}")
        keep = "first" if dedupe == "keep_first" else "last"
        out = out.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)

    # Stable ordering (helps reproducibility).
    out = out.sort_values(key_cols).reset_index(drop=True)

    return LabelsLoadResult(df=out)


def write_labels_store(
    result: LabelsLoadResult,
    *,
    paths: DataPaths,
    source_csv: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or paths.labels_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(out_path) + ".tmp")

    result.df.to_parquet(tmp_path, index=False)
    atomic_replace(tmp_path, out_path)

    # Small sidecar metadata (safe to ignore if missing).
    meta = {
        "source_csv": str(source_csv) if source_csv is not None else None,
        "rows": int(len(result.df)),
        "columns": list(result.df.columns),
        "key_columns": list(result.key_columns),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return out_path


def load_labels_store(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df
    # Ensure expected columns exist.
    required = {"ticker", "asof_date", "setup", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels store missing required columns: {sorted(missing)} (in {parquet_path})")
    return df


def list_setups(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return sorted(set(df["setup"].astype(str).tolist()))


def filter_labels(
    df: pd.DataFrame,
    *,
    setup: str | None = None,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = df
    if setup and setup.lower() != "all":
        out = out[out["setup"].astype(str) == str(setup)]
    if tickers is not None:
        s = set(str(t).upper() for t in tickers)
        out = out[out["ticker"].astype(str).str.upper().isin(s)]
    return out.reset_index(drop=True)


def assign_split(
    df: pd.DataFrame,
    *,
    mode: SplitMode = "time",
    split_column: str = "split",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
) -> pd.DataFrame:
    """
    Assign a split column (`train`/`val`/`test`) in a leakage-safe way.

    - time: split by asof_date ordering (preferred for market data)
    - random: IID split (useful for quick debugging)
    - column: use an existing column, validated and normalized
    """
    if df.empty:
        return df.copy()
    out = df.copy()

    if mode == "column":
        if split_column not in out.columns:
            raise ValueError(f"split_column {split_column!r} not found in labels")
        s = out[split_column].astype(str).str.strip().str.lower()
        ok = s.isin({"train", "val", "test"})
        if not ok.all():
            bad = sorted(set(s[~ok].head(10).tolist()))
            raise ValueError(f"Invalid split values in column {split_column!r}; examples: {bad}. Use train/val/test.")
        out["split"] = s
        return out

    train_frac = float(train_frac)
    val_frac = float(val_frac)
    test_frac = float(test_frac)
    total = train_frac + val_frac + test_frac
    if not (0.99 <= total <= 1.01):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    n = len(out)
    if mode == "random":
        rng = pd.Series(range(n)).sample(frac=1.0, random_state=int(seed)).to_numpy()
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        split = pd.Series(["test"] * n, index=out.index)
        split.iloc[rng[:n_train]] = "train"
        split.iloc[rng[n_train : n_train + n_val]] = "val"
        out["split"] = split.to_numpy()
        return out

    if mode == "time":
        if "asof_date" not in out.columns:
            raise ValueError("labels must contain asof_date for time split")
        out = out.sort_values(["asof_date", "ticker", "setup"]).reset_index(drop=True)
        # Split by index after sorting; simple and stable.
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        split = np.array(["test"] * n, dtype=object)
        split[:n_train] = "train"
        split[n_train : n_train + n_val] = "val"
        out["split"] = split
        return out

    raise ValueError(f"Unknown split mode: {mode}")

