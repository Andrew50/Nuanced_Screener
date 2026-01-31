from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import LoaderConfig
from .duckdb_utils import connect
from .embeddings import EmbeddingModel, stable_config_hash
from .paths import ensure_dirs
from .sliding_windows import MaskPolicy, NormalizationConfig, iter_sliding_windows_for_ticker, load_raw_ticker_history


IndexKind = Literal["fullinfo", "openonly"]


def default_feature_columns() -> list[str]:
    # Keep aligned with normalization.returns_relative assumptions.
    return ["open", "high", "low", "close", "volume"]


def _infer_ticker_from_raw(df: pd.DataFrame, fallback: str) -> str:
    if df.empty:
        return str(fallback).upper()
    if "ticker" in df.columns:
        vals = df["ticker"].astype(str).str.upper().unique().tolist()
        if len(vals) == 1:
            return str(vals[0])
    return str(fallback).upper()


def _ticker_bucket(ticker: str, *, bucket_chars: int = 1) -> str:
    t = str(ticker).upper()
    b = max(1, int(bucket_chars))
    return (t[:b] if t else "_")


@dataclass(frozen=True)
class EmbeddingIndexConfig:
    feature_columns: tuple[str, ...]
    window_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    normalization: NormalizationConfig
    mask_policy: MaskPolicy
    encoder: dict[str, Any]
    bucket_chars: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "window_sizes": [int(x) for x in self.window_sizes],
            "strides": [int(x) for x in self.strides],
            "normalization": self.normalization.as_dict(),
            "mask_policy": str(self.mask_policy),
            "encoder": self.encoder,
            "bucket_chars": int(self.bucket_chars),
        }

    def config_hash(self) -> str:
        return stable_config_hash(self.as_dict())


def index_dir_for(cfg: LoaderConfig, kind: IndexKind) -> Path:
    if kind == "fullinfo":
        return cfg.paths.window_embeddings_fullinfo_dir
    if kind == "openonly":
        return cfg.paths.window_embeddings_openonly_dir
    raise ValueError(f"Unknown index kind: {kind}")


def _write_index_meta(out_dir: Path, *, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_index_meta(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "index.meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing index.meta.json under {out_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def assert_index_config_matches(out_dir: Path, *, expected: EmbeddingIndexConfig) -> dict[str, Any]:
    meta = read_index_meta(out_dir)
    want = expected.config_hash()
    got = str(meta.get("config_hash") or "")
    if got != want:
        raise ValueError(
            "Index config mismatch.\n"
            f"- index_dir={out_dir}\n"
            f"- expected_hash={want}\n"
            f"- found_hash={got}\n"
            "This usually means normalization/mask_policy/features/encoder changed between build and use."
        )
    return meta


def _fixed_size_list_f16(values_2d: np.ndarray) -> pa.FixedSizeListArray:
    x = np.asarray(values_2d)
    if x.ndim != 2:
        raise ValueError("values_2d must be 2D")
    n, d = x.shape
    flat = pa.array(x.astype(np.float16, copy=False).reshape(n * d), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=int(d))


def _ensure_clean_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def build_embedding_index(
    cfg: LoaderConfig,
    *,
    out_dir: Path,
    encoder: EmbeddingModel,
    feature_columns: list[str] | None = None,
    window_sizes: Iterable[int] = (32, 48, 64, 96),
    strides: Iterable[int] = (2, 3, 4),
    normalization: NormalizationConfig = NormalizationConfig(),
    mask_policy: MaskPolicy = "none",
    tickers: list[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
    bucket_chars: int = 1,
    batch_size: int = 2048,
    overwrite_meta: bool = False,
) -> Path:
    """
    Build an on-disk embedding index from per-ticker raw history.

    Output is a Hive-partitioned Parquet dataset under out_dir with columns:
    - window_id, ticker, start_date, end_date, end_year_month
    - window_size, stride, start_idx, end_idx
    - embedding (fixed_size_list<float16>[D])
    """
    ensure_dirs(cfg.paths)
    _ensure_clean_dir(out_dir)

    feats = feature_columns or default_feature_columns()
    ws = tuple(int(x) for x in window_sizes)
    ss = tuple(int(x) for x in strides)
    if not ws or not ss:
        raise ValueError("window_sizes and strides must be non-empty")

    enc_cfg = encoder.config()
    cfg_obj = EmbeddingIndexConfig(
        feature_columns=tuple(feats),
        window_sizes=ws,
        strides=ss,
        normalization=normalization,
        mask_policy=mask_policy,
        encoder=enc_cfg,
        bucket_chars=int(bucket_chars),
    )
    cfg_hash = cfg_obj.config_hash()

    meta_path = out_dir / "index.meta.json"
    if meta_path.exists() and not overwrite_meta:
        # Enforce: if meta exists, it must match the requested config.
        assert_index_config_matches(out_dir, expected=cfg_obj)
    else:
        _write_index_meta(
            out_dir,
            meta={
                "config_hash": cfg_hash,
                "config": cfg_obj.as_dict(),
                "note": "Embeddings are L2-normalized; cosine similarity reduces to dot product.",
            },
        )

    raw_dir = cfg.paths.raw_dir
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if tickers is not None:
        want = set(str(t).upper() for t in tickers)
        raw_files = [p for p in raw_files if p.stem.upper() in want]

    total_windows = 0
    embed_dim: int | None = None

    for raw_path in raw_files:
        # Typical layout is one ticker per file; use filename as fallback.
        fallback_ticker = raw_path.stem.upper()
        df = load_raw_ticker_history(str(raw_path), date_start=date_start, date_end=date_end)
        if df.empty:
            continue
        ticker = _infer_ticker_from_raw(df, fallback=fallback_ticker)

        for L in ws:
            for stride in ss:
                for meta_df, x_seq, mask_seq in iter_sliding_windows_for_ticker(
                    df,
                    ticker=ticker,
                    feature_columns=feats,
                    window_size=int(L),
                    stride=int(stride),
                    mask_policy=mask_policy,
                    normalization=normalization,
                    batch_size=int(batch_size),
                ):
                    emb = encoder.encode(x_seq, mask_seq)
                    if emb.ndim != 2 or emb.shape[0] != x_seq.shape[0]:
                        raise ValueError("Encoder returned invalid embedding shape")
                    if embed_dim is None:
                        embed_dim = int(emb.shape[1])
                    elif int(emb.shape[1]) != int(embed_dim):
                        raise ValueError(f"Embedding dim changed within one index build: {embed_dim} -> {emb.shape[1]}")

                    # Partition columns
                    tb = _ticker_bucket(ticker, bucket_chars=bucket_chars)
                    # We write one file per (ticker, L, stride, batch), partitioned by end_year_month too.
                    # Split by end_year_month to allow pruning.
                    for end_ym, sub in meta_df.groupby("end_year_month", sort=False):
                        idx = sub.index.to_numpy()
                        sub_emb = emb[idx]
                        # Arrow table
                        table = pa.table(
                            {
                                "window_id": pa.array(sub["window_id"].astype(str).tolist(), type=pa.string()),
                                "ticker": pa.array(sub["ticker"].astype(str).tolist(), type=pa.string()),
                                "start_date": pa.array(sub["start_date"].tolist(), type=pa.date32()),
                                "end_date": pa.array(sub["end_date"].tolist(), type=pa.date32()),
                                "end_year_month": pa.array(sub["end_year_month"].astype(str).tolist(), type=pa.string()),
                                "window_size": pa.array(sub["window_size"].astype(int).tolist(), type=pa.int16()),
                                "stride": pa.array(sub["stride"].astype(int).tolist(), type=pa.int8()),
                                "start_idx": pa.array(sub["start_idx"].astype(int).tolist(), type=pa.int32()),
                                "end_idx": pa.array(sub["end_idx"].astype(int).tolist(), type=pa.int32()),
                                "ticker_bucket": pa.array([tb] * len(sub), type=pa.string()),
                                "embedding": _fixed_size_list_f16(sub_emb),
                            }
                        )

                        part_dir = out_dir / f"window_size={int(L)}" / f"ticker_bucket={tb}" / f"end_year_month={str(end_ym)}"
                        part_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{ticker}_L{int(L)}_s{int(stride)}_{total_windows:012d}.parquet"
                        pq.write_table(table, part_dir / fname, compression="zstd")

                    total_windows += int(len(meta_df))

    # Update meta with derived totals (idempotent-ish).
    meta = read_index_meta(out_dir)
    meta["built"] = {
        "windows": int(total_windows),
        "embedding_dim": int(embed_dim or 0),
    }
    _write_index_meta(out_dir, meta=meta)
    return out_dir


def open_index_dataset(out_dir: Path) -> ds.Dataset:
    """
    Open the embedding index as a PyArrow dataset (Hive partitioning).
    """
    return ds.dataset(out_dir, format="parquet", partitioning="hive")


def fetch_embeddings_by_examples(
    cfg: LoaderConfig,
    *,
    out_dir: Path,
    examples: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve examples rows (ticker,end_date,window_size,stride,start_idx) to embeddings.

    Returns a DataFrame with the original columns plus:
    - window_id
    - embedding (object: numpy float32 vector)
    """
    required = {"ticker", "end_date", "window_size", "stride", "start_idx"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"examples missing required columns: {sorted(missing)}")
    if examples.empty:
        return examples.copy()

    ex = examples.copy()
    ex["ticker"] = ex["ticker"].astype(str).str.upper()
    ex["end_date"] = pd.to_datetime(ex["end_date"]).dt.date
    ex["window_size"] = ex["window_size"].astype(int)
    ex["stride"] = ex["stride"].astype(int)
    ex["start_idx"] = ex["start_idx"].astype(int)

    # DuckDB filter is the simplest for tiny example lists.
    con = connect(cfg)
    con.register("examples", ex)
    glob = (out_dir / "**" / "*.parquet").as_posix()
    df = con.execute(
        f"""
        SELECT
          e.*,
          idx.window_id AS window_id,
          idx.embedding AS embedding
        FROM examples AS e
        JOIN read_parquet('{glob}') AS idx
          ON idx.ticker = e.ticker
         AND idx.end_date = e.end_date
         AND idx.window_size = e.window_size
         AND idx.stride = e.stride
         AND idx.start_idx = e.start_idx
        """,
    ).df()

    if df.empty:
        raise ValueError("No embeddings found for provided examples (did you build the index with matching L/stride/date range?)")

    # Convert embedding lists to numpy vectors.
    def _to_vec(v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v.astype(np.float32, copy=False)
        if isinstance(v, list):
            return np.asarray(v, dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    df["embedding"] = df["embedding"].map(_to_vec)
    return df

