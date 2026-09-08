"""Bulk last-N candidate windows and small-catalog example windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ...config import LoaderConfig
from ...duckdb_utils import connect
from ...setups.charts import load_ohlcv_window
from ...setups.spec import SetupSpec, SetupValidationError, VisionExample
from ...setups import store as setup_store
from ..snapshots import bars_from_rows, make_bar_window, snapshot_example
from ..types import (
    BarWindow,
    ExampleInput,
    InputSkip,
    InsufficientLastNError,
    VisionError,
)
from .freshness import coerce_date

_OHLC = ("open", "high", "low", "close")
_IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass(frozen=True)
class LastNCapacity:
    max_rn: int
    ticker_count: int
    row_count: int
    max_date: date | None
    min_date: date | None
    latest_by_ticker: dict[str, date]


def rebuild_last100_hint(needed_bars: int, repo_root: Path) -> str:
    return (
        f"ns rebuild-last100 --window-size {int(needed_bars)} --repo-root {Path(repo_root).resolve()}"
    )


def raise_insufficient_last_n(needed_bars: int, repo_root: Path) -> None:
    needed = int(needed_bars)
    err = InsufficientLastNError(needed)
    err.rebuild_hint = rebuild_last100_hint(needed, repo_root)  # type: ignore[attr-defined]
    raise err


def inspect_last_n_capacity(config: LoaderConfig) -> LastNCapacity:
    src = config.paths.last_100_bars_parquet
    if not src.exists():
        raise FileNotFoundError(
            f"Derived last-N bars not found: {src}. Run `{rebuild_last100_hint(100, config.repo_root)}` first."
        )
    con = connect(config)
    summary = con.execute(
        """
        SELECT
          CAST(max(rn) AS BIGINT) AS max_rn,
          CAST(count(DISTINCT UPPER(CAST(ticker AS VARCHAR))) AS BIGINT) AS ticker_count,
          CAST(count(*) AS BIGINT) AS row_count,
          CAST(max(date) AS DATE) AS max_date,
          CAST(min(date) AS DATE) AS min_date
        FROM read_parquet(?)
        """,
        [str(src)],
    ).fetchone()
    max_rn = int(summary[0] or 0)
    ticker_count = int(summary[1] or 0)
    row_count = int(summary[2] or 0)
    max_date = coerce_date(summary[3])
    min_date = coerce_date(summary[4])
    latest_df = con.execute(
        """
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS date
        FROM read_parquet(?)
        WHERE rn = 1
        """,
        [str(src)],
    ).df()
    latest: dict[str, date] = {}
    for _, row in latest_df.iterrows():
        d = coerce_date(row["date"])
        if d is None:
            continue
        latest[str(row["ticker"])] = d
    return LastNCapacity(
        max_rn=max_rn,
        ticker_count=ticker_count,
        row_count=row_count,
        max_date=max_date,
        min_date=min_date,
        latest_by_ticker=latest,
    )


def load_candidate_windows(
    config: LoaderConfig,
    *,
    tickers: Sequence[str],
    lookback_bars: int,
    expected_session: date,
    feature_asof: Mapping[str, date],
    volume_required: bool,
) -> tuple[dict[str, BarWindow], tuple[InputSkip, ...]]:
    """Bulk-load eligible windows from last_100_bars.parquet. Never per-ticker load_ohlcv_window."""

    wanted = [str(t).strip().upper() for t in tickers]
    if not wanted:
        return {}, ()
    src = config.paths.last_100_bars_parquet
    con = connect(config)
    sample = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(src)]).df()
    have_source = "source" in set(sample.columns.astype(str))
    source_sql = "CAST(source AS VARCHAR) AS source" if have_source else "CAST(NULL AS VARCHAR) AS source"
    df = con.execute(
        f"""
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS date,
          CAST(open AS DOUBLE) AS open,
          CAST(high AS DOUBLE) AS high,
          CAST(low AS DOUBLE) AS low,
          CAST(close AS DOUBLE) AS close,
          CAST(volume AS DOUBLE) AS volume,
          {source_sql},
          CAST(rn AS INTEGER) AS rn
        FROM read_parquet(?)
        WHERE rn <= ?
        ORDER BY ticker, date
        """,
        [str(src), int(lookback_bars)],
    ).df()
    wanted_set = set(wanted)
    if not df.empty:
        df = df[df["ticker"].astype(str).isin(wanted_set)].copy()
    skips: list[InputSkip] = []
    windows: dict[str, BarWindow] = {}
    grouped = df.groupby(df["ticker"].astype(str), sort=True) if not df.empty else []
    seen: set[str] = set()
    for ticker, group in grouped:
        ticker_s = str(ticker)
        seen.add(ticker_s)
        skip = _window_or_skip(
            ticker_s,
            group,
            lookback_bars=int(lookback_bars),
            expected_session=expected_session,
            feature_asof=feature_asof.get(ticker_s),
            volume_required=volume_required,
        )
        if isinstance(skip, InputSkip):
            skips.append(skip)
        else:
            windows[ticker_s] = skip
    for ticker_s in wanted:
        if ticker_s in seen:
            continue
        skips.append(
            InputSkip(
                ticker=ticker_s,
                kind="short_window",
                message=f"{ticker_s} has no last-N rows to chart",
                asof_date=feature_asof.get(ticker_s),
                bar_count=0,
            )
        )
    return windows, tuple(skips)


def _window_or_skip(
    ticker: str,
    group: pd.DataFrame,
    *,
    lookback_bars: int,
    expected_session: date,
    feature_asof: date | None,
    volume_required: bool,
) -> BarWindow | InputSkip:
    work = group.sort_values("date").reset_index(drop=True)
    n = len(work)
    last = coerce_date(work.iloc[-1]["date"]) if n else None
    if n < int(lookback_bars):
        return InputSkip(
            ticker=ticker,
            kind="short_window",
            message=(
                f"{ticker} has {n} last-N bars; lookback_bars={lookback_bars}. "
                "Isolated short history is skipped rather than silently shortened."
            ),
            asof_date=last,
            bar_count=n,
        )
    if last is None:
        return InputSkip(
            ticker=ticker,
            kind="unavailable_input",
            message=f"{ticker} window is missing a final session date",
            bar_count=n,
        )
    if last < expected_session:
        return InputSkip(
            ticker=ticker,
            kind="stale",
            message=(
                f"{ticker} last session {last.isoformat()} is behind expected "
                f"{expected_session.isoformat()}"
            ),
            asof_date=last,
            bar_count=n,
        )
    if last > expected_session:
        return InputSkip(
            ticker=ticker,
            kind="unavailable_input",
            message=(
                f"{ticker} last session {last.isoformat()} is after expected "
                f"{expected_session.isoformat()}"
            ),
            asof_date=last,
            bar_count=n,
        )
    if feature_asof is not None and feature_asof != last:
        return InputSkip(
            ticker=ticker,
            kind="unavailable_input",
            message=(
                f"{ticker} feature asof {feature_asof.isoformat()} does not match "
                f"window session {last.isoformat()}; refusing to mix old features with newer bars"
            ),
            asof_date=last,
            bar_count=n,
        )
    try:
        rows = _validated_bar_rows(work, volume_required=volume_required)
    except VisionError as exc:
        return InputSkip(
            ticker=ticker,
            kind="unavailable_input",
            message=str(exc),
            asof_date=last,
            bar_count=n,
        )
    provenance = _honest_provenance(work)
    bars = bars_from_rows(rows)
    return make_bar_window(ticker, bars, timeframe="1d", provenance=provenance)


def _validated_bar_rows(work: pd.DataFrame, *, volume_required: bool) -> list[dict[str, Any]]:
    dates: list[date] = []
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        d = coerce_date(row["date"])
        if d is None:
            raise VisionError("bar row is missing date")
        if dates and d <= dates[-1]:
            raise VisionError(f"bar dates are not strictly increasing near {d.isoformat()}")
        dates.append(d)
        payload: dict[str, Any] = {"date": d}
        for col in _OHLC:
            payload[col] = row.get(col)
        payload["volume"] = row.get("volume")
        rows.append(payload)
    if volume_required:
        vols = pd.to_numeric(work["volume"], errors="coerce")
        if vols.isna().any():
            raise VisionError("volume is required by the chart profile but missing")
    return rows


def _honest_provenance(work: pd.DataFrame) -> str:
    if "source" not in work.columns:
        return "unknown"
    values = [str(v).strip() for v in work["source"].tolist() if v is not None and str(v).strip() not in {"", "nan", "None"}]
    unique = set(values)
    if len(unique) == 1:
        return next(iter(unique))
    return "unknown"


def load_example_input(
    config: LoaderConfig,
    spec: SetupSpec,
    example: VisionExample,
    *,
    lookback_bars: int,
) -> ExampleInput:
    if example.type == "image":
        data, _media = read_upload_bytes(config, spec, example)
        return snapshot_example(spec, example, image_bytes=data)
    if example.date is None or not example.ticker:
        raise VisionError(f"market_window example {example.id!r} on {spec.id} is missing ticker/date")
    try:
        frame = load_ohlcv_window(config, example.ticker, example.date, int(lookback_bars))
    except FileNotFoundError as exc:
        raise VisionError(
            f"market_window example {example.id!r} on {spec.id} is missing history ending "
            f"{example.date.isoformat()} (no download/backfill will be attempted): {exc}"
        ) from exc
    if len(frame) != int(lookback_bars):
        raise VisionError(
            f"market_window example {example.id!r} on {spec.id} has {len(frame)} bars; "
            f"need exactly {int(lookback_bars)}. tail(N) returned less than N; will not backfill."
        )
    actual = coerce_date(frame.iloc[-1]["date"])
    if actual != example.date:
        raise VisionError(
            f"market_window example {example.id!r} on {spec.id} requested {example.date.isoformat()} "
            f"but the window ends on {actual.isoformat() if actual else 'none'}"
        )
    rows = _validated_bar_rows(frame, volume_required=bool(spec.chart.volume))
    window = make_bar_window(
        str(example.ticker),
        bars_from_rows(rows),
        timeframe=example.timeframe,
        provenance=_honest_provenance(frame),
    )
    return snapshot_example(spec, example, window=window)


def read_upload_bytes(
    config: LoaderConfig,
    spec: SetupSpec,
    example: VisionExample,
) -> tuple[bytes, str]:
    if example.type != "image" or not example.path:
        raise VisionError(f"image example {example.id!r} on {spec.id} is missing path")
    setup_dir = setup_store.setup_dir(config.paths, spec.id).resolve()
    dest = (setup_dir / str(example.path)).resolve()
    if setup_dir not in dest.parents and dest != setup_dir:
        raise VisionError(f"image example {example.id!r} path escapes the setup directory")
    if not dest.is_file():
        raise VisionError(f"image example {example.id!r} on {spec.id} is missing file {dest}")
    suffix = dest.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        raise SetupValidationError(f"Unsupported image type {suffix}. Use png/jpg/webp.")
    data = dest.read_bytes()
    if not data:
        raise VisionError(f"image example {example.id!r} on {spec.id} is empty")
    media = _sniff_image(data)
    expected = _IMAGE_SUFFIXES[suffix]
    if media != "application/octet-stream" and media != expected:
        raise VisionError(
            f"image example {example.id!r} on {spec.id} bytes do not match {suffix} ({media})"
        )
    return data, expected if media == "application/octet-stream" else media


def _sniff_image(data: bytes) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
