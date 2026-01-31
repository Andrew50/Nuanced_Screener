from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import typer
from rich import print

from .calendar_utils import TradingCalendar, latest_trading_day_on_or_before, subtract_years
from .config import LoaderConfig
from .duckdb_utils import connect
from .derived import rebuild_last_n_bars_from_polygon_date_partitions
from .labels import assign_split, filter_labels, load_labels_csv, write_labels_store
from .model_registry import get_default_registry, resolve_model_types
from .normalization import build_standard_batch_from_windowed_long, fit_global_zscore_stats, normalize_batch
from .paths import ensure_dirs
from .scoring import score_binary_predictions, write_score_artifacts
from .screening import run_named_query
from .universe import build_universe, load_universe
from .update import update_market_data
from .windowed_dataset import WindowedBuildSpec, build_windowed_bars, stable_sample_id

app = typer.Typer(add_completion=False, no_args_is_help=True)
models_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(models_app, name="models")


def _parse_ymd(value: str | None, flag_name: str) -> date | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(f"{flag_name} must be YYYY-MM-DD; got {value!r}") from e


def _config(
    repo_root: Path = Path("."),
    ticker_source: str = "nasdaq_trader",
    ohlcv_vendor: str = "polygon_grouped",
    exclude_test_issues: bool = True,
    exclude_etfs: bool = False,
    include_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    full_refresh: bool = False,
    interval: str = "1d",
    batch_size: int = 50,
    processes: int = 8,
    executor: str = "threads",
    pause_seconds: float = 0.0,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    fail_fast: bool = False,
    window_size: int = 100,
    feature_columns: Optional[list[str]] = None,
    duckdb_threads: int = 4,
    lookback_years: int = 2,
    refresh_tail_days: int = 3,
    polygon_adjusted: bool = True,
    polygon_include_otc: bool = False,
    calls_per_minute: int = 5,
) -> LoaderConfig:
    start_d = _parse_ymd(start_date, "--start-date")
    end_d = _parse_ymd(end_date, "--end-date")
    cfg = LoaderConfig(
        repo_root=repo_root,
        ticker_source=ticker_source,
        ohlcv_vendor=ohlcv_vendor,
        exclude_test_issues=exclude_test_issues,
        exclude_etfs=exclude_etfs,
        include_exchanges=tuple(x.upper() for x in include_exchanges),
        start_date=start_d,
        end_date=end_d,
        full_refresh=full_refresh,
        interval=interval,
        batch_size=batch_size,
        processes=processes,
        executor="processes" if executor == "processes" else "threads",
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        window_size=window_size,
        feature_columns=tuple(feature_columns or []),
        duckdb_threads=duckdb_threads,
        lookback_years=int(lookback_years),
        refresh_tail_days=int(refresh_tail_days),
        polygon_adjusted=bool(polygon_adjusted),
        polygon_include_otc=bool(polygon_include_otc),
        calls_per_minute=int(calls_per_minute),
    )
    ensure_dirs(cfg.paths)
    return cfg


@app.command()
def universe(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ticker_source: str = typer.Option("nasdaq_trader", "--ticker-source"),
    exclude_test_issues: bool = typer.Option(True, "--exclude-test-issues/--include-test-issues"),
    exclude_etfs: bool = typer.Option(False, "--exclude-etfs/--include-etfs"),
    include_exchanges: tuple[str, str, str] = typer.Option(("NASDAQ", "NYSE", "AMEX"), "--include-exchanges"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ticker_source=ticker_source,
        exclude_test_issues=exclude_test_issues,
        exclude_etfs=exclude_etfs,
        include_exchanges=include_exchanges,
    )
    out = build_universe(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def update(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ohlcv_vendor: str = typer.Option("polygon_grouped", "--ohlcv-vendor"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    full_refresh: bool = typer.Option(False, "--full-refresh"),
    batch_size: int = typer.Option(50, "--batch-size"),
    processes: int = typer.Option(8, "--processes"),
    executor: str = typer.Option("threads", "--executor"),
    pause_seconds: float = typer.Option(0.0, "--pause-seconds"),
    max_retries: int = typer.Option(3, "--max-retries"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
    lookback_years: int = typer.Option(2, "--lookback-years", help="Polygon date-mode: years to backfill."),
    refresh_tail_days: int = typer.Option(3, "--refresh-tail-days", help="Polygon date-mode: re-fetch last N trading days."),
    adjusted: bool = typer.Option(True, "--adjusted/--unadjusted", help="Polygon: adjusted prices."),
    include_otc: bool = typer.Option(False, "--include-otc/--exclude-otc", help="Polygon: include OTC tickers."),
    calls_per_minute: int = typer.Option(5, "--calls-per-minute", help="Polygon free tier: max calls per minute."),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ohlcv_vendor=ohlcv_vendor,
        start_date=start_date,
        end_date=end_date,
        full_refresh=full_refresh,
        batch_size=batch_size,
        processes=processes,
        executor=executor,
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
        lookback_years=lookback_years,
        refresh_tail_days=refresh_tail_days,
        polygon_adjusted=adjusted,
        polygon_include_otc=include_otc,
        calls_per_minute=calls_per_minute,
    )
    update_market_data(cfg)


@app.command("rebuild-last100")
def rebuild_last100(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
    )
    out = rebuild_last_n_bars_from_polygon_date_partitions(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def screen(
    query: str = typer.Option(..., "--query", help="Named query to run (see screener_loader/screening.py)."),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    cfg = _config(repo_root=repo_root)
    df = run_named_query(cfg, query=query, limit=limit)
    print(df)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_windowed_for_sample_ids(cfg: LoaderConfig, windowed_path: Path, sample_ids: list[str]) -> pd.DataFrame:
    if not sample_ids:
        return pd.DataFrame()
    con = connect(cfg)
    ids = pd.DataFrame({"sample_id": [str(x) for x in sample_ids]})
    con.register("ids", ids)
    return con.execute(
        """
        SELECT w.*
        FROM read_parquet(?) AS w
        JOIN ids ON ids.sample_id = w.sample_id
        ORDER BY w.sample_id, w.t
        """,
        [str(windowed_path)],
    ).df()


@models_app.command()
def train(
    labels_csv: Path = typer.Option(..., "--labels-csv", exists=True, dir_okay=False),
    model_type: str = typer.Option("all", "--model-type", help="Model type to train (or 'all')."),
    setup: str = typer.Option("all", "--setup", help="Setup name to filter (or 'all')."),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    normalization: str = typer.Option("none", "--normalization", help="none|per_window_zscore|returns_relative|global_fit"),
    split: str = typer.Option("time", "--split", help="time|random|column"),
    split_column: str = typer.Option("split", "--split-column"),
    train_frac: float = typer.Option(0.7, "--train-frac"),
    val_frac: float = typer.Option(0.15, "--val-frac"),
    test_frac: float = typer.Option(0.15, "--test-frac"),
    seed: int = typer.Option(1337, "--seed"),
    threshold: float = typer.Option(0.5, "--threshold"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
    resolve_non_trading: str = typer.Option("error", "--resolve-nontrading", help="error|previous|next"),
    dedupe: str = typer.Option("error", "--dedupe", help="error|keep_first|keep_last"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
    )

    # 1) Load + store labels
    cal = TradingCalendar("NYSE")
    labels_res = load_labels_csv(
        labels_csv,
        cal=cal,
        resolve_non_trading=resolve_non_trading,  # type: ignore[arg-type]
        dedupe=dedupe,  # type: ignore[arg-type]
    )
    write_labels_store(labels_res, paths=cfg.paths, source_csv=labels_csv)
    labels_df = labels_res.df

    # 2) Assign split (train/val/test)
    labels_df = assign_split(
        labels_df,
        mode=split,  # type: ignore[arg-type]
        split_column=split_column,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    # 3) Build or reuse cached windowed dataset (all labels, all setups)
    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(feature_columns),
        mask_current_day_to_open_only=True,
        require_full_window=True,
    )
    windowed_path = build_windowed_bars(
        labels_df,
        config=cfg,
        spec=spec,
        out_path=cfg.paths.windowed_bars_parquet,
        source_csv=labels_csv,
        cal=cal,
        reuse_if_unchanged=True,
    )

    # Align labels with windowed samples actually present (window builder can drop early samples).
    con = connect(cfg)
    have = con.execute("SELECT DISTINCT sample_id FROM read_parquet(?)", [str(windowed_path)]).df()
    have_set = set(have["sample_id"].astype(str).tolist())
    labels_df = labels_df.copy()
    labels_df["sample_id"] = labels_df.apply(lambda r: stable_sample_id(r["ticker"], r["asof_date"], r["setup"]), axis=1)
    labels_df = labels_df[labels_df["sample_id"].isin(have_set)].reset_index(drop=True)

    # Optional setup filter.
    if setup and setup.lower() != "all":
        labels_df = filter_labels(labels_df, setup=setup)
        if labels_df.empty:
            raise typer.BadParameter(f"No labels remain after filtering setup={setup!r}")

    registry = get_default_registry()
    model_types = resolve_model_types(model_type, registry)
    print(f"[cyan]Training[/cyan] model_types={model_types} on labels={len(labels_df)} windows={windowed_path}")

    # Build split sample id lists.
    splits = {k: v["sample_id"].astype(str).tolist() for k, v in labels_df.groupby("split", sort=False)}
    train_ids = splits.get("train", [])
    val_ids = splits.get("val", [])
    test_ids = splits.get("test", [])

    # Load long windowed data per split.
    feature_cols_dense = ["open", "high", "low", "close", "volume"]
    # If derived features were requested, they’re appended to the long table; include them too.
    for c in feature_columns:
        key = str(c).strip()
        if key:
            feature_cols_dense.append(key)

    train_long = _load_windowed_for_sample_ids(cfg, windowed_path, train_ids)
    val_long = _load_windowed_for_sample_ids(cfg, windowed_path, val_ids)
    test_long = _load_windowed_for_sample_ids(cfg, windowed_path, test_ids)

    if train_long.empty:
        raise RuntimeError("No training windows loaded; check your split/window_size/raw data availability.")

    # Convert to StandardBatch (single-batch per split for now).
    train_batch = build_standard_batch_from_windowed_long(train_long, feature_columns=feature_cols_dense, window_size=window_size)
    val_batch = build_standard_batch_from_windowed_long(val_long, feature_columns=feature_cols_dense, window_size=window_size) if not val_long.empty else None
    test_batch = build_standard_batch_from_windowed_long(test_long, feature_columns=feature_cols_dense, window_size=window_size) if not test_long.empty else None

    # Fit global stats if requested.
    global_stats = None
    if normalization == "global_fit":
        global_stats = fit_global_zscore_stats([train_batch])

    # Apply normalization.
    train_batch = normalize_batch(train_batch, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]
    if val_batch is not None:
        val_batch = normalize_batch(val_batch, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]
    if test_batch is not None:
        test_batch = normalize_batch(test_batch, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]

    for mt in model_types:
        runner = registry[mt]
        rid = _run_id()
        run_dir = cfg.paths.models_dir / mt / (setup if setup and setup.lower() != "all" else "all") / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        artifact = runner.train([train_batch], run_dir=run_dir)

        # Score on val/test splits when present.
        for split_name, batch in [("val", val_batch), ("test", test_batch)]:
            if batch is None:
                continue
            preds = runner.predict([batch], artifact=artifact)
            report = score_binary_predictions(preds, threshold=float(threshold))
            split_dir = run_dir / split_name
            write_score_artifacts(run_dir=split_dir, predictions=preds, report=report)

        print(f"[green]Wrote[/green] {run_dir}")


@models_app.command()
def warm(
    model_type: str = typer.Option(..., "--model-type"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    num_samples: int = typer.Option(500, "--num-samples"),
    window_size: int = typer.Option(100, "--window-size"),
    date_start: Optional[str] = typer.Option(None, "--date-start", help="YYYY-MM-DD"),
    date_end: Optional[str] = typer.Option(None, "--date-end", help="YYYY-MM-DD"),
    ticker_source: str = typer.Option("universe", "--ticker-source", help="universe|labels_only"),
    labels_csv: Optional[Path] = typer.Option(None, "--labels-csv", exists=True, dir_okay=False),
    seed: int = typer.Option(1337, "--seed"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
) -> None:
    cfg = _config(repo_root=repo_root, window_size=window_size, duckdb_threads=duckdb_threads)
    registry = get_default_registry()
    if model_type not in registry:
        raise typer.BadParameter(f"Unknown model_type {model_type!r}. Known: {sorted(registry)}")
    runner = registry[model_type]
    if not runner.supports_warm():
        raise typer.BadParameter(f"model_type {model_type!r} does not support warm() yet")

    cal = TradingCalendar("NYSE")
    start_d = _parse_ymd(date_start, "--date-start")
    end_d = _parse_ymd(date_end, "--date-end")
    if end_d is None:
        end_d = latest_trading_day_on_or_before(date.today(), cal)
    if start_d is None:
        start_d = subtract_years(end_d, 2)

    if ticker_source == "universe":
        uni = load_universe(cfg)
        tickers = uni["ticker"].astype(str).str.upper().tolist()
    elif ticker_source == "labels_only":
        if labels_csv is None:
            raise typer.BadParameter("--labels-csv is required when --ticker-source labels_only")
        labels_res = load_labels_csv(labels_csv, cal=cal)
        tickers = sorted(set(labels_res.df["ticker"].astype(str).str.upper().tolist()))
    else:
        raise typer.BadParameter("--ticker-source must be universe or labels_only")

    # Sample random (ticker, asof_date) pairs on trading days.
    days = cal.valid_trading_days(start_d, end_d)
    if not days:
        raise RuntimeError("No trading days in requested warm date range")

    rng = np.random.default_rng(int(seed))
    # Avoid duplicates because the window builder expects unique (ticker, asof_date, setup).
    target_n = int(num_samples)
    seen: set[tuple[str, date]] = set()
    sampled = []
    # Try a few rounds; duplicates are common with replacement.
    max_draws = max(100, target_n * 20)
    draws = 0
    while len(sampled) < target_n and draws < max_draws:
        draws += 1
        tkr = tickers[int(rng.integers(0, len(tickers)))]
        d = days[int(rng.integers(0, len(days)))]
        key = (tkr, d)
        if key in seen:
            continue
        seen.add(key)
        sampled.append({"ticker": tkr, "asof_date": d, "setup": "_warm", "label": False})
    samples_df = pd.DataFrame(sampled)
    if samples_df.empty:
        raise RuntimeError("Failed to sample any unique warm windows; widen date range or increase universe size.")

    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(),
        mask_current_day_to_open_only=True,
        require_full_window=True,
    )
    rid = _run_id()
    run_dir = cfg.paths.models_dir / model_type / "_warm" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    windowed_path = build_windowed_bars(
        samples_df,
        config=cfg,
        spec=spec,
        out_path=run_dir / "warm_windowed_bars.parquet",
        source_csv=None,
        cal=cal,
        reuse_if_unchanged=False,
    )

    warm_long = pd.read_parquet(windowed_path)
    if warm_long.empty:
        raise RuntimeError("Warm windowed dataset is empty")
    warm_batch = build_standard_batch_from_windowed_long(
        warm_long,
        feature_columns=["open", "high", "low", "close", "volume"],
        window_size=window_size,
    )
    runner.warm([warm_batch], run_dir=run_dir)
    print(f"[green]Wrote[/green] {run_dir}")


def main() -> None:
    app()

