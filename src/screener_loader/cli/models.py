from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import json
import pandas as pd
import numpy as np
import typer
from rich import print

from ..progress import progress_ctx
from ..experiment_tracking import (
    index_models_dir,
    resolve_experiment_id,
    sha256_file,
    stable_fingerprint,
    write_experiment_manifest,
    write_index,
    write_run_meta,
)
from ..calendar_utils import TradingCalendar, latest_trading_day_on_or_before, subtract_years
from ..config import LoaderConfig
from ..duckdb_utils import connect
from ..labels import assign_split, filter_labels, load_labels_csv, write_labels_store
from ..model_registry import TrainedArtifact, get_default_registry, resolve_model_types
from ..normalization import StandardBatch, build_standard_batch_from_windowed_long, fit_global_zscore_stats, normalize_batch
from ..scoring import score_binary_predictions, write_score_artifacts
from ..universe import load_universe
from ..windowed_dataset import (
    WindowedBuildSpec,
    build_windowed_bars,
    stable_sample_id,
)

from .common import _config, _parse_ymd, _run_id
from .root import models_app


def _load_windowed_for_sample_ids(
    cfg: LoaderConfig, windowed_path: Path, sample_ids: list[str], *, con=None  # noqa: ANN001
) -> pd.DataFrame:
    if not sample_ids:
        return pd.DataFrame()
    con = connect(cfg) if con is None else con
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
    normalization: str = typer.Option(
        "none",
        "--normalization",
        help="none|per_window_zscore|per_window_robust_zscore|returns_relative|global_fit",
    ),
    split: str = typer.Option("time", "--split", help="time|random|column"),
    split_column: str = typer.Option("split", "--split-column"),
    train_frac: float = typer.Option(0.7, "--train-frac"),
    val_frac: float = typer.Option(0.15, "--val-frac"),
    test_frac: float = typer.Option(0.15, "--test-frac"),
    seed: int = typer.Option(1337, "--seed"),
    threshold: float = typer.Option(0.5, "--threshold"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
    resolve_non_trading: str = typer.Option("error", "--resolve-nontrading", help="error|previous|next"),
    dedupe: str = typer.Option("error", "--dedupe", help="error|keep_first|keep_last"),
    encoder_dir: Optional[Path] = typer.Option(None, "--encoder-dir", exists=True, file_okay=False, dir_okay=True),
    head_epochs: int = typer.Option(10, "--head-epochs", help="SSL classifier: epochs for head training."),
    head_lr: float = typer.Option(1e-3, "--head-lr", help="SSL classifier: learning rate for head training."),
    head_hidden_dim: int = typer.Option(128, "--head-hidden-dim", help="SSL classifier: hidden dim for MLP head."),
    head_dropout: float = typer.Option(0.1, "--head-dropout", help="Head models: dropout probability."),
    head_batch_size: int = typer.Option(256, "--head-batch-size", help="Head models: minibatch size."),
    device: str = typer.Option("cpu", "--device", help="Device for torch-based models: cpu|cuda|xpu"),
    target_column: str = typer.Option("p_label", "--target-column", help="Pseudo-label column in meta (e.g. p_label)."),
    weight_column: str = typer.Option("weight", "--weight-column", help="Sample weight column in meta (e.g. weight)."),
    unfreeze_last_n_blocks: int = typer.Option(0, "--unfreeze-last-n-blocks", help="torch_ssl_head_student: unfreeze last N encoder blocks."),
    calibration_labels_csv: Optional[Path] = typer.Option(
        None, "--calibration-labels-csv", exists=True, dir_okay=False, help="Optional gold labels CSV for temperature scaling."
    ),
    sample_meta_columns: list[str] = typer.Option(
        [], "--sample-meta-column", help="Copy these per-sample columns from labels into windowed bars."
    ),
    extra_meta_columns: list[str] = typer.Option(
        [], "--extra-meta-column", help="Include these columns into StandardBatch.meta (from windowed bars)."
    ),
    prob_calibration: str = typer.Option("none", "--prob-calibration", help="none|platt|isotonic"),
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

    # Auto-include common per-sample meta columns when present in labels.
    # These must be copied into the windowed bars parquet (sample_meta_columns),
    # then explicitly requested into StandardBatch.meta (extra_meta_columns).
    sample_meta_cols_used = [str(x).strip() for x in (sample_meta_columns or []) if str(x).strip()]
    extra_meta_cols_used = [str(x).strip() for x in (extra_meta_columns or []) if str(x).strip()]
    for key in [str(weight_column).strip(), str(target_column).strip()]:
        if not key:
            continue
        if key in set(labels_df.columns):
            if key not in sample_meta_cols_used:
                sample_meta_cols_used.append(key)
            if key not in extra_meta_cols_used:
                extra_meta_cols_used.append(key)

    # 3) Build or reuse cached windowed dataset (all labels, all setups)
    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(feature_columns),
        sample_meta_columns=tuple(sample_meta_cols_used),
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
    windowed_meta_path = windowed_path.with_suffix(windowed_path.suffix + ".meta.json")
    windowed_meta = None
    try:
        if windowed_meta_path.exists():
            windowed_meta = json.loads(windowed_meta_path.read_text(encoding="utf-8"))
    except Exception:
        windowed_meta = None

    # Record dataset identity for robust experiment tracking.
    labels_stat = labels_csv.stat()
    labels_sha = sha256_file(labels_csv)
    dataset_fingerprint = stable_fingerprint(
        {
            "labels_csv": str(labels_csv.resolve()),
            "labels_csv_sha256": labels_sha,
            "windowed_path": str(windowed_path),
            "windowed_meta": windowed_meta,
            "spec": spec.__dict__,
        }
    )[:12]

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
    train_batch_raw = build_standard_batch_from_windowed_long(
        train_long,
        feature_columns=feature_cols_dense,
        window_size=window_size,
        extra_meta_columns=list(extra_meta_cols_used),
    )
    val_batch_raw = (
        build_standard_batch_from_windowed_long(
            val_long,
            feature_columns=feature_cols_dense,
            window_size=window_size,
            extra_meta_columns=list(extra_meta_cols_used),
        )
        if not val_long.empty
        else None
    )
    test_batch_raw = (
        build_standard_batch_from_windowed_long(
            test_long,
            feature_columns=feature_cols_dense,
            window_size=window_size,
            extra_meta_columns=list(extra_meta_cols_used),
        )
        if not test_long.empty
        else None
    )

    # Fit global stats if requested.
    global_stats = None
    if normalization == "global_fit":
        global_stats = fit_global_zscore_stats([train_batch_raw])

    setup_key = setup if setup and setup.lower() != "all" else "all"
    for mt in model_types:
        runner = registry[mt]
        rid = _run_id()
        run_dir = cfg.paths.models_dir / mt / setup_key / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_meta(
            run_dir,
            repo_root=repo_root,
            extra={"command": "ns models train", "model_type": mt, "run_id": rid, "setup": setup},
        )

        # Runner-specific configuration (kept out of the Runner protocol for now).
        head_cfg_payload = None
        encoder_dir_resolved = None
        if mt in {"ssl_tcn_classifier", "torch_ssl_head_student", "torch_reranker_head"}:
            if mt in {"ssl_tcn_classifier", "torch_ssl_head_student"} and encoder_dir is None:
                raise typer.BadParameter("--encoder-dir is required for this model-type")
            if encoder_dir is not None:
                encoder_dir_resolved = str(encoder_dir.resolve())
                (run_dir / "encoder_dir.txt").write_text(encoder_dir_resolved + "\n", encoding="utf-8")
            head_cfg_payload = {
                "head_epochs": int(head_epochs),
                "head_lr": float(head_lr),
                "head_hidden_dim": int(head_hidden_dim),
                "head_dropout": float(head_dropout),
                "batch_size": int(head_batch_size),
                "device": str(device),
                # Also used as the minibatch shuffle seed inside torch runners.
                "train_seed": int(seed),
                "target_column": str(target_column),
                "weight_column": str(weight_column),
                "unfreeze_last_n_blocks": int(unfreeze_last_n_blocks),
            }
            (run_dir / "head_config.json").write_text(
                json.dumps(head_cfg_payload, indent=2)
                + "\n",
                encoding="utf-8",
            )
            if mt == "torch_ssl_head_student" and calibration_labels_csv is not None:
                (run_dir / "calibration_labels_csv.txt").write_text(
                    str(calibration_labels_csv.resolve()) + "\n", encoding="utf-8"
                )

        # Train config artifact: capture dataset + split + key knobs in machine-readable form.
        train_cfg = {
            "dataset": {
                "labels_csv": str(labels_csv.resolve()),
                "labels_csv_size": int(labels_stat.st_size),
                "labels_csv_mtime_ns": int(labels_stat.st_mtime_ns),
                "labels_csv_sha256": labels_sha,
                "windowed_path": str(windowed_path),
                "windowed_meta": windowed_meta,
                "dataset_fingerprint": dataset_fingerprint,
            },
            "split": {
                "mode": str(split),
                "split_column": str(split_column),
                "train_frac": float(train_frac),
                "val_frac": float(val_frac),
                "test_frac": float(test_frac),
                "seed": int(seed),
            },
            "task": {
                "setup_filter": str(setup),
                "setup_key": str(setup_key),
                "threshold": float(threshold),
            },
            "features": {
                "window_size": int(window_size),
                "feature_columns": list(feature_cols_dense),
                "normalization": str(normalization),
                "sample_meta_columns": [str(x) for x in sample_meta_cols_used],
                "extra_meta_columns": [str(x) for x in extra_meta_cols_used],
            },
            "counts": {
                "labels_rows": int(len(labels_res.df)),
                "labels_rows_after_filter": int(len(labels_df)),
                "train_samples": int(len(train_ids)),
                "val_samples": int(len(val_ids)),
                "test_samples": int(len(test_ids)),
            },
            "runner": {
                "model_type": str(mt),
                "encoder_dir": encoder_dir_resolved,
                "head_config": head_cfg_payload,
                "calibration_labels_csv": (str(calibration_labels_csv.resolve()) if calibration_labels_csv else None),
                "prob_calibration": str(prob_calibration),
            },
        }
        (run_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Experiment manifest + upstream linkage (pretrain).
        upstream = None
        if encoder_dir_resolved:
            enc_dir_p = Path(encoder_dir_resolved)
            enc_exp = resolve_experiment_id(enc_dir_p)
            enc_schema = None
            try:
                sp = enc_dir_p / "schema.json"
                if sp.exists():
                    enc_schema = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                enc_schema = None
            upstream = {
                "encoder_dir": encoder_dir_resolved,
                "pretrain_experiment_id": enc_exp,
                "pretrain_schema_fingerprint": (enc_schema.get("fingerprint") if isinstance(enc_schema, dict) else None),
            }
        write_experiment_manifest(
            run_dir,
            kind="train",
            model_type=str(mt),
            setup=str(setup_key),
            repo_root=repo_root,
            config=train_cfg,
            upstream=upstream,
        )

        # For SSL finetune/head-student, normalization is schema-enforced inside the runner (on shape features).
        # For Stack 6 classic/tabular models, we always need raw OHLCV windows.
        if mt in {"ssl_tcn_classifier", "torch_ssl_head_student", "logreg_stack6", "lgbm_stack6", "hmm_regime"}:
            train_batch = train_batch_raw
            val_batch = val_batch_raw
            test_batch = test_batch_raw
        else:
            train_batch = normalize_batch(train_batch_raw, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]
            val_batch = (
                normalize_batch(val_batch_raw, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]
                if val_batch_raw is not None
                else None
            )
            test_batch = (
                normalize_batch(test_batch_raw, mode=normalization, global_stats=global_stats)  # type: ignore[arg-type]
                if test_batch_raw is not None
                else None
            )

        # Prefer an optional val-aware training API when available (e.g. LightGBM early stopping).
        if val_batch is not None and hasattr(runner, "train_with_val"):
            artifact = runner.train_with_val([train_batch], [val_batch], run_dir=run_dir)  # type: ignore[attr-defined]
        else:
            artifact = runner.train([train_batch], run_dir=run_dir)

        # Optional probability calibration (Platt / Isotonic) using val split.
        cal_method = str(prob_calibration).strip().lower()
        if cal_method not in {"none", "platt", "isotonic"}:
            raise typer.BadParameter("--prob-calibration must be none|platt|isotonic")
        if cal_method != "none" and val_batch is not None and hasattr(runner, "fit_probability_calibration"):
            runner.fit_probability_calibration(  # type: ignore[attr-defined]
                [val_batch],
                artifact=artifact,
                run_dir=run_dir,
                method=cal_method,
            )

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
def scan(
    model_type: str = typer.Option(..., "--model-type", help="Model type to use for inference."),
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    asof_date: Optional[str] = typer.Option(None, "--asof-date", help="YYYY-MM-DD (defaults to latest trading day)."),
    setup: str = typer.Option("scan", "--setup", help="Setup string for sample_id namespace."),
    ticker_source: str = typer.Option("universe", "--ticker-source", help="universe|csv"),
    tickers_csv: Optional[Path] = typer.Option(None, "--tickers-csv", exists=True, dir_okay=False),
    window_size: int = typer.Option(100, "--window-size"),
    limit: int = typer.Option(200, "--limit", help="Print top-N rows (0=print none)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output parquet path."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads"),
) -> None:
    """
    Score many tickers at a single asof_date using a trained model artifact.

    Notes:
    - Uses the same open-time leakage masking: only open on asof_date is visible.
    - Expects classic Stack-6 style artifacts to be present under run_dir (artifact.json).
    """
    cfg = _config(repo_root=repo_root, window_size=int(window_size), duckdb_threads=duckdb_threads)
    registry = get_default_registry()
    if model_type not in registry:
        raise typer.BadParameter(f"Unknown model_type {model_type!r}. Known: {sorted(registry)}")
    runner = registry[model_type]

    cal = TradingCalendar("NYSE")
    d = _parse_ymd(asof_date, "--asof-date")
    if d is None:
        d = latest_trading_day_on_or_before(date.today(), cal)

    ts = str(ticker_source).strip().lower()
    if ts == "universe":
        uni = load_universe(cfg)
        tickers = uni["ticker"].astype(str).str.upper().tolist()
    elif ts == "csv":
        if tickers_csv is None:
            raise typer.BadParameter("--tickers-csv is required when --ticker-source csv")
        df = pd.read_csv(tickers_csv, na_filter=False)
        if "ticker" not in df.columns:
            raise typer.BadParameter("--tickers-csv must have a 'ticker' column")
        tickers = df["ticker"].astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in tickers if t]
    else:
        raise typer.BadParameter("--ticker-source must be universe|csv")

    if not tickers:
        raise RuntimeError("No tickers to scan")

    samples_df = pd.DataFrame({"ticker": tickers, "asof_date": [d] * len(tickers), "setup": str(setup), "label": False})

    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(),
        sample_meta_columns=tuple(),
        mask_current_day_to_open_only=True,
        require_full_window=True,
    )
    windowed_path = build_windowed_bars(
        samples_df,
        config=cfg,
        spec=spec,
        out_path=(Path(run_dir) / "scan_windowed.parquet"),
        source_csv=None,
        cal=cal,
        reuse_if_unchanged=False,
    )

    windowed_long = pd.read_parquet(windowed_path)
    if windowed_long.empty:
        raise RuntimeError("Scan windowed dataset is empty (no full windows available)")

    batch = build_standard_batch_from_windowed_long(
        windowed_long,
        feature_columns=["open", "high", "low", "close", "volume"],
        window_size=int(window_size),
    )

    artifact_path = Path(run_dir) / "artifact.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact.json in run_dir={run_dir}")
    artifact = TrainedArtifact(runner_name=runner.name, path=artifact_path)
    preds = runner.predict([batch], artifact=artifact)
    if preds.empty:
        raise RuntimeError("No predictions returned")
    preds = preds.sort_values(["score"], ascending=[False]).reset_index(drop=True)

    out_path = out or (Path(run_dir) / "scan_predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)
    print(f"[green]Wrote[/green] {out_path} rows={len(preds)}")
    if int(limit) > 0:
        print(preds.head(int(limit)))


@models_app.command("index")
def index(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output path (.parquet|.csv|.jsonl)."),
    limit: int = typer.Option(25, "--limit", help="Max rows to display in the terminal."),
    sort_by: str = typer.Option("meta.timestamp_utc", "--sort-by", help="Column to sort by (best-effort)."),
    desc: bool = typer.Option(True, "--desc/--asc", help="Sort descending/ascending."),
    show_all_columns: bool = typer.Option(False, "--show-all-columns", help="Print many columns (wide table)."),
) -> None:
    """
    Build a single experiment index table from artifacts under data/models/.

    This is meant for methodical optimization: each run already writes configs and
    metrics; this command consolidates them into a single tabular file.
    """
    cfg = _config(repo_root=repo_root)
    res = index_models_dir(cfg.paths.models_dir)
    out_path = out or (cfg.paths.models_dir / "_index.parquet")
    wrote = write_index(res.df, out_path=out_path)
    print(f"[green]Wrote[/green] {wrote} rows={len(res.df)}")

    # Neat terminal display (compact Rich table).
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        return

    df = res.df.copy()
    if df.empty:
        return

    # Best-effort sort.
    if sort_by in df.columns:
        try:
            df = df.sort_values(sort_by, ascending=not bool(desc), na_position="last")
        except Exception:
            pass
    else:
        # Fallback: run_id tends to be timestamp-like (YYYYMMDDTHHMMSSZ).
        if "run_id" in df.columns:
            try:
                df = df.sort_values("run_id", ascending=not bool(desc), na_position="last")
            except Exception:
                pass

    n_show = max(1, min(int(limit), int(len(df))))
    # Add a compact display column to keep tables readable even in narrow terminals.
    if "model_type" in df.columns:
        if "kind" in df.columns:
            kind_letter = (
                df["kind"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"pretrain": "p", "train": "t", "warm": "w"})
                .fillna("u")
            )
            df["_display_model"] = kind_letter + ":" + df["model_type"].astype(str)
        else:
            df["_display_model"] = df["model_type"].astype(str)

    view = df.head(n_show)

    if show_all_columns:
        # Show a lot, but keep stable base columns first.
        base = [c for c in ["kind", "model_type", "setup", "run_id"] if c in df.columns]
        cols = base + [c for c in df.columns if c not in set(base)]
        table = Table(title=f"Models index (showing {n_show}/{len(df)})", show_lines=False)
        for c in cols:
            table.add_column(str(c), overflow="fold", no_wrap=False)

        def _fmt(v) -> str:  # noqa: ANN001
            if v is None:
                return ""
            if isinstance(v, float):
                if v != v:  # NaN
                    return ""
                return f"{v:.4f}"
            return str(v)

        for _, r in view.iterrows():
            table.add_row(*[_fmt(r.get(c)) for c in cols])

        Console().print(table)
        return

    # Default compact view: short headers + truncation.
    exp_col = "exp.experiment_id" if "exp.experiment_id" in df.columns else ("meta.experiment_id" if "meta.experiment_id" in df.columns else None)
    up_col = "upstream.pretrain_experiment_id" if "upstream.pretrain_experiment_id" in df.columns else None
    sha_col = "meta.git_sha" if "meta.git_sha" in df.columns else None
    ts_col = "meta.timestamp_utc" if "meta.timestamp_utc" in df.columns else None

    # Keep default output narrow enough for typical terminals.
    col_specs: list[tuple[str, str]] = []
    if exp_col:
        col_specs.append((exp_col, "exp"))
    model_key = "_display_model" if "_display_model" in df.columns else "model_type"
    col_specs += [(model_key, "model"), ("setup", "setup"), ("run_id", "run")]
    if up_col:
        col_specs.append((up_col, "up"))
    col_specs += [
        ("val_metrics.overall.auprc", "val.auprc"),
        ("val_metrics.overall.at_threshold.f1", "val.f1"),
        ("test_metrics.overall.auprc", "test.auprc"),
    ]
    if ts_col:
        col_specs.append((ts_col, "ts"))

    # Drop missing columns.
    col_specs = [(k, h) for (k, h) in col_specs if k in df.columns]

    # Small summary (helps quickly sanity-check counts).
    try:
        kind_counts = df["kind"].value_counts(dropna=False).to_dict() if "kind" in df.columns else {}
        if kind_counts:
            summary = ", ".join([f"{k}={int(v)}" for k, v in kind_counts.items()])
            print(f"[dim]Kinds:[/dim] {summary}")
    except Exception:
        pass

    table = Table(
        title=f"Models index (showing {n_show}/{len(df)})",
        show_lines=False,
        pad_edge=False,
    )
    # Keep things on one line; truncate long cells.
    for k, h in col_specs:
        if h == "kind":
            table.add_column("k", overflow="ellipsis", no_wrap=True, width=2, justify="center")
        elif h == "exp":
            table.add_column("exp", overflow="ellipsis", no_wrap=True, max_width=12)
        elif h == "up":
            table.add_column("up", overflow="ellipsis", no_wrap=True, max_width=12)
        elif h == "model":
            table.add_column("model", overflow="ellipsis", no_wrap=True, max_width=22)
        elif h == "setup":
            table.add_column("setup", overflow="ellipsis", no_wrap=True, max_width=10)
        elif h == "run":
            table.add_column("run", overflow="ellipsis", no_wrap=True, max_width=16)
        elif h in {"val.auroc", "val.auprc", "val.f1", "test.auroc", "test.auprc"}:
            table.add_column(h, overflow="ellipsis", no_wrap=True, max_width=9, justify="right")
        elif h == "sha":
            table.add_column("sha", overflow="ellipsis", no_wrap=True, max_width=10)
        elif h == "ts":
            table.add_column("ts", overflow="ellipsis", no_wrap=True, max_width=19)
        else:
            table.add_column(h, overflow="ellipsis", no_wrap=True)

    def _fmt_cell(col: str, v) -> str:  # noqa: ANN001
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:  # NaN
                return ""
            return f"{v:.4f}"
        s = str(v)
        if col in {exp_col, up_col} and s:
            return s[:12]
        if col == sha_col and s:
            return s[:10]
        if col == ts_col and s:
            # Keep compact: drop microseconds if present.
            return s.split(".")[0].replace("T", " ")
        return s

    for _, r in view.iterrows():
        table.add_row(*[_fmt_cell(k, r.get(k)) for (k, _h) in col_specs])

    Console().print(table)


@models_app.command()
def pretrain(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    resume_from: Optional[Path] = typer.Option(
        None, "--resume-from", exists=True, file_okay=False, help="Resume an existing pretrain run_dir."
    ),
    reuse_windowed_from: Optional[Path] = typer.Option(
        None,
        "--reuse-windowed-from",
        exists=True,
        file_okay=False,
        help="Reuse precomputed windowed data from another pretrain run_dir (faster sweeps).",
    ),
    num_samples: int = typer.Option(5000, "--num-samples"),
    window_max: int = typer.Option(96, "--window-max", help="Max window length used for window building."),
    crop_length: list[int] = typer.Option([64, 96], "--crop-length", help="Crop lengths used during training."),
    date_start: Optional[str] = typer.Option(None, "--date-start", help="YYYY-MM-DD"),
    date_end: Optional[str] = typer.Option(None, "--date-end", help="YYYY-MM-DD"),
    ticker_source: str = typer.Option("universe", "--ticker-source", help="universe|labels_only"),
    labels_csv: Optional[Path] = typer.Option(None, "--labels-csv", exists=True, dir_okay=False),
    seed: int = typer.Option(1337, "--seed"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
    # SSL feature/normalization
    include_pos: bool = typer.Option(False, "--include-pos/--no-include-pos", help="Include pos-to-trend feature."),
    normalization: str = typer.Option(
        "per_window_zscore", "--normalization", help="per_window_zscore|per_window_robust_zscore"
    ),
    # Masking policy
    mask_mode: list[str] = typer.Option(["both"], "--mask-mode", help="time|feature|both (repeatable)"),
    mask_rate_time: float = typer.Option(0.10, "--mask-rate-time"),
    mask_rate_feat: float = typer.Option(0.15, "--mask-rate-feat"),
    use_mask_token: bool = typer.Option(True, "--use-mask-token/--no-mask-token"),
    loss: str = typer.Option("huber", "--loss", help="huber|l1"),
    huber_delta: float = typer.Option(1.0, "--huber-delta"),
    # Encoder
    d_model: int = typer.Option(128, "--d-model"),
    num_blocks: int = typer.Option(8, "--num-blocks"),
    kernel_size: int = typer.Option(3, "--kernel-size"),
    dropout: float = typer.Option(0.1, "--dropout"),
    # Training loop
    epochs: int = typer.Option(5, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    chunk_size: int = typer.Option(2048, "--chunk-size", help="How many sample_ids to load per DuckDB chunk."),
    lr: float = typer.Option(1e-3, "--lr"),
    max_steps: int = typer.Option(0, "--max-steps", help="Optional cap on total optimizer steps (0=uncapped)."),
    device: str = typer.Option("cpu", "--device", help="cpu|cuda|xpu"),
    amp: bool = typer.Option(False, "--amp/--no-amp", help="Use AMP when supported (cuda only)."),
    grad_accum_steps: int = typer.Option(1, "--grad-accum-steps", help="Accumulate gradients for N microbatches."),
    save_every_steps: int = typer.Option(0, "--save-every-steps", help="Write a resumable checkpoint every N optimizer steps."),
    # Augmentations
    jitter_sigma: float = typer.Option(0.0, "--jitter-sigma"),
    augment_censor_last_prob: float = typer.Option(
        1.0,
        "--augment-censor-last-prob",
        help="Randomly censor the last timestep (recommended=1.0 to match open-only inference regime).",
    ),
) -> None:
    """
    Self-supervised pretraining: masked modeling over unlabeled OHLCV windows.

    Writes artifacts under:
      data/models/ssl_tcn_masked_pretrain/_pretrain/<run_id>/
    """
    # Torch is optional; import lazily so non-ML commands work without it.
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required for `ns models pretrain`. Install with: pip install -e '.[ml]'\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e

    # Make pretraining reproducible by default: weight init + any torch-side randomness.
    # (Mask sampling is already seeded deterministically from the numpy RNG inside masked_modeling.)
    torch.manual_seed(int(seed))
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass

    from ..ssl.augment import CensorConfig, CropConfig, JitterConfig, jitter, maybe_censor_last_timestep, random_crop
    from ..ssl.features import ShapeFeatureSpec, build_shape_features_from_ohlcv_batch
    from ..ssl.masked_modeling import MaskedModelingConfig, MaskingConfig, MaskedModelingModel
    from ..ssl.schema import SSLSchema, read_schema, write_schema
    from ..ssl.tcn import TCNConfig

    cfg = _config(repo_root=repo_root, window_size=int(window_max), duckdb_threads=duckdb_threads)
    cal = TradingCalendar("NYSE")

    rng = np.random.default_rng(int(seed))

    # Resolve run_dir + windowed dataset (fresh run or resume).
    sampled: list[dict[str, object]] = []
    sampling_payload: dict[str, object] = {}
    model_type = "ssl_tcn_masked_pretrain"

    if resume_from is not None and reuse_windowed_from is not None:
        raise typer.BadParameter("Use only one of --resume-from or --reuse-windowed-from")

    if resume_from is not None:
        run_dir = Path(resume_from)
        if not (run_dir / "pretrain_windowed_bars.parquet").exists():
            raise typer.BadParameter("--resume-from must point to a pretrain run_dir with pretrain_windowed_bars.parquet")
        windowed_path = run_dir / "pretrain_windowed_bars.parquet"
        try:
            sampling_payload = json.loads((run_dir / "sampling.json").read_text(encoding="utf-8"))
        except Exception:
            sampling_payload = {}

        # Best-effort: load schema.json so resume does not depend on re-specifying flags.
        try:
            sch = read_schema(run_dir / "schema.json")
            window_max = int(sch.window_max)
            crop_length = [int(x) for x in (sch.crop_lengths or [])] or crop_length
            normalization = str(sch.normalization)
            mask_mode = [str(x) for x in (sch.mask_mode or [])] or mask_mode
            mask_rate_time = float(sch.mask_rate_time)
            mask_rate_feat = float(sch.mask_rate_feat)
            use_mask_token = bool(sch.use_mask_token)
            loss = str(sch.loss)
            huber_delta = float(sch.huber_delta)
            augment_censor_last_prob = float(sch.augment_censor_last_timestep_prob)
            include_pos = "pos" in set([str(x) for x in (sch.feature_names or [])])

            enc = sch.encoder or {}
            d_model = int(enc.get("d_model", d_model))
            num_blocks = int(enc.get("num_blocks", num_blocks))
            kernel_size = int(enc.get("kernel_size", kernel_size))
            dropout = float(enc.get("dropout", dropout))
        except Exception:
            pass

        # Recompute config with schema window size (doesn't affect run_dir path).
        cfg = _config(repo_root=repo_root, window_size=int(window_max), duckdb_threads=duckdb_threads)
    elif reuse_windowed_from is not None:
        src_dir = Path(reuse_windowed_from)
        src_windowed = src_dir / "pretrain_windowed_bars.parquet"
        if not src_windowed.exists():
            raise typer.BadParameter("--reuse-windowed-from must contain pretrain_windowed_bars.parquet")
        windowed_path = src_windowed
        try:
            sampling_payload = json.loads((src_dir / "sampling.json").read_text(encoding="utf-8"))
        except Exception:
            sampling_payload = {}

        # If possible, align window_max with the source schema (to avoid reshape mismatch).
        try:
            sch = read_schema(src_dir / "schema.json")
            window_max = int(sch.window_max)
            cfg = _config(repo_root=repo_root, window_size=int(window_max), duckdb_threads=duckdb_threads)
        except Exception:
            pass

        # New run dir (we do NOT overwrite the source run).
        rid = _run_id()
        run_dir = cfg.paths.models_dir / model_type / "_pretrain" / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_meta(
            run_dir,
            repo_root=repo_root,
            extra={
                "command": "ns models pretrain",
                "model_type": model_type,
                "run_id": rid,
                "reuse_windowed_from": str(src_dir),
            },
        )
        (run_dir / "windowed_source.json").write_text(
            json.dumps(
                {
                    "reuse_windowed_from": str(src_dir),
                    "windowed_path": str(src_windowed),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
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

        # If we're relying on per-ticker raw parquet (no polygon partitions), prefilter tickers
        # to avoid wasting sampling budget on tickers with no data.
        try:
            parts = cfg.paths.list_polygon_grouped_daily_partitions()
        except Exception:
            parts = {}
        if not parts:
            have_raw = [t for t in tickers if cfg.paths.raw_ticker_parquet(str(t)).exists()]
            if have_raw:
                tickers = have_raw

        days = cal.valid_trading_days(start_d, end_d)
        if not days:
            raise RuntimeError("No trading days in requested pretrain date range")

        # Fresh run dir.
        rid = _run_id()
        run_dir = cfg.paths.models_dir / model_type / "_pretrain" / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_meta(
            run_dir,
            repo_root=repo_root,
            extra={"command": "ns models pretrain", "model_type": model_type, "run_id": rid},
        )

        # Sample unlabeled windows.
        target_n = int(num_samples)
        seen: set[tuple[str, date]] = set()
        max_draws = max(100, target_n * 25)
        draws = 0
        while len(sampled) < target_n and draws < max_draws:
            draws += 1
            tkr = tickers[int(rng.integers(0, len(tickers)))]
            d = days[int(rng.integers(0, len(days)))]
            key = (tkr, d)
            if key in seen:
                continue
            seen.add(key)
            sampled.append({"ticker": tkr, "asof_date": d, "setup": "_pretrain", "label": False})

        samples_df = pd.DataFrame(sampled)
        if samples_df.empty:
            raise RuntimeError("Failed to sample any unique pretrain windows; widen date range or increase universe size.")

        spec = WindowedBuildSpec(
            window_size=int(window_max),
            feature_columns=tuple(),
            mask_current_day_to_open_only=False,  # SSL pretraining: uncensored (use censor augmentation instead)
            require_full_window=True,
        )

        # Sampling config artifact (documents the unlabeled dataset identity).
        sample_keys = sorted([f"{r['ticker']}|{r['asof_date'].isoformat()}" for r in sampled])
        sampling_fingerprint = stable_fingerprint(
            {
                "ticker_source": str(ticker_source),
                "labels_csv": str(labels_csv.resolve()) if labels_csv is not None else None,
                "date_start": start_d.isoformat(),
                "date_end": end_d.isoformat(),
                "seed": int(seed),
                "num_samples": int(num_samples),
                "sample_keys": sample_keys,
            }
        )[:12]
        sampling_payload = {
            "ticker_source": str(ticker_source),
            "labels_csv": (str(labels_csv.resolve()) if labels_csv is not None else None),
            "date_start": start_d.isoformat(),
            "date_end": end_d.isoformat(),
            "seed": int(seed),
            "num_samples_requested": int(num_samples),
            "num_samples_sampled": int(len(sampled)),
            "sampling_fingerprint": sampling_fingerprint,
            "sample_keys_sha1": stable_fingerprint({"sample_keys": sample_keys}),
            "unique_tickers": int(len(set([r["ticker"] for r in sampled]))),
        }
        (run_dir / "sampling.json").write_text(
            json.dumps(sampling_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        windowed_path = build_windowed_bars(
            samples_df,
            config=cfg,
            spec=spec,
            out_path=run_dir / "pretrain_windowed_bars.parquet",
            source_csv=None,
            cal=cal,
            reuse_if_unchanged=False,
        )

    # Determine sample_ids actually present (window builder can drop samples).
    con = connect(cfg)
    have = con.execute("SELECT DISTINCT sample_id FROM read_parquet(?)", [str(windowed_path)]).df()
    sample_ids = have["sample_id"].astype(str).tolist()
    if not sample_ids:
        raise RuntimeError("No sample_id rows found in pretrain windowed dataset")

    # Build model/config.
    feat_spec = ShapeFeatureSpec(include_pos=bool(include_pos))
    in_features = len(feat_spec.feature_names)
    enc_cfg = TCNConfig(
        in_features=int(in_features),
        d_model=int(d_model),
        num_blocks=int(num_blocks),
        kernel_size=int(kernel_size),
        dropout=float(dropout),
    )
    masking_cfg = MaskingConfig(
        mask_mode=tuple(str(m).strip().lower() for m in mask_mode if str(m).strip()),
        mask_rate_time=float(mask_rate_time),
        mask_rate_feat=float(mask_rate_feat),
        use_mask_token=bool(use_mask_token),
    )
    model_cfg = MaskedModelingConfig(
        encoder=enc_cfg,
        masking=masking_cfg,
        loss=str(loss).strip().lower(),  # type: ignore[arg-type]
        huber_delta=float(huber_delta),
    )

    model = MaskedModelingModel(model_cfg)
    dev = torch.device(str(device))
    model.to(dev)
    model.train(True)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    use_amp = bool(amp) and (str(dev.type).lower() == "cuda")
    import contextlib

    if use_amp:
        # Prefer non-deprecated torch.amp API; fall back for older torch builds.
        try:
            scaler = torch.amp.GradScaler("cuda")

            def _autocast():  # noqa: ANN001
                return torch.amp.autocast("cuda")

        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

            def _autocast():  # noqa: ANN001
                return torch.cuda.amp.autocast(enabled=True)

    else:
        scaler = None

        def _autocast():  # noqa: ANN001
            return contextlib.nullcontext()

    crop_cfg = CropConfig(crop_lengths=tuple(int(x) for x in crop_length))
    jit_cfg = JitterConfig(sigma=float(jitter_sigma))
    censor_cfg = CensorConfig(prob=float(augment_censor_last_prob))

    total_steps = 0  # optimizer steps
    last_loss = None
    loss_sum = 0.0
    loss_n = 0  # number of microbatches (forward passes)

    # Resume: restore model/opt/rng/step counter.
    ckpt_path = run_dir / "pretrain_ckpt.pt"
    if resume_from is not None:
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=str(dev))
            if "model" not in ckpt or "opt" not in ckpt:
                raise RuntimeError(f"Invalid checkpoint (missing model/opt): {ckpt_path}")
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            total_steps = int(ckpt.get("total_steps") or 0)
            rng_state = ckpt.get("rng_state")
            if isinstance(rng_state, dict):
                rng.bit_generator.state = rng_state
        else:
            # Fallback: allow resuming weights-only from encoder.pt (but optimizer state is lost).
            enc_p = run_dir / "encoder.pt"
            if not enc_p.exists():
                raise RuntimeError(f"--resume-from requested but no checkpoint found: {ckpt_path} (and no {enc_p})")
            model.encoder.load_state_dict(torch.load(enc_p, map_location=str(dev)))

    grad_accum_steps = max(1, int(grad_accum_steps))
    save_every_steps = max(0, int(save_every_steps))

    # Progress bars:
    # - epochs: always known
    # - steps: known only when max_steps is set; otherwise show an indeterminate bar
    steps_total = int(max_steps) if int(max_steps) > 0 else None
    with progress_ctx(transient=False) as progress:
        ep_task = progress.add_task("SSL pretrain epochs", total=int(epochs))
        step_task = progress.add_task("SSL pretrain steps", total=steps_total)

        accum = 0
        opt.zero_grad(set_to_none=True)

        for ep in range(int(epochs)):
            progress.update(ep_task, completed=ep)

            # Shuffle sample_ids each epoch.
            perm = rng.permutation(len(sample_ids))
            shuffled = [sample_ids[int(i)] for i in perm.tolist()]

            for i0 in range(0, len(shuffled), int(chunk_size)):
                chunk_ids = shuffled[i0 : i0 + int(chunk_size)]
                long_df = _load_windowed_for_sample_ids(cfg, windowed_path, chunk_ids, con=con)
                if long_df.empty:
                    continue
                ohlcv_batch = build_standard_batch_from_windowed_long(
                    long_df,
                    feature_columns=["open", "high", "low", "close", "volume"],
                    window_size=int(window_max),
                )
                shape_batch = build_shape_features_from_ohlcv_batch(ohlcv_batch, spec=feat_spec)

                x_all = shape_batch.x_seq.astype(float, copy=False)
                m_all = shape_batch.mask_seq.astype(bool, copy=False)
                n = int(x_all.shape[0])
                order = rng.permutation(n)

                for j0 in range(0, n, int(batch_size)):
                    sel = order[j0 : j0 + int(batch_size)]
                    xb = x_all[sel].copy()
                    mb = m_all[sel].copy()

                    # Optional censoring augmentation (simulate inference regime), applied on shape features.
                    xb, mb = maybe_censor_last_timestep(xb, mb, rng=rng, cfg=censor_cfg)
                    # Multi-scale crop
                    xb, mb = random_crop(xb, mb, rng=rng, cfg=crop_cfg)
                    # Mild noise
                    xb = jitter(xb, mb, rng=rng, cfg=jit_cfg)

                    # Normalize per window; keep mask for loss.
                    mini = StandardBatch(
                        x_seq=xb,
                        mask_seq=mb,
                        y=shape_batch.y[sel],
                        meta={k: np.asarray(v)[sel] for k, v in shape_batch.meta.items()},
                    )
                    mini = normalize_batch(mini, mode=normalization, global_stats=None)  # type: ignore[arg-type]

                    x_in = np.where(mini.mask_seq, mini.x_seq, 0.0).astype(np.float32)
                    vmask = mini.mask_seq.astype(bool)

                    xt = torch.from_numpy(x_in).to(dev)
                    mt = torch.from_numpy(vmask).to(dev)

                    with _autocast():
                        loss_t, _recon, _m = model.forward(xt, mt, rng=rng)

                    # Accumulate gradients over microbatches.
                    loss_scaled = loss_t / float(grad_accum_steps)
                    if use_amp:
                        assert scaler is not None
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    accum += 1

                    # Track raw (unscaled) loss for logging.
                    last_loss = float(loss_t.detach().float().cpu().item())
                    loss_sum += float(last_loss)
                    loss_n += 1

                    if accum >= grad_accum_steps:
                        # Step optimizer.
                        if use_amp:
                            assert scaler is not None
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
                        if use_amp:
                            assert scaler is not None
                            scaler.step(opt)
                            scaler.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                        accum = 0

                        total_steps += 1

                        # Avoid updating task description too frequently (keeps overhead low).
                        if total_steps % 10 == 0:
                            progress.update(step_task, advance=1, description=f"SSL pretrain steps (loss={last_loss:.4f})")
                        else:
                            progress.update(step_task, advance=1)

                        if save_every_steps and total_steps % save_every_steps == 0:
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "opt": opt.state_dict(),
                                    "total_steps": int(total_steps),
                                    "rng_state": rng.bit_generator.state,
                                },
                                ckpt_path,
                            )

                        if max_steps and total_steps >= int(max_steps):
                            break

                if max_steps and total_steps >= int(max_steps):
                    break
            if max_steps and total_steps >= int(max_steps):
                break

        progress.update(ep_task, completed=int(epochs))

    # Persist artifacts.
    enc_path = run_dir / "encoder.pt"
    torch.save(model.encoder.state_dict(), enc_path)

    schema = SSLSchema(
        model_type=model_type,
        feature_names=list(feat_spec.feature_names),
        normalization=str(normalization),
        window_max=int(window_max),
        crop_lengths=[int(x) for x in crop_length],
        mask_mode=[str(m) for m in mask_mode],
        mask_rate_time=float(mask_rate_time),
        mask_rate_feat=float(mask_rate_feat),
        use_mask_token=bool(use_mask_token),
        loss=str(loss),
        huber_delta=float(huber_delta),
        encoder={
            "type": "tcn",
            "in_features": int(enc_cfg.in_features),
            "d_model": int(enc_cfg.d_model),
            "num_blocks": int(enc_cfg.num_blocks),
            "kernel_size": int(enc_cfg.kernel_size),
            "dropout": float(enc_cfg.dropout),
        },
        pretrain_mask_current_day_to_open_only=False,
        augment_censor_last_timestep_prob=float(augment_censor_last_prob),
    )
    write_schema(run_dir / "schema.json", schema)

    (run_dir / "pretrain_config.json").write_text(
        json.dumps(
            {
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "chunk_size": int(chunk_size),
                "lr": float(lr),
                "seed": int(seed),
                "device": str(device),
                "amp": bool(use_amp),
                "grad_accum_steps": int(grad_accum_steps),
                "save_every_steps": int(save_every_steps),
                "num_sample_ids": int(len(sample_ids)),
                "microbatches": int(loss_n),
                "mean_loss": (float(loss_sum) / float(loss_n) if int(loss_n) > 0 else None),
                "total_steps": int(total_steps),
                "last_loss": last_loss,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[green]Wrote[/green] {run_dir}")
    # Experiment manifest (deterministic, for grouping).
    try:
        schema_payload = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    except Exception:
        schema_payload = {}
    try:
        pretrain_payload = json.loads((run_dir / "pretrain_config.json").read_text(encoding="utf-8"))
    except Exception:
        pretrain_payload = {}
    write_experiment_manifest(
        run_dir,
        kind="pretrain",
        model_type=model_type,
        setup="_pretrain",
        repo_root=repo_root,
        config={
            "schema_fingerprint": schema_payload.get("fingerprint"),
            "schema": schema_payload,
            "pretrain": pretrain_payload,
            "sampling": sampling_payload,
        },
        upstream=None,
    )


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
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
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
    write_run_meta(
        run_dir,
        repo_root=repo_root,
        extra={"command": "ns models warm", "model_type": model_type, "run_id": rid},
    )
    # Sampling config for warm runs.
    sample_keys = sorted([f"{r['ticker']}|{r['asof_date'].isoformat()}" for r in sampled])
    sampling_fingerprint = stable_fingerprint(
        {
            "ticker_source": str(ticker_source),
            "labels_csv": str(labels_csv.resolve()) if labels_csv is not None else None,
            "date_start": start_d.isoformat(),
            "date_end": end_d.isoformat(),
            "seed": int(seed),
            "num_samples": int(num_samples),
            "sample_keys": sample_keys,
        }
    )[:12]
    (run_dir / "sampling.json").write_text(
        json.dumps(
            {
                "ticker_source": str(ticker_source),
                "labels_csv": (str(labels_csv.resolve()) if labels_csv is not None else None),
                "date_start": start_d.isoformat(),
                "date_end": end_d.isoformat(),
                "seed": int(seed),
                "num_samples_requested": int(num_samples),
                "num_samples_sampled": int(len(sampled)),
                "sampling_fingerprint": sampling_fingerprint,
                "sample_keys_sha1": stable_fingerprint({"sample_keys": sample_keys}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    # Experiment manifest for warm runs (groups by data + code + runner).
    write_experiment_manifest(
        run_dir,
        kind="warm",
        model_type=str(model_type),
        setup="_warm",
        repo_root=repo_root,
        config={"window_size": int(window_size), "num_samples": int(num_samples), "sampling_fingerprint": sampling_fingerprint},
        upstream=None,
    )
    print(f"[green]Wrote[/green] {run_dir}")


@models_app.command("rerank-latest")
def rerank_latest(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    encoder_dir: Optional[Path] = typer.Option(None, "--encoder-dir", exists=True, file_okay=False, dir_okay=True),
    candidate_parquet: Optional[Path] = typer.Option(None, "--candidate-parquet", exists=True, dir_okay=False),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    end_lookback_bars: int = typer.Option(7, "--end-lookback-bars"),
    limit: int = typer.Option(50, "--limit", help="Print top-N ranked candidates."),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional output parquet path for predictions."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    """
    Propose candidates (or load a candidates parquet), build windows, and score with a trained reranker head.
    """
    cfg = _config(
        repo_root=repo_root,
        window_size=int(window_size),
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
    )
    cal = TradingCalendar("NYSE")

    if candidate_parquet is not None:
        cands = pd.read_parquet(candidate_parquet)
    else:
        from ..candidates import CandidateSpec, propose_latest_candidates

        cands = propose_latest_candidates(
            cfg, spec=CandidateSpec(window_size=int(window_size), end_lookback_bars=int(end_lookback_bars))
        )

    if cands.empty:
        print("[yellow]No candidates to rerank[/yellow]")
        return

    # Ensure required columns exist.
    base_cols = {"ticker", "asof_date", "setup", "label"}
    missing = base_cols - set(cands.columns)
    if missing:
        raise typer.BadParameter(f"Candidates missing required columns: {sorted(missing)}")

    # Determine which cand_* meta columns to carry.
    meta_cols = sorted([c for c in cands.columns if str(c).startswith("cand_")])

    # Build a windowed dataset for candidate samples.
    out_windowed = cfg.paths.derived_dir / "windowed_candidates.parquet"
    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(feature_columns),
        sample_meta_columns=tuple(meta_cols),
        mask_current_day_to_open_only=True,
        require_full_window=True,
    )
    windowed_path = build_windowed_bars(
        cands,
        config=cfg,
        spec=spec,
        out_path=out_windowed,
        source_csv=None,
        cal=cal,
        reuse_if_unchanged=False,
    )
    long_df = pd.read_parquet(windowed_path)
    if long_df.empty:
        print("[yellow]Candidate windowed dataset is empty[/yellow]")
        return

    dense_features = ["open", "high", "low", "close", "volume"]
    for c in feature_columns:
        key = str(c).strip()
        if key:
            dense_features.append(key)

    batch = build_standard_batch_from_windowed_long(
        long_df,
        feature_columns=dense_features,
        window_size=int(window_size),
        extra_meta_columns=meta_cols,
    )

    # Load runner.
    registry = get_default_registry()
    if "torch_reranker_head" not in registry:
        raise RuntimeError("torch_reranker_head is not available. Install with: pip install -e '.[ml]'")
    runner = registry["torch_reranker_head"]

    # Load artifact config, optionally overriding encoder_dir.
    artifact_path = model_dir / "reranker_head.json"
    if not artifact_path.exists():
        raise typer.BadParameter(f"Expected reranker_head.json under --model-dir; got {artifact_path}")

    from ..model_registry import TrainedArtifact

    if encoder_dir is None:
        artifact = TrainedArtifact(runner_name=runner.name, path=artifact_path)
    else:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["encoder_dir"] = str(encoder_dir.resolve())
        tmp = cfg.paths.meta_dir / "rerank_override.json"
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        artifact = TrainedArtifact(runner_name=runner.name, path=tmp)

    preds = runner.predict([batch], artifact=artifact)
    if preds.empty:
        print("[yellow]No predictions returned[/yellow]")
        return

    preds = preds.sort_values(["score"], ascending=False).reset_index(drop=True)
    out_path = out or (cfg.paths.derived_dir / "rerank_latest_predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)
    print(f"[green]Wrote[/green] {out_path} rows={len(preds)}")
    if int(limit) > 0:
        print(preds.head(int(limit)))

