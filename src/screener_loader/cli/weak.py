from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import json
import pandas as pd
import typer
from rich import print

from ..calendar_utils import TradingCalendar
from ..model_registry import TrainedArtifact, get_default_registry
from ..normalization import build_standard_batch_from_windowed_long
from ..windowed_dataset import (
    ContextWindowBuildSpec,
    WindowedBuildSpec,
    build_context_window_bars,
    build_windowed_bars,
    stable_sample_id,
)
from .common import _config, _parse_ymd, _run_id, _sha1_json
from .root import weak_app


@weak_app.command("build-candidate-pool")
def weak_build_candidate_pool(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    setup: list[str] = typer.Option(["flag"], "--setup", help="Repeatable. Examples: flag, gap_go, gap_fade."),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    min_close: float = typer.Option(1.0, "--min-close"),
    min_dollar_vol: float = typer.Option(0.0, "--min-dollar-vol"),
    max_candidates: int = typer.Option(0, "--max-candidates", help="0=unlimited; otherwise keep top-N by recency/liquidity."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output parquet path."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    """
    Build a reusable candidate pool parquet for weak supervision.
    """
    from ..weak_supervision.candidate_pool import CandidatePoolSpec, build_candidate_pool

    cfg = _config(repo_root=repo_root, duckdb_threads=duckdb_threads)
    start_d = _parse_ymd(start_date, "--start-date")
    end_d = _parse_ymd(end_date, "--end-date")

    spec = CandidatePoolSpec(
        setups=tuple(str(s).strip() for s in setup if str(s).strip()),
        start_date=start_d,
        end_date=end_d,
        min_close=float(min_close),
        min_dollar_vol=float(min_dollar_vol),
        max_candidates=(int(max_candidates) if int(max_candidates) > 0 else None),
    )
    df = build_candidate_pool(cfg, spec=spec)
    if df.empty:
        print("[yellow]No candidates in pool[/yellow]")
        return
    out_path = out or (cfg.paths.derived_dir / "candidate_pool.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[green]Wrote[/green] {out_path} rows={len(df)} setups={sorted(set(df['setup']))}")


@weak_app.command("generate-pseudo")
def weak_generate_pseudo(
    candidate_pool: Path = typer.Option(..., "--candidate-pool", exists=True, dir_okay=False),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    setup: str = typer.Option(..., "--setup", help="flag|gap_go|gap_fade"),
    past_window: int = typer.Option(48, "--past-window"),
    future_window: int = typer.Option(32, "--future-window"),
    combine: str = typer.Option("label_model", "--combine", help="majority|label_model"),
    label_def_version: str = typer.Option("v1", "--label-def-version"),
    max_candidates: int = typer.Option(0, "--max-candidates", help="0=use all rows in candidate_pool; else head-N."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory for artifacts."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    """
    Generate pseudo labels from weak heuristics + a label model.
    """
    from ..weak_supervision.diagnostics import lf_summary, sample_vote_stats
    from ..weak_supervision.features import ContextSpec, build_sample_features_from_context_long
    from ..weak_supervision.label_model import fit_independent_label_model, majority_vote
    from ..weak_supervision.labeling_functions import apply_labeling_functions, make_flag_lfs, make_gap_lfs

    setup_key = str(setup).strip()
    if setup_key not in {"flag", "gap_go", "gap_fade"}:
        raise typer.BadParameter("--setup must be flag|gap_go|gap_fade")
    combine_key = str(combine).strip().lower()
    if combine_key not in {"majority", "label_model"}:
        raise typer.BadParameter("--combine must be majority|label_model")

    cfg = _config(repo_root=repo_root, duckdb_threads=duckdb_threads)
    rid = _run_id()
    out_dir = out_dir or (cfg.paths.derived_dir / "weak_supervision" / "pseudo" / setup_key / rid)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(candidate_pool)
    if pool.empty:
        raise RuntimeError("candidate_pool is empty")
    pool = pool.copy()
    pool["ticker"] = pool["ticker"].astype(str).str.upper()
    pool["asof_date"] = pd.to_datetime(pool["asof_date"], errors="coerce").dt.date
    pool["setup"] = pool["setup"].astype(str)
    pool = pool[pool["setup"] == setup_key].reset_index(drop=True)
    if pool.empty:
        raise RuntimeError(f"No rows in candidate_pool for setup={setup_key!r}")
    if int(max_candidates) > 0 and len(pool) > int(max_candidates):
        pool = pool.head(int(max_candidates)).reset_index(drop=True)

    # Stable hash of LF+combiner config for reproducibility.
    lf_config = {
        "setup": setup_key,
        "past_window": int(past_window),
        "future_window": int(future_window),
        "combine": combine_key,
        "label_def_version": str(label_def_version),
    }
    lf_config_hash = _sha1_json(lf_config)

    # Build context windows (teacher-only; separate artifact).
    ctx_spec = ContextWindowBuildSpec(
        past_window=int(past_window),
        future_window=int(future_window),
        feature_columns=tuple(),
        sample_meta_columns=tuple([c for c in ["source", "close", "volume", "dollar_vol"] if c in pool.columns]),
        require_full_window=True,
    )
    ctx_path = build_context_window_bars(
        pool[["ticker", "asof_date", "setup"] + list(ctx_spec.sample_meta_columns)].copy(),
        config=cfg,
        spec=ctx_spec,
        out_path=out_dir / "context_windows.parquet",
        source_pool=candidate_pool,
        cal=TradingCalendar("NYSE"),
        reuse_if_unchanged=True,
        lf_config_hash=lf_config_hash,
    )
    ctx_long = pd.read_parquet(ctx_path)
    if ctx_long.empty:
        raise RuntimeError("context window dataset is empty")

    # Build per-sample teacher features.
    feat = build_sample_features_from_context_long(ctx_long, spec=ContextSpec(past_window=int(past_window), future_window=int(future_window)))
    if feat.empty:
        raise RuntimeError("No sample features produced")

    # Apply LFs.
    lfs = make_flag_lfs() if setup_key == "flag" else make_gap_lfs(setup_key)
    lf_mat = apply_labeling_functions(feat, lfs=lfs)
    lf_mat_path = out_dir / "lf_matrix.parquet"
    lf_mat.to_parquet(lf_mat_path, index=False)
    lf_summary(lf_mat).to_parquet(out_dir / "lf_summary.parquet", index=False)
    sample_vote_stats(lf_mat).to_parquet(out_dir / "vote_stats.parquet", index=False)

    # Combine into probabilistic labels.
    label_model_path = out_dir / "label_model.json"
    if combine_key == "majority":
        post = majority_vote(lf_mat)
        label_model_path.write_text(json.dumps({"combine": "majority", "lf_config": lf_config, "lf_config_hash": lf_config_hash}, indent=2) + "\n")
    else:
        post, params = fit_independent_label_model(lf_mat)
        label_model_path.write_text(
            json.dumps(
                {
                    "combine": "label_model",
                    "lf_config": lf_config,
                    "lf_config_hash": lf_config_hash,
                    "params": json.loads(params.to_json()),
                },
                indent=2,
            )
            + "\n"
        )

    # Join back to candidate pool.
    pool2 = pool.copy()
    pool2["sample_id"] = pool2.apply(lambda r: stable_sample_id(r["ticker"], r["asof_date"], r["setup"]), axis=1)
    merged = pool2.merge(post, on="sample_id", how="inner")
    if merged.empty:
        raise RuntimeError("No pseudo labels produced after joining")

    created_ts = datetime.now(timezone.utc).isoformat()
    merged["label"] = merged["label_hard"].astype("boolean")
    merged["label_def_version"] = str(label_def_version)
    merged["lf_config_hash"] = str(lf_config_hash)
    merged["created_ts"] = created_ts

    pseudo_parquet = out_dir / "pseudo_labels.parquet"
    merged.to_parquet(pseudo_parquet, index=False)

    # CSV for immediate use with `ns models train`.
    pseudo_csv = out_dir / "pseudo_labels.csv"
    out_csv = pd.DataFrame(
        {
            "ticker": merged["ticker"].astype(str),
            "date": merged["asof_date"].astype(str),
            "setup": merged["setup"].astype(str),
            "label": merged["label"].astype("boolean"),
            "p_label": pd.to_numeric(merged["p_label"], errors="coerce"),
            "weight": pd.to_numeric(merged["weight"], errors="coerce"),
            "label_def_version": merged["label_def_version"].astype(str),
            "lf_config_hash": merged["lf_config_hash"].astype(str),
            "created_ts": merged["created_ts"].astype(str),
        }
    )
    out_csv.to_csv(pseudo_csv, index=False)

    print(f"[green]Wrote[/green] {out_dir}")
    print(f"  - pseudo labels parquet: {pseudo_parquet}")
    print(f"  - pseudo labels csv: {pseudo_csv}")
    print(f"  - lf matrix: {lf_mat_path}")


@weak_app.command("suggest-gold")
def weak_suggest_gold(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    candidate_pool: Path = typer.Option(..., "--candidate-pool", exists=True, dir_okay=False),
    model_type: str = typer.Option(..., "--model-type", help="Model type (e.g. ssl_tcn_classifier)."),
    model_run_dir: Path = typer.Option(..., "--model-run-dir", exists=True, file_okay=False, dir_okay=True),
    setup: str = typer.Option(..., "--setup", help="Setup name to sample for (must match candidate pool)."),
    window_size: int = typer.Option(96, "--window-size"),
    num_uncertain: int = typer.Option(100, "--num-uncertain"),
    num_pos_spotcheck: int = typer.Option(50, "--num-pos-spotcheck"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output CSV path for manual labeling."),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    """
    Suggest a small gold-label batch from a large candidate pool.
    """
    cfg = _config(repo_root=repo_root, duckdb_threads=duckdb_threads, window_size=int(window_size))
    pool = pd.read_parquet(candidate_pool)
    if pool.empty:
        raise RuntimeError("candidate_pool is empty")
    pool = pool.copy()
    pool["ticker"] = pool["ticker"].astype(str).str.upper()
    pool["asof_date"] = pd.to_datetime(pool["asof_date"], errors="coerce").dt.date
    pool["setup"] = pool["setup"].astype(str)
    pool = pool[pool["setup"] == str(setup)].reset_index(drop=True)
    if pool.empty:
        raise RuntimeError(f"No rows in candidate_pool for setup={setup!r}")

    # Build leakage-safe windows for scoring (decision at asof open).
    rid = _run_id()
    work_dir = cfg.paths.derived_dir / "weak_supervision" / "suggest_gold" / str(setup) / rid
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_labels = pool[["ticker", "asof_date", "setup"]].copy()
    tmp_labels["label"] = False

    spec = WindowedBuildSpec(
        window_size=int(window_size),
        feature_columns=tuple(),
        sample_meta_columns=tuple(),
        mask_current_day_to_open_only=True,
        require_full_window=True,
    )
    windowed_path = build_windowed_bars(
        tmp_labels,
        config=cfg,
        spec=spec,
        out_path=work_dir / "windowed_bars.parquet",
        source_csv=None,
        cal=TradingCalendar("NYSE"),
        reuse_if_unchanged=False,
    )
    windowed_long = pd.read_parquet(windowed_path)
    if windowed_long.empty:
        raise RuntimeError("windowed dataset is empty")

    batch = build_standard_batch_from_windowed_long(
        windowed_long,
        feature_columns=["open", "high", "low", "close", "volume"],
        window_size=int(window_size),
    )

    registry = get_default_registry()
    if model_type not in registry:
        raise typer.BadParameter(f"Unknown model_type {model_type!r}. Known: {sorted(registry)}")
    runner = registry[str(model_type)]
    art_path = model_run_dir / "trained.json"
    if not art_path.exists():
        raise typer.BadParameter(f"model_run_dir must contain trained.json; missing at {art_path}")
    artifact = TrainedArtifact(runner_name=str(model_type), path=art_path)
    preds = runner.predict([batch], artifact=artifact)
    if preds.empty:
        raise RuntimeError("No predictions produced")

    preds = preds.copy()
    preds["uncertainty"] = (preds["score"] - 0.5).abs()

    # Select uncertain and high-confidence positives.
    uncertain = preds.sort_values(["uncertainty", "score"], ascending=[True, True]).head(int(num_uncertain)).copy()
    uncertain["selection_reason"] = "uncertain"
    pos = preds.sort_values(["score"], ascending=[False]).head(int(num_pos_spotcheck)).copy()
    pos["selection_reason"] = "pos_spotcheck"
    picked = pd.concat([uncertain, pos], ignore_index=True).drop_duplicates(subset=["sample_id"], keep="first")

    out_path = out or (work_dir / "labels_to_review.csv")
    out_df = pd.DataFrame(
        {
            "ticker": picked["ticker"].astype(str),
            "date": picked["asof_date"].astype(str),
            "setup": picked["setup"].astype(str),
            "label": pd.NA,
            "model_score": pd.to_numeric(picked["score"], errors="coerce"),
            "selection_reason": picked["selection_reason"].astype(str),
        }
    )
    out_df.to_csv(out_path, index=False)
    print(f"[green]Wrote[/green] {out_path} rows={len(out_df)}")

