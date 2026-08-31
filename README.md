# Nuanced Screener

Local market-data loader and ML screening toolkit: Parquet on disk, DuckDB for scans, and leakage-aware models for chart-shape setups.

## Overview

This is a single-machine pipeline for pulling US equity OHLCV, building derived feature tables, and training/scoring setup detectors against hand-labeled examples.

Storage and compute stay local. Vendors write Parquet; DuckDB reads those files for market-wide screens. Training uses the same window builder as inference, with decision-day bars censored to the open so models cannot peek at same-day high/low/close.

The interesting part is the ML stack on top of that data path: self-supervised TCN pretraining on OHLCV shape features, classical Stack-6 baselines (logistic regression, LightGBM, HMM regimes), and a weak-supervision path for expanding labels from pattern candidates.

## Highlights

- **Leakage-aware windows**: decision at day-open; only `open` is kept on the as-of bar (`mask_current_day_to_open_only`).
- **Vendor-swappable ingest**: Polygon grouped-daily date partitions by default; Stooq and Yahoo per-ticker paths also supported. Per-host rate limiting and retries are built in.
- **SSL + classical models**: masked TCN pretrain → finetune heads, plus LightGBM/logreg/HMM runners under one CLI and artifact layout (`data/models/<model>/<setup>/<run_id>/`).
- **Weak supervision**: labeling functions + Snorkel-style independent label model to generate pseudo-labels from candidate pools.
- **Experiment index**: `ns models index` flattens run configs/metrics into a comparable table.

**Stack:** Python 3.10+, DuckDB, Parquet/PyArrow, Typer, optional PyTorch / LightGBM / scikit-learn / hmmlearn

## Layout

```
src/screener_loader/   # package + `ns` CLI
tests/                 # unit + CLI smoke tests
data/meta/             # cached ticker universe (tracked)
data/raw/              # OHLCV parquet (gitignored)
data/derived/          # last-N bars, embeddings (gitignored)
data/models/           # training artifacts (gitignored)
labels.csv             # hand-labeled setups (~330 rows, 2020–2024)
docs/                  # design notes (pretrain roadmap)
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set POLYGON_API_KEY if using Polygon
```

Optional extras: `pip install -e ".[ml]"` (PyTorch) or `pip install -e ".[ml_classic]"` (LightGBM / sklearn / hmmlearn).

```bash
ns --help
ns universe --exclude-test-issues --include-exchanges NASDAQ NYSE AMEX
ns update --ohlcv-vendor polygon_grouped --lookback-years 2 --calls-per-minute 5
ns rebuild-last100 --window-size 100
ns screen --query top_momentum_21d --limit 50
```

Train a classical baseline (needs local OHLCV for labeled tickers):

```bash
pip install -e ".[dev,ml_classic]"
ns models train --labels-csv labels.csv --model-type lgbm_stack6 --setup flag --window-size 96
ns models scan --model-type lgbm_stack6 --run-dir data/models/lgbm_stack6/flag/<RUN_ID> --limit 50
```

SSL pretrain / finetune:

```bash
pip install -e ".[dev,ml]"
ns models pretrain --ticker-source universe --num-samples 50000 --window-max 96 --epochs 5 --device cpu
ns models train --labels-csv labels.csv --model-type ssl_tcn_classifier --setup flag \
  --window-size 96 --encoder-dir data/models/ssl_tcn_masked_pretrain/_pretrain/<RUN_ID>
```

See `ns models --help`, `ns weak --help`, and `ns candidates --help` for the full surface. Design notes for next pretrain objectives live in `docs/PRETRAIN_MODEL_ROADMAP.md`.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Smoke tests that hit Torch or LightGBM need the matching optional extras. Most data/vendor tests use fixtures or monkeypatches and do not require a live API key.

## Notes

- `POLYGON_API_KEY` is read from the environment or `.env` (never commit `.env`).
- Large datasets under `data/raw/`, `data/derived/`, and `data/models/` are gitignored on purpose.
- `labels.csv` is the source of truth for supervised setups; generated Parquet label stores are local artifacts.
