# Nuanced Screener (local): Parquet storage + DuckDB compute

This repo implements a **single-client/local** market data loader optimized for fast **“scan whole market current values”** operations.

- **Parquet is the storage format** (durable files under `data/`).
- **DuckDB is the query/compute engine** (embedded database library) that reads/writes Parquet and runs screening SQL.

## Layout

- `data/meta/tickers.csv`: cached ticker universe (from NASDAQ Trader Symbol Directory)
- `data/raw/{TICKER}.parquet`: per-ticker OHLCV history (gitignored)
- `data/derived/last_100_bars.parquet`: consolidated last-`window_size` bars per ticker (gitignored)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands (cookbook)

Help:

```bash
ns --help
ns models --help
ns models train --help
ns models pretrain --help
```

### Data: universe

```bash
ns universe --exclude-test-issues --include-exchanges NASDAQ NYSE AMEX
```

### Data: update raw OHLCV

Per-ticker mode (non-Polygon vendors):

```bash
ns update --ohlcv-vendor stooq --start-date 2010-01-01 --processes 16 --batch-size 100
```

Polygon grouped-daily date partitions (default):

```bash
ns update --ohlcv-vendor polygon_grouped --lookback-years 2 --calls-per-minute 5
```

### Data: derived datasets

Rebuild last-N bars (also runs at end of `update`):

```bash
ns rebuild-last100 --window-size 100
```

Run a named market scan on `data/derived/last_100_bars.parquet`:

```bash
ns screen --query top_momentum_21d --limit 50
ns screen --query liquid_momo_21d --limit 50
ns screen --query range_pct_today --limit 50
```

### SSL pretrain (TCN masked modeling)

Install ML deps:

```bash
pip install -e ".[dev,ml]"
```

Pretrain (unlabeled windows, uncensored OHLCV):

```bash
ns models pretrain --ticker-source universe --num-samples 50000 --window-max 96 --crop-length 64 --crop-length 96 --epochs 5 --batch-size 64 --device cpu
```

Common variations:

```bash
# Use labels-only universe (tickers from labels CSV)
ns models pretrain --ticker-source labels_only --labels-csv labels.csv --num-samples 20000

# Robust normalization
ns models pretrain --normalization per_window_robust_zscore

# Masking policy
ns models pretrain --mask-mode time --mask-rate-time 0.10
ns models pretrain --mask-mode feature --mask-rate-feat 0.15
ns models pretrain --mask-mode both --mask-rate-time 0.10 --mask-rate-feat 0.15

# Optional context feature
ns models pretrain --include-pos

# Optional: simulate inference censoring as augmentation (last timestep)
ns models pretrain --augment-censor-last-prob 0.25
```

Artifacts:

```bash
ls -1 data/models/ssl_tcn_masked_pretrain/_pretrain/
ls -1 data/models/ssl_tcn_masked_pretrain/_pretrain/<RUN_ID>/
```

### Finetune (per-setup binary head)

Train a head from labels (uses inference-time censoring via the standard window builder):

```bash
ns models train --labels-csv labels.csv --model-type ssl_tcn_classifier --setup flag --window-size 96 --encoder-dir data/models/ssl_tcn_masked_pretrain/_pretrain/<RUN_ID>
```

Common variations:

```bash
# Head hyperparams
ns models train --labels-csv labels.csv --model-type ssl_tcn_classifier --setup flag --encoder-dir data/models/ssl_tcn_masked_pretrain/_pretrain/<RUN_ID> --head-epochs 5 --head-lr 0.001 --head-hidden-dim 128

# Split policy
ns models train --labels-csv labels.csv --model-type ssl_tcn_classifier --setup flag --split time
ns models train --labels-csv labels.csv --model-type ssl_tcn_classifier --setup flag --split random --seed 1337

# Baselines
ns models train --labels-csv labels.csv --model-type dummy_constant_prior --setup flag
```

### Inspect results

Where runs are written:

```bash
ls -1 data/models/<MODEL_TYPE>/<SETUP>/
ls -1 data/models/<MODEL_TYPE>/<SETUP>/<RUN_ID>/
```

Metrics + predictions (val/test):

```bash
cat data/models/<MODEL_TYPE>/<SETUP>/<RUN_ID>/val/metrics.json
cat data/models/<MODEL_TYPE>/<SETUP>/<RUN_ID>/test/metrics.json
```

Quick peek at prediction rows:

```bash
python -c "import pandas as pd; df=pd.read_parquet('data/models/<MODEL_TYPE>/<SETUP>/<RUN_ID>/val/predictions.parquet'); print(df.head(10))"
```

## Notes

- `data/raw/` and `data/derived/` are gitignored by default; `data/meta/` is intended to be trackable.
- `data/models/` is gitignored by default (weights/artifacts can be large).
- The initial vendor is `yfinance` (prototype). The code is structured so you can swap to Polygon/IEX/etc. without changing storage/query layers.

