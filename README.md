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

## Usage

Fetch ticker universe (NASDAQ/NYSE/AMEX) and cache to `data/meta/tickers.csv`:

```bash
ns universe --exclude-test-issues --include-exchanges NASDAQ NYSE AMEX
```

Incremental update (downloads only missing dates per ticker, tolerates partial failures):

```bash
ns update --start-date 2010-01-01 --processes 16 --batch-size 100 --pause-seconds 1.0
```

Rebuild derived “last 100 bars” (also happens at end of `update`):

```bash
ns rebuild-last100 --window-size 100
```

Run example screening queries on the derived Parquet (no pandas loading):

```bash
ns screen --query top_momentum_21d --limit 50
```

## Notes

- `data/raw/` and `data/derived/` are gitignored by default; `data/meta/` is intended to be trackable.
- The initial vendor is `yfinance` (prototype). The code is structured so you can swap to Polygon/IEX/etc. without changing storage/query layers.

