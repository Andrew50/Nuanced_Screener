# Labels

`labels.csv` at the repository root is the hand-labeled source used for supervised setup experiments (`ns models train --labels-csv labels.csv`).

Generated Parquet label stores under `data/labels/` are local artifacts (gitignored). Prefer editing and committing `labels.csv`.

## Columns

| Column | Meaning |
|--------|---------|
| `ticker` | Equity symbol (normalized to uppercase on load). |
| `date` | Decision / as-of date (`YYYY-MM-DD`). Must be a NYSE trading day unless resolution options are used. |
| `setup` | Setup class identifier for that sample (string). |
| `label` | Boolean target (`True` / `False`). |
| `weight` | Optional numeric sample weight. When present, classical trainers (e.g. LightGBM / logistic regression) pass it through as `sample_weight`. |

Primary key after load: `(ticker, asof_date, setup)` — duplicates are rejected by default.

## Setup identifiers

The current file uses short internal codes, including:

`F`, `P`, `EP`, `NF`, `NP`, `NEP`, `MR`

The repository does **not** define expansions for these abbreviations. Treat them as opaque setup labels chosen by the labeler. Heuristic candidate generators elsewhere in the code use longer names such as `flag`, `gap_go`, and `gap_fade`; those are separate from the codes in `labels.csv`.

## Provenance

- Source of truth for supervised experiments: `labels.csv`
- Loader: `screener_loader.labels.load_labels_csv`
- Approximate contents of the tracked file: ~330 positive examples spanning 2020–2024 across multiple setup codes
- Weights in the file are numeric (roughly 1.5–5 in the current snapshot); the code does not assign a semantic meaning beyond sample weighting
