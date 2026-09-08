# Agent 2 results handoff

Foundation used: `54afa5bc258be6e80d1be3c22f43254a61c8386a` (and the later pipeline commit already on this checkout). Implements `RunStore`, `ResultReader`, `ReviewStore`, and the Streamlit results page against frozen contracts. No catalog, runner, or dependency-file edits.

## Public symbols

| Symbol | Module | Role |
| --- | --- | --- |
| `FilesystemRunStore` | `screener_loader.vision.store` | Durable `RunStore` under an injected `scan_root` |
| `open_vision_scans(scan_root)` | `screener_loader.vision.store` | `(store, reader, reviews)` sharing one root |
| `FilesystemResultReader` | `screener_loader.vision.query` | One row per candidate; filters/paging/export |
| `QueryCounts`, `QueryExport`, `NeighborPosition` | `screener_loader.vision.query` | Extra count/export/nav types (not forked contract models) |
| `FilesystemReviewStore` | `screener_loader.vision.reviews` | Append-only reviews; optional `expected_current_id` |
| `render_results_page(reader, reviews)` | `screener_loader.vision_ui` | Mountable page; **does not** call `st.set_page_config` |
| `seed_synthetic_runs(scan_root)` | `screener_loader.vision_ui.synthetic` | Labeled fake demo data |

Widget/session keys are namespaced with `vr_` (`vision_ui.components.state_key`). Never uses the builder's `setup_id` key.

## Layout

`scan_root/<run_id>/` with:

- Authoritative: `meta.json`, `frozen.json` + frozen YAML/bytes/windows, `artifacts/**.png` + `artifacts/index.json`, `attempts/*.json`, `batches/*.json`, `results/*.json`, `reviews/*.json`
- Rebuildable: `derived/manifest.json`, `derived/counters.json`, `derived/candidates.parquet`, `derived/assessments.parquet`
- Lock: `lock` (`fcntl.flock` / portable process lock)
- Temporary `.tmp-*`, `*.tmp`, `*.partial`, and other dotfiles are ignored by readers

Artifact relative paths stay inside the run root (`artifacts/candidates/{id}.png`, `artifacts/examples/{setup}/{example}.png`). Bytes are served through `get_artifact_bytes`; the UI does not reconstruct charts.

## Query / export semantics

`ResultQuery` is the frozen filter object. Concrete extras (protocol-compatible kwargs):

- `ticker_query`: case-insensitive substring on ticker / candidate id
- `statuses`: execution-status filter used by Errors/skipped

Setup filter is ANY of the selected ids. Strength filter/sort uses match assessments in that subset, then `candidate_id` ascending. Pagination is 1-based. Export uses the same filtered sequence as the list, **all rows**, not the visible page.

Null/unit/list flattening (also shown in the UI export caption):

- One candidate row per stock
- Lists joined with `;`
- `adr_pct_20` stored fraction (`0.04 = 4%`); `adr_pct_20_display_pct` is percent
- Units: `close=USD/share`, `dollar_vol_avg_20=USD/day`, `adr_pct_20=fraction`
- Nulls stay empty; assessment export is one row per setup (+ current review)

Selection rule: if the selected candidate id remains in the filtered sequence, keep it; otherwise take the first id; empty sequence clears selection. Switching runs is the same rule against the new run's ids.

## Demo command

From the repo root, with the `ui` extra (Streamlit) installed:

```
NS_VISION_DEMO_ROOT=/tmp/ns-vision-demo \
  python -m streamlit run src/screener_loader/vision_ui/demo.py
```

The demo alone calls `st.set_page_config`. Data is synthetic and labeled as such.

## Tests actually run

```
ruff check src/screener_loader/vision/store.py src/screener_loader/vision/query.py \
  src/screener_loader/vision/reviews.py src/screener_loader/vision_ui \
  tests/test_vision_store.py tests/test_vision_ui.py
pytest tests/test_vision_store.py tests/test_vision_ui.py \
  tests/test_vision_contract.py tests/test_vision_scan.py -q
```

19 store/UI tests passed; 41 including contract/scan. Coverage includes JSON/Parquet/PNG round-trip, atomic batch recovery, idempotent vs conflicting commits, process lock, unsupported schema (no overwrite), partial/dry-run pending rows, multi-flag one-row, setup-aware strength, paging/neighbors/selection, review supersession, skip/error/unavailable diagnostics, Scanner+FakeClassifier against the real store with **zero extra classify calls** on query/detail, and Streamlit `AppTest` of the demo (Previous/Next present, no page exception).

## Visual verification

**Occurred** in a real browser at ordinary laptop width (`http://localhost:8503`, Streamlit demo). First load of the main synthetic run showed:

- Newest-first run selector, `partial · demo · synthetic`, session `2026-06-18`
- Counts: 30 candidates / 26 completed / 11 any-match / 20 setup-matches / 7 uncertain / 2 error / 2 skipped / 2 reviewed pairs
- `not eligible unavailable` (not invented)
- Default Matches view; AAPL row with **Bull Flag 3 · Episodic Pivot 2**
- Detail `1 of 11`, Previous disabled / Next enabled, persisted 8×6 synthetic chart, frozen criteria, human review separate from model output
- Page 1 of 2; full-filter CSV export buttons

A live-reload overlay intercepted a later Next click after code edits; `AppTest` still exercised Next. After escaping `$` in captions, currency rendered as `$10.00` / `$5,000,000/day` instead of disappearing into Streamlit LaTeX.

## Integration needs (Agent 3)

- Bind `FilesystemRunStore(repo_root / "data" / "vision_scans")` (or equivalent). Agent 2 does not touch `paths.py`.
- `store, reader, reviews = open_vision_scans(scan_root)`
- Wire `Scanner(..., store=store)` and `classifier.get_image=store.get_artifact_bytes`
- Mount `render_results_page(reader, reviews)` in the shared Streamlit shell. Do **not** call `set_page_config` from the page. Keep `ns setups ui` on the builder until the shell exists.
- Streamlit stays in the existing `ui` extra. PyArrow is already a base dependency. Do not add OpenAI from this work.
- Production must never silently substitute `seed_synthetic_runs` / FakeClassifier.

## Contract notes for Agent 1 (no forked types)

Please consider adding to the frozen protocol (we implemented as extras for now):

1. `ResultQuery` ticker search and execution-status filter (today: kwargs on `FilesystemResultReader.query`)
2. `ResultReader.list_runs`, `load_run`, `load_frozen_inputs`, `neighbor`, `export_query` / `extended_counts`
3. Distinct any-match / uncertain / per-setup / reviewed-pair counts (`QueryCounts`; `ResultCounts` is still implemented as specified)
4. Runner `CandidateResult.artifact_id` (`candidate:{ticker}:{date}`) may differ from `ChartArtifact.artifact_id`. The reader joins charts by `candidate_id` when the stored artifact id misses.

`recover_attempt` returns the latest usable journaled attempt for a fingerprint (accepted preferred; also unaccepted stored JSON with results/output and no error) so resume can avoid a resubmit.

## Limitations

- Storage does not retry the provider; the runner owns retries.
- Manifest/parquet exports are derived conveniences; committed JSON/PNG shards are durable before finalize.
- Identical `artifact_id` from the test `FakeRenderer` can collide example vs candidate; the store keys files by relative path so both PNGs are kept.
- Two runs created in the same minute need unique selectbox labels (now includes `run_id`).
- Demo charts are 8×6 synthetic PNGs, not matplotlib market plots.
- Keyboard shortcuts are not implemented.
- No CLI registration, shared app shell, or lockfile/README edits (Agent 3).
