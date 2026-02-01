## 0) Purpose of this document

This repo already has a working self-supervised pretrainer (`ns models pretrain`) and finetune flow (`ns models train --model-type ssl_tcn_classifier`). After a small hyperparameter sweep, we observed **diminishing returns** for the current objective (masked reconstruction with Huber on per-window z-scored “shape” features): losses cluster tightly and scaling model size doesn’t reliably help.

This document proposes **next pretrainer model types** (objective + encoder family) ordered by **highest expected opportunity** for downstream “human chart-shape / pattern detection” under a **no-lookahead, realtime screener** constraint.

It is also an implementation guide: it explains how each proposed pretrainer should integrate with the repo’s existing architecture (artifacts, schema enforcement, CLI conventions, indexing, and downstream probes).

---

## 1) Repo goals and constraints (as inferred from the code)

### 1.1 Primary product goal

Per `README.md`, this is a **local single-client screener**:
- Parquet-backed storage under `data/`
- DuckDB for compute
- “scan whole market current values” style workflows (`ns screen`, derived last-100 bars, etc.)

### 1.2 ML goal

You are building an ML-assisted screener that:
- learns **shape/pattern representations** of OHLCV sequences (chart-like motifs)
- supports **fast inference** in a market-wide scan (latency matters)
- avoids **lookahead leakage** (“decision at D open” is explicit in multiple places)

### 1.3 Hard constraints (encoded in the pipeline)

- **No-lookahead / decision-time censoring**:
  - Window builder masking semantics (`WindowedBuildSpec.mask_current_day_to_open_only=True`) enforce that on the decision day you only retain open.
  - The finetune runner (`ssl/runner_classifier.py`) operates on “shape features” derived from OHLCV and expects the masking regime.
  - Practical consequence: the last timestep’s “shape features” are often undefined unless you intentionally censor it during pretrain.

- **Schema-locked pretrain → finetune**:
  - `ssl/schema.py` writes `schema.json` with fingerprint to prevent silent mismatch between pretrain and finetune.
  - Any new pretrainer must keep the schema contract stable and explicit.

- **Artifacts-first experiment tracking**:
  - Runs are stored under `data/models/<model_type>/<setup>/<run_id>/`
  - `experiment_tracking.py` builds a consolidated index (`ns models index`) that flattens JSON artifacts into columns.
  - Any new objective should write stable JSON configs/metrics so it can be compared in `_index.csv/_index.parquet`.

---

## 2) Current pretraining system (baseline)

### 2.1 What exists today

- **Pretrain command**: `ns models pretrain` in `src/screener_loader/cli.py`
- **Objective**: masked modeling reconstruction (Huber/L1) over “shape” features
- **Encoder**: `ssl/tcn.py` (TCN encoder)
- **Feature pipeline**: OHLCV → `ShapeFeatureSpec` features (`ssl/features.py`)
- **Normalization**: `per_window_zscore` (stable); robust z-score exists but has exhibited occasional native crashes when multithreaded.
- **Augmentation**:
  - random crop, jitter, and last-timestep censor augmentation (recommended for regime match)

### 2.2 Key observation from sweep

With a fixed dataset and fixed step budget, reconstruction `mean_loss` changes are small. This usually means:
- the model can already reconstruct masked values well enough
- the objective may not be strongly aligned with the downstream classification/screening transfer goal

**Implication**: the highest ROI is exploring *different self-supervision objectives* that directly reward invariances and predictive structure that matter for pattern detection.

---

## 3) How new pretrainers should integrate with the repo (architecture guide)

This section applies to all proposed model types.

### 3.1 Directory + artifact conventions (must follow)

Write runs under:

- `data/models/<model_type>/_pretrain/<run_id>/`

Include:
- `run_meta.json` (already handled by `experiment_tracking.write_run_meta`)
- `experiment.json` via `write_experiment_manifest(...)`
- `schema.json` (must include a stable fingerprint and enough info to reproduce downstream feature pipeline)
- a weights file for the encoder (e.g., `encoder.pt`)
- a config file (e.g., `pretrain_config.json`) including:
  - all hyperparameters
  - step counts
  - **aggregated training metrics** (mean loss, etc.)

Notes:
- Treat `_pretrain` as a “setup-like” partition (same idea as `_warm`). This keeps the artifact layout consistent with the rest of `data/models/` and the existing indexer (`ns models index`).
- For sweeps, prefer reusing a fixed windowed dataset for apples-to-apples comparisons (see `--reuse-windowed-from`).

### 3.2 Schema contract (must be explicit)

`ssl/schema.py` is currently oriented around:
- feature names
- normalization
- window_max / crop_lengths
- masking policy and loss type
- encoder config
- censoring regime

When adding new objectives, extend schema carefully:
- bump `schema_version` when incompatible
- add fields like `objective: "contrastive" | "cpc" | ...` and objective-specific config
- keep `feature_names`, `normalization`, and censoring semantics intact to guarantee finetune compatibility

Practical additions that should be recorded in schema/config (not just “names”):
- Augmentation **parameter ranges** (e.g., jitter sigma, scaling ranges, warp ranges, patch mask ratios).
- Any decision/censor regime configuration (see 3.3.1).
- Threading/env stability knobs used for the run (also belongs in `run_meta.json`): `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and DuckDB thread count.

### 3.3 Keep “open-only decision” consistent

Downstream is built around “decision at D open” semantics. For any pretrainer:
- enforce **prefix-consistent / causal** augmentations where relevant
- keep `augment_censor_last_timestep_prob=1.0` by default unless explicitly justified otherwise

#### 3.3.1 Decision-index randomization (recommended default)

“Censor the last timestep” is necessary, but not sufficient, because real screening/backtests may evaluate at many decision indices (not only the end of the window).

Recommended pretrain regime:
- Sample a **decision index** \(t^\*\) inside each window (per-example).
- Apply your “decision-time censor semantics” at \(t^\*\):
  - at \(t^\*\): keep only information allowed at decision-time (e.g., open-only)
  - optionally mask all timesteps \(> t^\*\) entirely (strict causality)
- For objectives with paired views (contrastive/BYOL/VICReg/Barlow), ensure **both views share the same \(t^\*\)** so the positive pair has the same available information.

This trains **prefix-robust** representations and makes “prefix probe” a first-class optimization target.

### 3.4 Evaluation must optimize transfer, not only SSL loss

Every new pretrainer should be ranked by at least one downstream probe:
- **Frozen-encoder head probe** using `ssl_tcn_classifier`:
  - train a small head for a short budget (few epochs or few thousand steps)
  - measure AUPRC (and ideally stability/latency style metrics)

Optional but strongly aligned:
- **prefix probe**: evaluate on prefixes (20%, 40%, 60%, 80%, 100% window) to measure earliest-detection capability without lookahead.

Make “screener-real” metrics first-class artifacts (recommended standard set):
- **Alerts/day/ticker @ fixed precision** (or fixed FP/day) for practical thresholding.
- **Earliest-detection curve**: prefix length → recall @ fixed FP/day (or @ fixed precision).
- **Stability under augmentations**: variance of score across N augmentations of the same window.
- **Calibration**: reliability/ECE if you intend to threshold scores for alerts.

Write probe outputs under each run directory (so `ns models index` can compare them), e.g.:
- `probe_metrics.json` (metrics above, plus config for the probe)
- `probe_predictions.parquet` (optional, but useful for audits)

### 3.5 Where to implement new objectives (recommended layout)

Add under `src/screener_loader/ssl/`:
- `contrastive.py` (loss + heads)
- `cpc.py`
- `vq.py` (tokenizer/codebook)
- `transformer.py` or `patch_transformer.py`
- `ssm.py`
- shared utilities in `ssl/augment.py` (add time-warp, amplitude scaling, feature dropout)

Then:
- add a new `ns models pretrain-*` subcommand or add `--objective <name>` to `ns models pretrain`
- write schema/config artifacts and keep indexing compatible

Backbone plug-in rule:
- Implement objectives against a small common interface (conceptually `SSLBackbone`) so we can swap backbones (TCN / PatchTransformer / SSM) without rewriting the objective logic.

---

## 4) Highest-opportunity pretrainer model types (ranked)

### 4.1 (1) Causal multi-view representation learning (InfoNCE *and* VICReg/Barlow as first-class options)

**Why it’s top opportunity**
- Your downstream wants embeddings that are **invariant** to chart-view nuisances (crop/scale/jitter/censor), not pixel-perfect reconstruction.
- Multi-view SSL trains “same underlying shape → similar embedding” directly.
- With the decision-index regime (3.3.1), this can be made genuinely **prefix-robust** and aligned with no-lookahead screening.

**Important caveat: false negatives in markets**
Classic SimCLR treats other batch items as negatives. In financial time series, many different windows are legitimately similar (repeating motifs across tickers/regimes), so “all other items are negatives” can push away true pattern neighbors.

**Mitigations (implement at least one)**
- **VICReg / Barlow Twins** option (preferred default for early iterations): avoids explicit negatives and often behaves better in repeating-signal domains.
- **Soft negatives / debiased contrastive**: down-weight penalties for very-high-similarity negatives (treat them as “don’t care”).
- **Positive expansion**: optionally treat nearby subwindows from the same ticker/time neighborhood as additional positives.

**Objective (InfoNCE variant)**
- Two views \(v_1, v_2\) of the same censored window (share the same decision index \(t^\*\))
- Encoder \(f_\theta\) + projection head \(g_\phi\)
- InfoNCE:
\[
z_i = g_\phi(f_\theta(v_i)),\quad
\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(z_1, z_2)/\tau)}{\sum_{k} \exp(\mathrm{sim}(z_1, z_k)/\tau)}
\]

**Augmentations (should match your real distortions)**
- decision-index randomization + censor (3.3.1)
- prefix-consistent crop (two different crops that do not require “future-only” context)
- mild jitter (already exists)
- amplitude scaling (global + per-feature)
- feature dropout (channel masking)
- time warp/resample (monotonic; avoid leakage)

**Implementation notes (repo integration)**
- Add `ssl/contrastive.py` with:
  - `ContrastiveConfig` (temperature, projection dim, loss_type: infonce|vicreg|barlow, etc.)
  - `ContrastiveModel` = backbone + projection head (+ optional predictor head)
- Extend `SSLSchema` to include:
  - `objective="contrastive"`
  - decision-index regime config (3.3.1)
  - full augmentation ranges

**Evaluation (required)**
- downstream probe with `probe_metrics.json` including:
  - AUPRC
  - alerts/day/ticker @ fixed precision (or FP/day)
  - prefix curve (earliest detection)
  - augmentation stability

---

### 4.2 (2) Hybrid objective: contrastive + masked reconstruction

**Why it’s high opportunity**
- Reconstruction stabilizes and keeps embeddings grounded in the input signal.
- Contrastive improves invariance and transfer.
- Hybrid objectives are frequently stronger than either alone when the downstream is discriminative.

**Objective**
\[
\mathcal{L} = \lambda \mathcal{L}_\text{contrastive} + (1-\lambda)\mathcal{L}_\text{masked-recon}
\]

**Implementation notes**
- Reuse `MaskedModelingModel` for recon and add the contrastive head (same backbone).
- Keep recon masking conservative (time-masking too high can degrade recon), and rely on contrastive views for invariances.

---

### 4.3 (3) Patch-MAE (Ti-MAE / TS-MAE style masked autoencoding)

**Why it’s a big opportunity for “shape”**
Point-wise masking often rewards local interpolation. Patch masking forces the model to reason in **segments/motifs**, which matches chart pattern semantics.

**Approach**
- Patchify the sequence (e.g., 16–64 timesteps per patch).
- Mask **30–70%** of patches (prefer contiguous spans).
- Encoder sees only visible patches; decoder reconstructs masked patches (raw features or latent).
- Optional: add a small contrastive term (ties into the hybrid theme).

**Implementation notes**
- Implement in `ssl/patch_mae.py` (or `ssl/patch_transformer.py` if sharing code):
  - patch embedding + mask generator
  - encoder backbone (TCN/Transformer/SSM via `SSLBackbone`)
  - lightweight decoder head
- Add schema fields:
  - `objective="patch_mae"`
  - patch size, mask ratio, whether reconstruction target is raw vs latent

---

### 4.4 (4) Predict-the-future in latent space (CPC / autoregressive predictive SSL)

**Why it’s high opportunity**
- Strongly aligned with realtime causality: “past context → predictive latent”.
- Often improves earliest detection in prefix probes.

**Approach**
- Use a strict causal split based on a sampled decision index \(t^\*\):
  - context = prefix up to \(t^\*\)
  - targets = one or more future segments/latents after \(t^\*\)
- Use a contrastive predictive objective (InfoNCE across candidate futures) or a distributional prediction head.

**Implementation notes**
- Add `ssl/cpc.py`:
  - segmenter (context/target definition based on \(t^\*\))
  - prediction head
  - loss with careful negative design (or negative-free variants)

---

### 4.5 (5) Negative-free / low-negative variants (BYOL / SimSiam / VICReg)

**Why**
- Useful when batch negatives are unreliable (false negatives) or batching is constrained.
- Often simpler to optimize than InfoNCE.

**Implementation notes**
- In practice, treat VICReg/Barlow as a first-class option under 4.1; BYOL/SimSiam are additional variants if needed.

---

### 4.6 (6) Discrete token objectives (VQ + masked LM) (later-stage bet)

**Why**
- Potentially strong motif vocabulary learning, but more finicky and time-consuming.

**Approach**
- Learn a codebook over patches (VQ) and predict masked patch IDs (masked LM).

---

### 4.7 (7) Backbone upgrades (Patch Transformer, SSM/Mamba-like) as plug-ins

**Why**
- Multi-scale and long-range structure can matter for nuanced patterns.

**Rule**
- Treat backbone choice as a plug-in to the objective implementation (3.5), not a separate research track.

---

### 4.8 (8) Chart-image pretraining + distillation (optional “human-shape aligned”)

**Why**
- Potentially very aligned with “human chart view,” but operationally heavy.

**Approach**
- Pretrain a vision encoder on rendered charts; distill into a lightweight 1D encoder for market-wide scanning.

---

## 5) Recommended immediate roadmap (minimal rewrites, max learning)

1. Implement **multi-view SSL** with **decision-index randomization** (3.3.1) and a **negative-safe default** (VICReg/Barlow), with InfoNCE as an optional mode.
2. Implement the **hybrid contrastive + masked recon** objective (reuse existing recon head).
3. Implement **Patch-MAE** (patch masking, not point masking).
4. Implement **CPC / predictive SSL** after the decision-index regime is in place.
5. Add a standardized **downstream probe** runner that writes `probe_metrics.json` so `ns models index` can compare transfer outcomes.
6. Operational policy: pin and record threading env vars for stability in sweeps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, DuckDB threads).

---

## 6) Practical implementation checklist (per new model type)

For each new pretrainer, ensure:
- [ ] Writes `schema.json` with fingerprint + objective config + full augmentation parameter ranges
- [ ] Writes `pretrain_config.json` with mean loss and step counts (and throughput if possible)
- [ ] Uses decision-index randomization (3.3.1) for prefix robustness (unless the objective is strictly causal by construction)
- [ ] Provides reproducibility: seed controls sampling + torch RNG, and records env/threading
- [ ] Can reuse a fixed dataset for sweeps (`--reuse-windowed-from`)
- [ ] Writes probe artifacts (`probe_metrics.json`, optional `probe_predictions.parquet`)
- [ ] `ns models index` captures configs/metrics in the flattened index

