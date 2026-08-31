# Self-Supervised Learning Research Notes

Notes on the current masked-modeling pretrainer, pipeline constraints, and candidate next objectives. Nothing below is claimed as implemented unless stated under **Current approach**.

## Current approach

The repo ships a working self-supervised pretrainer (`ns models pretrain`) and a finetune path (`ns models train --model-type ssl_tcn_classifier`).

- **Objective**: masked reconstruction (Huber / L1) over per-window z-scored OHLCV “shape” features
- **Encoder**: temporal CNN (`ssl/tcn.py`)
- **Features**: OHLCV → shape features (`ssl/features.py`)
- **Normalization**: `per_window_zscore` by default (robust z-score also exists)
- **Augmentation**: random crop, jitter, optional last-timestep censor (to approximate open-only inference)
- **Schema lock**: `ssl/schema.py` writes `schema.json` with a fingerprint so pretrain and finetune stay compatible
- **Artifacts**: runs under `data/models/<model_type>/_pretrain/<run_id>/`, indexable via `ns models index`

## Constraints

Hard constraints encoded in the pipeline:

1. **Decision-time / no-lookahead censoring**  
   Supervised window building defaults to `WindowedBuildSpec.mask_current_day_to_open_only=True`: on the as-of bar, only `open` is retained; high / low / close / volume are masked. Finetune expects that regime. Pretrain itself can use uncensored windows plus censor augmentation so the last timestep is not required for shape features during reconstruction.

2. **Schema-locked transfer**  
   New objectives should extend `SSLSchema` explicitly (bump `schema_version` on incompatible changes) and keep feature names, normalization, and censor semantics clear.

3. **Artifacts-first tracking**  
   Stable JSON configs and metrics so `ns models index` can compare runs.

4. **Local-first screening**  
   Representations should support market-wide scans without a separate database service.

## Observed limitations

After a small hyperparameter sweep on the current masked-reconstruction objective (fixed dataset and step budget), reconstruction losses clustered tightly and scaling model capacity did not reliably help. That usually means the model already reconstructs masked values well enough, and the objective may not be strongly aligned with downstream setup classification / screening transfer.

Highest expected ROI is therefore exploring *different* self-supervision objectives that reward invariances and predictive structure relevant to chart-shape detection—not only local reconstruction.

## Candidate objectives

Ordered by expected opportunity for downstream pattern detection under the no-lookahead constraint. **None of these are implemented yet** unless noted above.

1. **Causal multi-view representation learning** (VICReg / Barlow Twins as a negative-safe default; InfoNCE as an option)  
   Train embeddings that are invariant to crop / scale / jitter / censor. Market series have many legitimately similar windows, so naive “all other batch items are negatives” can hurt; prefer negative-free or soft-negative variants.

2. **Hybrid: contrastive + masked reconstruction**  
   Keep reconstruction grounding while adding an invariance term.

3. **Patch-MAE style masked autoencoding**  
   Mask contiguous patches/segments rather than individual timesteps so the model must reason about motifs, not only interpolate locally.

4. **Predictive / CPC-style latent forecasting**  
   Context prefix up to a decision index predicts future latents—aligned with causal screening.

5. **Later bets**: discrete token (VQ + masked LM) objectives; backbone plug-ins (patch transformer / SSM); optional chart-image pretrain + distillation into a 1D scanner encoder.

## Evaluation methodology

When comparing pretrainers, prefer transfer probes over reconstruction loss alone:

- Downstream head metrics on held-out labeled setups (e.g. AUPRC), written as stable JSON so `ns models index` can flatten them
- Alert-rate style operating points at fixed precision, if used in screening
- Prefix / earliest-detection curves under decision-index randomization
- Augmentation stability of embeddings
- Reproducibility: seed sampling + torch RNG; record threading env (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, DuckDB threads)
- Apples-to-apples sweeps via `--reuse-windowed-from`

## Future experiments

Practical next steps that fit the existing architecture:

1. Multi-view SSL with decision-index randomization and a negative-safe default (VICReg / Barlow).
2. Hybrid contrastive + masked recon (reuse the existing reconstruction head).
3. Patch masking (Patch-MAE style).
4. Predictive SSL once decision-index sampling is first-class.
5. A small standardized probe runner that writes `probe_metrics.json` for the experiment index.

Integration expectations for any new pretrainer: write `schema.json`, `pretrain_config.json`, encoder weights, and run metadata under `data/models/<model_type>/_pretrain/<run_id>/`, and remain compatible with `ns models index`.
