# Training the DeBERTa ranker

> ## ⚠ BOOTSTRAP-LABEL WARNING
>
> **Every label in the current datasets is a `bootstrap_prior`, NOT ground
> truth.** They are computed from external benchmark data (Artificial
> Analysis, LMArena) through the same quality model as `router/policy.py`.
> There is (almost) no real execution telemetry yet. A ranker trained on
> these labels learns the *deterministic policy's opinion of the priors* —
> useful to validate the training pipeline and compare architectures, but
> the resulting model must NOT be trusted for production routing until
> telemetry-derived labels (`label_provenance="telemetry"`) replace the
> priors. The same warning ships in every `manifest.json`.

## Objective

Per SDD phase, score every `(model, deployment, effort)` candidate with a
DeBERTa-V3 encoder and rank by score. This is a **ranker, not a multiclass
classifier**: the project's objective is `argmin tokens_per_success` subject
to a quality floor, which is an ordering problem over a variable candidate
set.

## Input representation (hybrid, deliberate)

1. **Text branch**: `[task] … [candidate] model=…; deployment=…; effort=…
   [features] key=value; …`. Serializing numeric features into the text lets
   the encoder attend over them semantically and generalize to unseen
   feature combinations.
2. **Numeric branch**: the same feature vector, concatenated with the pooled
   `[CLS]` output into a small MLP head (`hidden → GELU → 1`). Text
   tokenization loses numeric precision; the MLP branch gives exact numeric
   gradients. Both branches feed one scalar score.

## Objectives: pointwise vs pairwise vs listwise

| Objective | Loss | When |
|---|---|---|
| `pointwise` | MSE(score, utility) | Simple baseline; scores stay interpretable. |
| `pairwise` (default) | BCE-with-logits on `s_a − s_b` | **Preferred while labels are noisy priors**: only the order matters, not the exact utility gap. Pairs are margin-filtered (default 0.05), so the model trains on confident orderings only. |
| `listwise` | future work | The dataset builder already emits full candidate lists per task group, but listwise losses are less robust to label noise. Revisit once telemetry labels exist. |

## Determinism & seeds

`seed` is fixed everywhere (`TrainingConfig.seed`, default 42; python/numpy/
torch seeds set in `training/train.py`). No randomness without a seed.

## Smoke path without heavy downloads

`tiny_deberta_config()` (in `training/model.py`) builds a random-initialized
2-layer DeBERTa-V2 — tests and dev experiments run fully offline. The
default `model_name` remains `microsoft/deberta-v3-base`.

## Commands

```bash
# 1. Build a dataset (labels are bootstrap priors — see warning above)
router build-dataset --name router-priors \
  --train-end 2026-08-01 --val-end 2026-09-01

# 2. Train (requires: uv sync --extra train)
router train --dataset data/datasets/router-priors/v1 --objective pairwise

# 3. Evaluate against references
router evaluate --dataset data/datasets/router-priors/v1 \
  --checkpoint models/deberta-router/v1
```

Checkpoints land in `models/deberta-router/v<N>/` with `config.json`,
`metrics.json`, model + tokenizer, feature schema version, dataset version,
normalization version, source snapshot ids, git commit and timestamp.
