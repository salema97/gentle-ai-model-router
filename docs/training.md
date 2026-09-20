# Training the ModernBERT ranker

> ## ⚠ LABEL PROVENANCE & BOOTSTRAP WARNING
>
> In cold-start datasets without telemetry, every label is a `bootstrap_prior`,
> NOT ground truth. They are computed from external benchmark data (Artificial
> Analysis, LMArena) through the same quality model as `router/policy.py`.
>
> With Phase 5, the retrain loop consumes real execution telemetry or bootstrap
> telemetry seeded from empirical benchmarks (`router shim seed --dataset ...`).
> The dataset, checkpoint, and evaluation harness report the **true provenance**
> via `derive_label_provenance`:
> - **Pure bootstrap**: `bootstrap_prior`
> - **Mixed**: `bootstrap_prior+telemetry` or `bootstrap_prior+empirical_benchmark+telemetry`
> - **Pure telemetry**: `telemetry`
>
> The exact provenance string is recorded in `manifest.json`, checkpoint `metrics.json`,
> and evaluation results docs.

## Objective

Per SDD phase, score every `(model, deployment, effort)` candidate with a
ModernBERT encoder and rank by score. This is a **ranker, not a multiclass
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
| `pointwise` | MSE(score, utility) with `telemetry_weight` | Simple baseline; scores stay interpretable. Telemetry loss weighting allows upweighting real execution rows over bootstrap priors. |
| `pairwise` (default) | BCE-with-logits on `s_a − s_b` | **Preferred while labels are noisy priors**: only the order matters, not the exact utility gap. Pairs are margin-filtered (default 0.05), so the model trains on confident orderings only. |
| `listwise` | future work | The dataset builder already emits full candidate lists per task group, but listwise losses are less robust to label noise. Revisit once telemetry labels exist. |

### Telemetry loss weighting (`telemetry_weight`)

In pointwise training, `TrainingConfig.telemetry_weight` (CLI `--telemetry-weight`, default 1.0)
multiplies the MSE loss contribution of `label_provenance='telemetry'` examples.
- Setting `--telemetry-weight > 1.0` prioritizes measured execution feedback over benchmark priors.
- When `telemetry_weight == 1.0` (or when no telemetry examples exist), the loss computation
  is byte-identical to standard unweighted MSE.

## Evaluation with measured token accounting

Offline evaluation (`training/evaluate.py`) computes ranking and business metrics across
held-out splits (`test`, `temporal_test`, `validation`):
- **Measured token counts**: Telemetry rows carry `actual_total_tokens` (measured).
  Business metrics (`tokens_per_task`, `tokens_per_success`) use `actual_total_tokens`
  whenever available, falling back to `est_total_tokens` for bootstrap rows.
- **Mix accounting**: Evaluator documents count `measured_token_rows` and `estimated_token_rows`
  alongside the dataset's `label_provenance`.

## Determinism & seeds

`seed` is fixed everywhere (`TrainingConfig.seed`, default 42; python/numpy/
torch seeds set in `training/train.py`). No randomness without a seed.

## Smoke path without heavy downloads

`tiny_modernbert_config()` (in `training/model.py`) builds a random-initialized
2-layer ModernBERT — tests and dev experiments run fully offline. The
default `model_name` is `answerdotai/ModernBERT-base`.

## Commands

```bash
# 1. Build a dataset with telemetry enabled & ground truth traces
router build-dataset --name router-telemetry \
  --train-end 2026-08-01 --val-end 2026-09-01 \
  --telemetry-db data/telemetry.sqlite

# 2. Train pairwise ModernBERT ranker with low memory and bf16 acceleration
router train --dataset data/datasets/router-telemetry/v1 \
  --objective pairwise \
  --device cuda \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --bf16 \
  --max-vram-fraction 0.5 \
  --progress

# 3. Evaluate against references
router evaluate --dataset data/datasets/router-telemetry/v1 \
  --checkpoint models/modernbert-router/v14
```

### High-Throughput & Low-Memory Training (RTX 50-Series / Blackwell & Ada)

To eliminate CPU bottlenecks and GPU OOM errors in resource-constrained or WSL environments:
1. **Candidate Pre-Tokenization Cache**: `precompute_pairwise_cache` tokenizes distinct candidates once into tensor memory, bypassing per-step Python tokenization loops and accelerating throughput from ~0.5 it/s to ~3.8 steps/sec (~30.4 samples/sec).
2. **Gradient Accumulation**: `--batch-size 4 --gradient-accumulation-steps 2` maintains an effective batch size of 8 while cutting peak activation memory in half (~3.3 GB peak VRAM).
3. **Bfloat16 Precision (`--bf16`)**: Uses 5th-Gen Tensor Cores on NVIDIA Blackwell GPUs with `torch.bfloat16`, preserving full dynamic range.
4. **VRAM Fraction Capping (`--max-vram-fraction 0.5`)**: Strictly bounds PyTorch CUDA allocations with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid out-of-memory crashes.

Checkpoints land in `models/modernbert-router/v<N>/` with `config.json`,
`metrics.json`, model + tokenizer, feature schema version, dataset version,
normalization version, source snapshot ids, truthful `label_provenance`, git commit, and timestamp.

## Promotion workflow: checkpoints & thresholds

> **NEVER auto-replace the active router.** Training a new checkpoint or proposing
> new thresholds changes nothing in production. Promotion is an explicit, audited decision:
> **candidate / proposals → evaluate / filter → compare → promote.**

Promotion supports two distinct artifact kinds (`checkpoint` and `thresholds`)
recorded in `models/promoted/promoted.json`.

```bash
# Candidate checkpoint with an "eval" section in metrics.json:
router promote --candidate models/modernbert-router/v2

# Or evaluate fresh on a dataset first (requires: uv sync --extra train):
router promote --candidate models/modernbert-router/v2 \
  --dataset data/datasets/router-telemetry/v1

# Decide and print the comparison, write nothing:
router promote --candidate models/modernbert-router/v2 --dry-run

# Promote an evidence-backed thresholds proposal artifact:
router promote --thresholds proposals.json

# Show the currently promoted artifact (checkpoint or thresholds):
router promote --status
```

Rules enforced by `training/promote.py`:

1. **Provenance gate.** A candidate checkpoint's `metrics.json` must carry
   `dataset.version`, `source_snapshot_ids`, and `label_provenance`; otherwise
   promotion is refused (exit 2). An untraceable checkpoint must never route
   production traffic.
2. **Eval gate.** Without `--dataset`, the checkpoint must already contain an
   `eval` section with per-split metrics in the shape `training/evaluate.py`
   produces. Train loss alone is NEVER a promotion criterion.
3. **Comparison (Checkpoints).** Primary metric: `tokens_per_success` (lower is better) on
   the newest available evaluated split (`temporal_test` > `test` >
   `validation`). **Guardrails**: `success_rate` and `mean_quality` must not
   regress beyond `--epsilon` (absolute, default 0.02). Ranking metrics
   (`ndcg@5`, `mrr`) are reported as information but do not gate promotion.
4. **Evidence (Thresholds).** Proposals from `router thresholds propose` carry
   measured sample counts and empirical success rates. Only applyable kinds (`upgrade`,
   `downgrade`) are retained in the promotion record.
5. **Decision & Record.** Promote only if the primary improves AND guardrails hold.
   First promotion (nothing promoted yet) always promotes when eval metrics
   exist. A `KEEP` decision leaves the current router active (exit 1); usage
   and validation errors exit 2.

On promotion, `models/promoted/` receives an atomic write of:
- `promoted.json` — `{artifact_kind, promoted_checkpoint | thresholds, metrics_path | evidence,
  promoted_at, git_commit, promoted_by: "manual"}`;
- `metrics.json` — a full copy of the promoted checkpoint's metrics (for checkpoint promotions).

### Serving honoring promoted artifacts

When running `router serve`, the server checks `models/promoted/promoted.json`:
- For `artifact_kind: "checkpoint"`, it automatically resolves and loads `model.quant.onnx`
  (falling back to `model.onnx`) from the promoted checkpoint directory without requiring
  manual `--ranker` CLI flags.
- If the promotion record points to a missing artifact, `router serve` fails closed with exit 2.
