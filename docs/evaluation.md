# Offline evaluation

> ## ⚠ BOOTSTRAP-LABEL CAVEAT
>
> All metrics below are computed against `label_utility` values that are
> **priors from external benchmarks**, not measured task success
> (`label_provenance="bootstrap_prior"` in every dataset). Use these numbers
> to **compare routers relative to each other** and to catch regressions —
> never as absolute production-quality estimates.

## Splits

Evaluation runs on the temporally held-out splits only:

- `test` — examples whose snapshot date falls after the validation cutoff.
- `temporal_test` — examples whose candidate **model first appeared in the
  registry after the train cutoff** (new-model simulation). First appearance
  is computed from dated provenance rows (benchmark/price `source_snapshot_id`
  → snapshot `fetched_at`). Models with no dated rows fall back to the newest
  snapshot bucket — documented, conservative: they count as new models.

Random splits are forbidden: benchmark priors drift over time, and a random
split leaks the future into training (a model's later, better scores would
be visible during training).

## Anti-leakage rules (build errors, not warnings)

1. Every benchmark/price row must reference a known snapshot id (else it
   cannot be dated → `DatasetBuildError`).
2. Every row used must be dated ≤ the build's `as_of` cutoff (default:
   newest snapshot) → else `DatasetLeakageError`.
3. No train example may share a `task_id` with validation/test/
   temporal_test; overlapping train rows are **dropped** and the count is
   recorded in `manifest.json` (`dropped_train_task_overlap`).

## Ranking metrics (per split, per router)

`top1_accuracy` (pick == best-by-utility), `top3_recall` (pick within the 3
best), `MRR`, `NDCG@3`, `NDCG@5`. For single-pick routers the NDCG gain of
the predicted ordering places the pick first, then the ideal order.

## Business metrics (per split, per router)

Given per-candidate estimated tokens/cost from the dataset:

- `tokens_per_task` — mean estimated tokens of the picked candidate.
- `tokens_per_success` — total tokens / successful groups
  (success = utility ≥ phase `threshold_quality`).
- `success_rate`, `mean_quality`.
- `routing_regret` — mean(oracle_utility − picked_utility), ≥ 0 by
  construction; the single most honest number in this file.

## Reference routers (`training/baselines.py`)

| Router | Rule | Represents |
|---|---|---|
| `fixed_strong` | highest benchmark prior, cost ignored | "always buy the best" |
| `fixed_cheap` | lowest estimated cost | "always buy the cheapest" |
| `benchmark_only` | highest raw benchmark score | arena-to-workload mismatch |
| `baseline_policy` | `router/policy.py` replayed on the registry | the bar the learned ranker must beat |
| `learned_ranker` | trained checkpoint (needs `--checkpoint`) | candidate replacement |

## Commands

```bash
# Reference routers only — no checkpoint, no [train] extra needed
router evaluate --dataset data/datasets/router-priors/v1 \
  --evaluate-baselines-only --output metrics.json

# Full comparison against a trained checkpoint ([train] extra required)
router evaluate --dataset data/datasets/router-priors/v1 \
  --checkpoint models/deberta-router/v1 --output metrics.json
```

Output: rich tables to stdout + `metrics.json` with per-split, per-router
metric maps and the label-provenance caveat.
