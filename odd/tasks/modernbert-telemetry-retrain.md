# Feature: ModernBERT retrain on telemetry + threshold tuning + learned-policy promotion

## Objective

Roadmap phase 5: retrain the ModernBERT phase/context ranker on
telemetry-provenance examples, add telemetry-driven threshold tuning, and
generalize the promotion workflow so learned artifacts (checkpoints, tuned
thresholds) actually reach the serving path.

## Problem

Phase 4 closed the loop (decision → execution → reward → bandit) and the
dataset bridge emits `PROVENANCE_TELEMETRY` rows — but nothing retrains the
ranker on them, thresholds are static priors, and promotion is
checkpoint-only with a hardcoded served ranker path (`models/.../v9/`).

Blocker found in exploration: **`data/telemetry.sqlite` does not exist** —
zero telemetry rows in any dataset. Retraining "on telemetry" needs measured
outcomes. Bootstrap path: seed the shim from the empirical benchmark
observations already in the datasets (measured benchmark outcomes, not
fabricated numbers), clearly labeled as bootstrap telemetry.

## Why

Without retraining, the neural ranker keeps learning bootstrap-prior labels
forever. Without threshold tuning, quality floors stay static. Without
promotion wiring, a better checkpoint never reaches `/route`.

## Scope

Authorized: shim seeding from empirical data, training/eval provenance fixes,
provenance weighting, threshold tuner + router.yaml applier, promotion
generalization + serve honoring promotion record, tests, docs.

NOT in scope: auto-retrain scheduling (promotion stays `promoted_by:
"manual"` by design), listwise objective (documented future work in
train.py), changing the bandit's UCB algorithm.

## Constraints

- Fail closed; determinism (seeded, stable sorts, sha256 versions).
- Empirical-derived telemetry is labeled bootstrap and never mixed silently
  with real shim executions (different `router_version` / provenance note).
- The serving bench-feature-name set must match the trained dim (fix the
  fixed-15 default in `router/neural.py`).
- Promotion NEVER auto-replaces: explicit `--dry-run` first, guardrails
  unchanged (tokens_per_success primary, epsilon guardrails).
- Tests: `uv run pytest` green, `uv run ruff check` clean. TDD not
  configured — ordinary functional checks.

## Tasks

- [x] **R1** — Shim seeding from empirical observations: CLI `router shim
  seed --dataset <path>` converting empirical-provenance dataset examples
  into scored execution records (measured quality → outcome rubric inputs),
  stamped as bootstrap (`router_version: empirical-bootstrap`).
- [x] **R2** — Training provenance fixes: derive `label_provenance` in
  `train.py` metrics + `evaluate.py` results from the dataset instead of
  hardcoded `bootstrap_prior`; eval metric branch using
  `actual_total_tokens` for telemetry rows (measured tokens_per_success).
- [x] **R3** — Provenance weighting: `TrainingConfig.telemetry_weight` to
  upweight telemetry rows in the pointwise objective.
- [x] **R4** — Threshold tuning: pure `router/threshold_tune.py` over
  RewardAggregates (cheapest-effort frontier meeting a success-rate floor
  per phase) + explicit `router thresholds apply` writing router.yaml
  backup-first/atomic. Wire BanditConfig from RouterConfig while touching
  this.
- [x] **R5** — Promotion workflow: generalize `models/promoted/promoted.json`
  to artifact kinds (checkpoint | thresholds); `router serve` honors the
  promotion record instead of the hardcoded v9 path.
- [x] **R6** — End-to-end verification: seed shim → build-dataset with
  telemetry → retrain tiny → evaluate → promote --dry-run, all scripted in a
  test; docs (README roadmap, runbook, training.md).

## Acceptance criteria

- A retrain run consuming telemetry rows completes and its metrics.json
  reports the true mixed provenance.
- Evaluator reports tokens_per_success on measured tokens for telemetry rows.
- Threshold tuner emits evidence-backed proposals; apply is atomic + backed
  up + reversible.
- After promote, `router serve` uses the promoted checkpoint without flags.
- Full suite green; new tests per module.
