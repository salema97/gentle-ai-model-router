"""Training: ranker and constrained routing policy.

Phase 3+ will implement:
- a deterministic baseline policy (prior-weighted, threshold-constrained)
  before any learned model is trusted;
- a constrained bandit over (model, deployment, effort) candidates that
  minimizes tokens_per_success subject to per-phase quality floors
  (see router.yaml phase_thresholds);
- a ModernBERT-based phase/context ranker fine-tuned on telemetry-labeled rows
  (checkpoint choice pending verification in Phase 1).

Every emitted decision must carry provenance: prior vs telemetry weight,
constraint check result, and ranker score.
"""
