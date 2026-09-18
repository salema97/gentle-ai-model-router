"""Dataset builder: join priors with telemetry into training rows.

Phase 2 will implement:
- feature assembly per phase run (phase id, repo/task context features,
  candidate features from the registry, effort);
- labeling: tokens per success, success/failure outcome;
- versioned dataset artifacts (parquet) keyed by registry snapshot id;
- train/eval split logic that prevents leakage across repositories.

External benchmark data is a prior; telemetry is the personalization signal.
"""
