# Routing Policy: thresholds, minimum-sufficient effort, and calibration

How the deterministic baseline policy (`src/gentle_ai_model_router/router/policy.py`)
uses per-phase quality floors, and how to keep those floors sane as the
registry evolves.

## How `threshold_quality` drives selection

The policy ranks **(model, deployment, effort)** candidates per phase with a
prior-weighted heuristic:

1. **Benchmark prior**: per benchmark key, scores are min-max normalized
   across the candidate set and combined with the phase's configured weights
   (`PhaseConfig.weights`). Models with no weighted benchmark data fall back
   to `policy.flat_prior` and are flagged `missing_benchmark_data:using_prior`.
2. **Effort quality**: diminishing returns via
   `quality = prior + (ceiling - prior) * gain[effort]`
   (`policy.effort_quality_gain`, ceiling = `policy.effort_ceiling`).
3. **Minimum-sufficient effort**: for each (model, deployment) pair the policy
   picks the LOWEST effort whose quality **meets the phase threshold**
   (`PhaseConfig.threshold_quality`). Pairs that never meet the threshold are
   excluded; among the rest it minimizes estimated tokens, tie-broken by
   quality, then canonical id. Estimated tokens grow superlinearly with
   effort (`policy.effort_token_multiplier`), so a higher threshold directly
   buys more reasoning effort per task — and a threshold set too high
   **starves the phase**: `select_candidate` fails closed with
   "no candidate meets threshold_quality=..." instead of guessing.

Threshold semantics by phase are documented in `router/config.py`
(`DEFAULT_PHASE_THRESHOLDS`): apply/verify floors are lower because test
success is handled by escalation, not by the threshold.

## `router calibrate-thresholds` — decision support, not an auto-writer

```bash
router calibrate-thresholds                 # all 11 canonical phases
router calibrate-thresholds --phase design  # one phase
```

For each phase the command prints:

| column | meaning |
|---|---|
| `current` | the configured `threshold_quality` |
| `p25 / p50 / p75` | percentiles of the pooled achievable-quality distribution — one quality value per (candidate, effort level), computed with the policy's **own** `effort_quality` + `benchmark_priors` (imported, not reimplemented) |
| `meet %` | fraction of candidates whose BEST effort meets the current threshold |
| `suggested` | `min(max(current, p50), p90)` of the achievable distribution |

The heuristic is deliberately conservative:

- **Never suggest going below the current floor** — `max(current, p50)`. A
  threshold under the median achievable quality filters nothing; the p50 term
  only ever *raises* a floor that has become toothless as the registry
  improved.
- **Never suggest going above p90** — `min(..., p90)`. A threshold above the
  90th percentile of achievable quality fails closed for nearly every
  candidate, which is an operations bug (the phase gets no routing), not a
  quality bar.

The command is **read-only**: it never touches `router.yaml` or the registry,
and it always exits 0 (including on an unknown phase name, which is printed
as an error). Adopting a suggestion is a human decision: edit
`phases.<name>.threshold_quality` in `router.yaml`, then re-check with
`router policy --phase <name>` and `router explain --phase <name>`.

## Caveat: priors are bootstrap-quality

Everything above is computed from **min-max-normalized external benchmark
scores plus a flat prior** — the same bootstrap labels the dataset builder
uses (see the honesty rule in `docs/training.md` and the caveat in
`docs/evaluation.md`). The registry's Artificial Analysis rows come from the
free API tier, whose intelligence index is a 0–100-style scale that gets
normalized into [0, 1] within the candidate set — so percentile positions are
meaningful **relative to the current registry**, but absolute quality values
are NOT measured task success. Use `calibrate-thresholds` to spot thresholds
that filter nothing (meet % ≈ 100 and threshold < p25) or starve the phase
(meet % ≈ 0); validate any adopted change against real telemetry
(`router shim ingest` + escalation outcomes) before trusting it in production.
